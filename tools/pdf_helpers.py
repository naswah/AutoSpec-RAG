import os
import fitz
import base64
import re
from io import BytesIO
from PIL import Image
import pytesseract
from config import OCR_PATH

EXCLUDE_KEYWORDS = [
    "accessibility", "cover page","cover sheet", "title sheet", "delta", "project summary", "site plan", "plot plan", "mechanical", "electrical plan", "project information", "plumbing plan", "mechanical notes", "mechanical plan", "fire protection", "lighting plan", "power plan", "life safety plan", "water piping", "sanitary", "specifications", "vent piping", "cover page", "building data sheet", "building code summary", "abbreviations", "symbols", "construction notes", "waste", "water supply", "plumbing calculations", "mechanical equiptments specifications", "mechanical details","electical roof plan", "plumbing general notes and sheet index", "water supply", "plumbing", "gas floor plan", "cover sheet and index of drawings", "elec",
]

pytesseract.pytesseract.tesseract_cmd = OCR_PATH

def is_page_excluded(page):
    rect = page.rect
    width, height = rect.width, rect.height
    
    zones = [
        fitz.Rect(0, height * 0.85, width, height),   #bottom title block
        #fitz.Rect(width * 0.85, 0, width, height)  #right title block
    ]
    
    for i, zone in enumerate(zones):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=zone)
        
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        
        raw_text = pytesseract.image_to_string(img).lower()
        
        zone_clean = " ".join(raw_text.split())
        dense_zone_text = zone_clean.replace(" ", "")
        
        if not zone_clean.strip():
            continue

        for keyword in EXCLUDE_KEYWORDS:
            kw_clean = keyword.lower()
            kw_dense = kw_clean.replace(" ", "")
            
            if kw_clean in zone_clean or kw_dense in dense_zone_text:
                zone_name = "Right 15%" if zone.x0 > 0 else "Bottom 15%"
                
                print(f"[OCR MATCH] Page flagged! Keyword '{keyword}' visually found in {zone_name}.")
                return True
                
    return False


def is_schedule_page(page):
    
    text = page.get_text() or ""
    if not text.strip():
        # No embedded text layer (scanned page) - fall back to a full-page OCR pass.
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
    return "schedule" in text.lower()


def pdf_to_image(pdf_path, output_base):
    doc = fitz.open(pdf_path)
    filtered_data = []
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    extraction_folder = os.path.join(output_base, "data", "output_images", pdf_name)
    os.makedirs(extraction_folder, exist_ok=True)
    
    print(f"Saving filtered images to: {extraction_folder}")
    for i in range(len(doc)):
        page = doc[i]
        
        if is_page_excluded(page):
            print(f"🚫 Page {i+1}: Excluded due to keyword boundary match.")
            continue

        schedule_flag = is_schedule_page(page)

        render_dpi = 300 if schedule_flag else 150
        target_zoom = render_dpi / 72
        rect = page.rect
        target_width = rect.width * target_zoom
        target_height = rect.height * target_zoom

        if target_width > 8000 or target_height > 8000:
            scale_down_factor = 8000.0 / max(target_width, target_height)
            final_zoom = target_zoom * scale_down_factor
            print(f"⚠️Page {i+1} exceeds 8000px at {render_dpi} DPI. Dynamically scaling down to fit within constraints.")
        else:
            final_zoom = target_zoom

        pix = page.get_pixmap(matrix=fitz.Matrix(final_zoom, final_zoom))
        image_filename = f"page_{i+1}.png"
        image_path = os.path.join(extraction_folder, image_filename)
        pix.save(image_path)
        
        img_bytes = pix.tobytes("png")
        b64_string = base64.b64encode(img_bytes).decode('utf-8')
        
        filtered_data.append({
            "page_no": i + 1,
            "image_b64": b64_string,
            "local_path": image_path,
            "is_schedule": schedule_flag,
        })
        print(f"Page {i+1}: Saved and converted to b64.{' [SCHEDULE PAGE]' if schedule_flag else ''}")
        
    doc.close()
    return filtered_data


#Plan-type detection (floor plans / elevations)
MIN_PLAN_COUNT = 2

_LEVEL = r"(1st|2nd|3rd|[4-9]th|first|second|third|fourth|fifth|ground|basement|main|upper|lower|loft|attic|roof|top)"
FLOOR_RE = re.compile(_LEVEL + r"(floor|level)")

_DIRECTION = r"(front|rear|back|left|right|north|south|east|west|side)"
ELEVATION_DIR_RE = re.compile(_DIRECTION + r"(?:side)?elevation")

# Title block zones as fractions of the page: (x0, y0, x1, y1)
BOTTOM_ZONE = (0.0, 0.85, 1.0, 1.0)
RIGHT_ZONE = (0.85, 0.0, 1.0, 1.0)


def _ocr_zone(page, frac, rotate=0):
    """OCR one zone of the page. Returns whitespace/punctuation-stripped lowercase text."""
    rect = page.rect
    x0, y0, x1, y1 = frac
    zone = fitz.Rect(rect.width * x0, rect.height * y0, rect.width * x1, rect.height * y1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=zone)
    img = Image.open(BytesIO(pix.tobytes("png")))
    if rotate:
        img = img.rotate(rotate, expand=True)  # for vertically printed title blocks
    raw = pytesseract.image_to_string(img).lower()
    return re.sub(r"[^a-z0-9]", "", raw)


def _find_plans(dense_texts):
    """Search each zone's text separately (so words don't join across zones) and merge the results."""
    levels, directions, has_elevation_word = set(), set(), False
    for dense in dense_texts:
        levels |= {m.group(1) for m in FLOOR_RE.finditer(dense)}
        directions |= {m.group(1) for m in ELEVATION_DIR_RE.finditer(dense)}
        has_elevation_word = has_elevation_word or "elevation" in dense

    levels = {{"1st": "first", "2nd": "second", "3rd": "third"}.get(l, l) for l in levels}
    directions = {"rear" if d == "back" else d for d in directions}
    if not directions and has_elevation_word:
        directions = {"generic"}
    return levels, directions


def _scan_title_blocks(page):
    # Pass 1: bottom strip and right strip, text as printed
    texts = [_ocr_zone(page, BOTTOM_ZONE), _ocr_zone(page, RIGHT_ZONE)]
    levels, directions = _find_plans(texts)
    if levels or directions:
        return levels, directions

    # Pass 2 (only if pass 1 found nothing): title blocks on the right are often printed vertically, so retry the right strip rotated both ways.
    texts = [_ocr_zone(page, RIGHT_ZONE, rotate=90), _ocr_zone(page, RIGHT_ZONE, rotate=270)]
    return _find_plans(texts)


def count_floor_and_elevation_plans(pdf_path):
    doc = fitz.open(pdf_path)
    floor_count, elevation_count = 0, 0
    details = []

    for i in range(len(doc)):
        levels, directions = _scan_title_blocks(doc[i])

        floor_count += len(levels)
        elevation_count += len(directions)

        if levels or directions:
            details.append({"page": i + 1, "floors": sorted(levels), "elevations": sorted(directions)})
            print(f"[PLAN COUNT] Page {i+1}: floors={sorted(levels)}, elevations={sorted(directions)}")

    doc.close()
    return {
        "floor_plans": floor_count,
        "elevation_plans": elevation_count,
        "total": floor_count + elevation_count,
        "details": details,
    }


def is_valid_architectural_pdf(pdf_path):
    result = count_floor_and_elevation_plans(pdf_path)
    print(f"[PLAN COUNT] Floor plans: {result['floor_plans']}, "
          f"Elevations: {result['elevation_plans']}, Total: {result['total']}")
    return result["total"] >= MIN_PLAN_COUNT, result