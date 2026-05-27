import os
import fitz
import base64
import pdfplumber
import re
from PIL import Image
import pytesseract

EXCLUDE_KEYWORDS = [
    "accessibility", "cover page","cover sheet", "title sheet", "delta", "project summary", "site plan", "plot plan", "mechanical", "electrical plan", "project information", "plumbing plan", "mechanical notes", "mechanical plan", "fire protection", "lighting plan", "power plan", "life safety plan", "water piping", "sanitary", "specifications", "vent piping", "cover page", "building data sheet", "building code summary", "abbreviations", "symbols", "construction notes", "waste", "water supply", "plumbing calculations", "mechanical equiptments specifications", "mechanical details","electical roof plan", "plumbing general notes and sheet index", "water supply", "plumbing", "gas floor plan", "cover sheet and index of drawings", "elec"
]

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def is_page_excluded(page):
    rect = page.rect
    width, height = rect.width, rect.height
    
    zones = [
        # fitz.Rect(0, height * 0.90, width, height),   #bottom title block
        fitz.Rect(width * 0.80, 0, width, height)  #right title block
    ]
    
    for i, zone in enumerate(zones):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=zone)
        
        img_data = pix.tobytes("png")
        from io import BytesIO
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
                zone_name = "Bottom 20%" if i == 0 else "Right 20%"
                print(f"[OCR MATCH] Page flagged! Keyword '{keyword}' visually found in {zone_name}.")
                return True
                
    return False


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
            
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image_filename = f"page_{i+1}.png"
        image_path = os.path.join(extraction_folder, image_filename)
        pix.save(image_path)
        
        img_bytes = pix.tobytes("png")
        b64_string = base64.b64encode(img_bytes).decode('utf-8')
        
        filtered_data.append({
            "page_no": i + 1,
            "image_b64": b64_string,
            "local_path": image_path
        })
        print(f"Page {i+1}: Saved and converted to b64.")
        
    doc.close()
    return filtered_data


def clean_masterformat(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            h = p.height
            crop = p.within_bbox((0, h*0.08, p.width, h*0.92))
            t = crop.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


def chunk_masterformat(text):
    pattern = r"(\d{2}\s\d{2}\s\d{2}(?:\.\d{2})?)\s+([^\n\d\.]+)(.*?)(?=\s\d{2}\s\d{2}\s\d{2}(?:\.\d{2})?|\Z)"
    chunks = []
    
    for m in re.finditer(pattern, text, re.DOTALL):
        content_body = m.group(0).strip()
        chunks.append({
            "code": m.group(1).strip(),
            "title": m.group(2).strip(),
            "content": content_body,
            "type": "content"
        })
    return chunks