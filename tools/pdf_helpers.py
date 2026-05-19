import os
import fitz
import base64
import pdfplumber
import re

EXCLUDE_KEYWORDS = [
    "accessibility", "cover", "title sheet", "delta sheet", "cover sheet", "project summary", "site plan", "elctric", "electrical", "project information", "plumbing", "mechanical", "fire", "lighting", "power", "life safety", "water piping", "sanitary", "specifications", "vent piping", "cover page", "building data sheet"
]

def is_page_excluded(page):
    rect=- page.rect
    width, height = rect.width, rect.height
    zones = [
        fitz.Rect(0, height * 0.85, width, height),   
        fitz.Rect(width * 0.85, 0, width, height)
    ]
    for zone in zones:
        text = page.get_text("text", clip=zone).lower()
        if any(keyword in text for keyword in EXCLUDE_KEYWORDS):
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
    pattern = r"(\d{2} \d{2} \d{2})\s+([^\n]+)(.*?)(?=\n\d{2} \d{2} \d{2}|$)"
    chunks = []
    for m in re.finditer(pattern, text, re.DOTALL):
        chunks.append({
            "code": m.group(1),
            "title": m.group(2),
            "content": m.group(0),
            "type": "content"
        })
    return chunks