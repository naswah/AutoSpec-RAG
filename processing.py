import pdfplumber
import re

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