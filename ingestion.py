import os
import json
from time import time
import fitz
import cv2
import pytesseract
from ultralytics import YOLO
from dotenv import load_dotenv
from groq import Groq
import time

load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def pdf_to_image(pdf_path, output_base):
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    folder = os.path.join(output_base, name)
    os.makedirs(folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(300 / 72, 300 / 72)

    for i in range(len(doc)):
        pix = doc[i].get_pixmap(matrix=matrix)
        output_path = os.path.join(folder, f"{name}_page_{i+1}.png")
        pix.save(output_path)

    doc.close()
    return folder

# pdf lai image ma lageko haru lai yolo detection model ma pass garne

def yolo_detection(model_path, image_folder):
    model = YOLO(model_path)
    
    detections_path = os.path.join(image_folder, "detections")
    os.makedirs(detections_path, exist_ok=True)

    results = model.predict(source=image_folder, imgsz=1024, conf=0.25)

    for result in results:
        filename = os.path.basename(result.path)
        save_path = os.path.join(detections_path, filename)
        
        annotated_frame = result.plot()
        cv2.imwrite(save_path, annotated_frame)
        
    print(f"✅ Annotated images saved to: {detections_path}")
    return results, model



# Extract the text from detected regions from yolo using ocr
def run_ocr(results, model):
    all_pages = []

    for i, result in enumerate(results):
        image = cv2.imread(result.path)
        page = {"page_no": i + 1, "detections": []}

        for box in result.boxes:
            cls = model.names[int(box.cls[0])].lower()

            if cls in ["text", "textbox", "text-box"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = image[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray).strip()

                if text:
                    page["detections"].append(text)

        all_pages.append(page)

    return all_pages


#OCR bata extract gareko data lai json ma stotre garne

def save_ocr_json(ocr_data, pdf_path, output_base):
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    project_root = os.getcwd() 
    ocr_folder = os.path.join(project_root, "OCR_Results")
    os.makedirs(ocr_folder, exist_ok=True)

    output_path = os.path.join(ocr_folder, f"{name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, indent=4, ensure_ascii=False)

    print(f" Raw OCR saved to root folder → {output_path}")
    return output_path


#Filter out the structured JSON

def filter_materials(ocr_json_path):
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)


    prompt = f"""
You are a professional construction material estimator.
Analyze the following OCR data extracted from an architectural house plan.

TASK:
Filter the text to extract ONLY building materials.
Examples: lumber, shingles, concrete, insulation, drywall, windows, doors, fixtures, steel, plywood.
Ignore: dimensions alone, room labels, drawing titles, revision notes, architect stamps.

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown, no backticks, no explanation.
{{
    "document": "{os.path.basename(ocr_json_path)}",
    "materials_found": [
        {{
            "item": "Material Name",
            "specs": "Details like size, grade, or type if mentioned",
            "page": <page number as integer>
        }}
    ]
}}

OCR DATA:
{json.dumps(ocr_data)}
"""

    print(f"Sending to Gemini: {os.path.basename(ocr_json_path)} ...")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.2,
        )

        raw_text = chat_completion.choices[0].message.content.strip()
        # Clean potential markdown if Groq includes it
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        materials = json.loads(raw_text)

        output_path = ocr_json_path.replace(".json", "_materials.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(materials, f, indent=4, ensure_ascii=False)

        print(f"Materials saved → {output_path}")
        return materials

    except Exception as e:
        print(f"Groq error: {e}")
        return None



def run_ingestion(pdf_path, output_base, model_path):
    print(f"\nStarting pipeline for: {os.path.basename(pdf_path)}")

    image_folder = pdf_to_image(pdf_path, output_base)
    results, yolo_model = yolo_detection(model_path, image_folder)
    ocr_data = run_ocr(results, yolo_model)

    ocr_json_path = save_ocr_json(ocr_data, pdf_path, output_base)

    materials = filter_materials(ocr_json_path)

    return {
        "ocr_data": ocr_data,
        "materials": materials,
        "ocr_json_path": ocr_json_path,
    }

