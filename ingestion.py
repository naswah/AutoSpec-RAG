import os
import json
import fitz
from dotenv import load_dotenv
from groq import Groq
import base64


load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXCLUDE_KEYWORDS = [
    "accessibility", "project summary", "site plan", "electrical", "project information"
    "plumbing", "mechanical", "fire", "lighting", "power", 
    "life safety", "water piping", "sanitary", "vent piping", "cover sheet", "cover page"
]

def is_page_excluded(page):
    """
    Checks the right and bottom margins of a page for exclusion keywords.
    """
    rect = page.rect
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


# Jun page ma keyword chaina, tyo page lai matra convert garne images ma

def pdf_to_image(pdf_path, output_base):
   
    doc = fitz.open(pdf_path)
    filtered_data = []
    
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    extraction_folder = os.path.join(output_base, "data", pdf_name)
    os.makedirs(extraction_folder, exist_ok=True)
    
    print(f"Saving filtered images to: {extraction_folder}")

    for i in range(len(doc)):
        page = doc[i]
        
        if is_page_excluded(page):
            print(f"🚫 Page {i+1}: Excluded (Keyword match)")
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
        
        print(f"✅ Page {i+1}: Saved and prepared for VLM")
        
    doc.close()
    return filtered_data


# Images lai vision languuage model ma pathaune (Please note that normal llm that can only process text cannot be used.)
def filter_materials_vision(page_data):
 
    results = []
    
    for page in page_data:
        print(f"Processing Page {page['page_no']} with Vision LLM...")
        
        prompt =""" Your role is a professional construction material estimator. Analyze this architectural drawing and extract ONLY building materials used in civil  engineering and technical specs.
        Include: Material Name, exact dimensions/sizes, thickness, and any specific notes (PSI) in the 'notes' section. If nothing is specified, write 'none'.  
        **CRITICAL** There may be codes in the image.(Eg X-20, F-76) If you see the codes just add it to JSON and write "Mapping Required" in notes. Do not add the "code" section for nomal materials.
        Return the result as a JSON object with this structure:

        [
            {
                "page": "String",
                "views": [
                {
                    "view_name": "String (Eg: Front Elevation, Bathrrom Elevation, Exterior Wall etc)",
                    "materials": {
                    "Material 1": "String"{
                        "notes": "String (Provide dimension/sizes/thickness or PSI if given, else none)",
                    }
                    "Material 2": "X-02" {
                        "notes": "Mapping Required",
                    }
                    "Material 3": "String" {
                        "notes": "String (Provide dimension/sizes/thickness or PSI if given, else none)",
                    }
                .....
                    }
                {
                "view_name": "String"
                    ........
                }
                }
                ]

                "page" : "String",
                .....
            }
        ] """
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{page['image_b64']}"}
                            },
                        ],
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct", 
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            data["page_no"] = page["page_no"]
            results.append(data)
        except Exception as e:
            print(f"Error page {page['page_no']}: {e}")
    return results



def run_ingestion(pdf_path, output_base):
    print(f"\nStarting Vision Pipeline for: {os.path.basename(pdf_path)}")

    valid_pages = pdf_to_image(pdf_path, output_base)
    materials = filter_materials_vision(valid_pages)

    #Save json
    json_folder = os.path.join(os.getcwd(), "Filtered JSON")
    os.makedirs(json_folder, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_filename = f"{pdf_name}_materials.json"
    json_save_path = os.path.join(json_folder, json_filename)

    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(materials, f, indent=4, ensure_ascii=False)

    print(f"✅ Success! Filtered JSON saved to: {json_save_path}")

    return {
        "materials": materials,
        "page_count": len(valid_pages),
        "json_path": json_save_path
    }