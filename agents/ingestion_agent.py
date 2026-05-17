import os
import json
from groq import Groq
from state.graph_state import AgenticState
from tools.pdf_helpers import pdf_to_image

def ingestion_agent_node(state: AgenticState):
    print(f"\n[Agent 1: Ingestion] Extracting Blueprint Views")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    valid_pages = pdf_to_image(state["pdf_path"], state["output_base"])
    results = []
    
    prompt = """Your role is a professional construction material estimator. Analyze this architectural drawing and extract ONLY building materials used in civil engineering and technical specs.
    Include: Material Name, exact dimensions/sizes, thickness, and any specific notes (PSI) in the 'notes' section. If nothing is specified, write 'none'.   
    - 'Material 1' and 'Material 2' are the index of material names.
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

                "page": "String",
                "views":[
                {
                  "view_name" : "Schedules",
                  "materials":
                  {
                     "Material 1": "X02"{    (If code detail arises) 
                        "prpperties": {
                        "Table_column1_name": "String"
                        "Table_column2_name": "String",
                        "Table_column3_name":"String"
                        .....
                        }
                     }
                  }
                 }]
            }
        ] 
        !!! CRITICAL !!! : Collect all layout table columns dynamically inside the "properties" object.

        Example json:
        [
    {
        "page": "A1.1",
        "views": [
            {
                "view_name": "Exterior Wall Section",
                "materials": {
                    "Concrete Foundation Wall": {
                        "notes": "8 inches thick, 4000 PSI continuous pour structural grade"
                    },
                    "X-02": {
                        "notes": "Mapping Required"
                    },
                    "Fiberglass Batt Insulation": {
                        "notes": "R-19 thermal rating, unfaced, fits tight between 2x6 framing studs"
                    }
                }
            }
        ],
        "page_no": 3
    },
    {
        "page": "A5.0",
        "views": [
            {
                "view_name": "Schedules",
                "materials": {
                    "X-02": {
                        "properties": {
                            "ITEM": "KITCHEN SINK",
                            "SIZE": "24\"x18\"",
                            "MATERIAL": "Vitreous China",
                            "NOTES": "Single bowl style, wall-hung installation config"
                        }
                    },
                    "F-20": {
                        "properties": {
                            "ITEM": "FLOOR DECKING",
                            "SIZE": "1\"x4\"",
                            "MATERIAL": "Painted Wood",
                            "NOTES": "Tongue & groove joinery pattern, unpainted cedar preferred"
                        }
                    }
                }
            }
        ],
        "page_no": 12
    }
]

        """

    for page in valid_pages:
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page['image_b64']}"}}
                        ],
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct", 
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            if isinstance(data, dict) and "views" in data:
                data["page_no"] = page["page_no"]
                results.append(data)
            elif isinstance(data, list):
                for item in data:
                    item["page_no"] = page["page_no"]
                    results.append(item)
        except Exception as e:
            print(f"Error handling page {page['page_no']}: {e}")


    pdf_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    raw_json_path = os.path.join(state["output_base"], "data", f"{pdf_name}_materials.json")
    os.makedirs(os.path.dirname(raw_json_path), exist_ok=True)
    
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Raw extracted JSON successfully saved to: {raw_json_path}")

    return {"valid_pages": valid_pages, "extracted_materials": results}