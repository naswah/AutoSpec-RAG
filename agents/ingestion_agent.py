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
    
    prompt = """Your role is a professional construction material estimator. Analyze this architectural drawing and extract building materials used in CIVIL ENGINEERING and technical specs.

   🚨 REGEX RULE FOR CODES (CRITICAL)
    For codes (e.g., X-02, F-60, F-62, F-64, F-76, W-01), go automatically to category B.
   
   🚨CRITICAL🚨: LEGEND EXCLUSION RULE:
   - IF YOU SEE STANDALONE KEY NOTES, GENERAL LEGENDS, OR INDEX KEYS LOCATED ON THE SAME PAGE AS PLAN VIEWS, YOU MUST IGNORE THEM. Do not parse these static master legends as views or schedules.

    ### CATEGORY CLASSIFICATION RULES

    #### CATEGORY A: STANDARD MATERIALS (No Codes Present)
    Use this formatting ONLY if there is absolutely no schedule code (like F-60 or X-02) associated with the material.
    - Format: Provide "name" and "notes". Do NOT include a "code" key.
    - Example:
      "Material 1": {
         "name": "Fiberglass Batt Insulation",
         "notes": "R-19 thermal rating, unfaced"
      }

    #### CATEGORY B: CODED MATERIALS & SCHEDULES (Codes Present)
    Use this formatting if a code (e.g., F-60, X-02) is detected anywhere on the drawing or inside a schedule layout.
    
    1. If it's on a plan view/detail pointing to a layout area:
       - You MUST strip out the "name" key completely. 
       - Provide ONLY the "code" key and set "notes" strictly to "Mapping Required".
       - Example (If you see "Hardwood Floor F-60" or just "F-60"):
         "Material 2": {
            "code": "F-60",
            "notes": "Mapping Required"
         }

    2. DETECTING FULL SCHEDULES & TABLES (e.g., MATERIALS SCHEDULE, FIXTURE & EQUIPMENT SCHEDULE):
       - If the page contains large master index tables (such as Sheet A5.0 containing "MATERIALS SCHEDULE" or "FIXTURE & EQUIPMENT SCHEDULE"), you MUST extract EVERY single row systematically.
       - Do NOT ignore tables just because columns are empty or contain dashes ("-"). Empty values or dashes are structurally valid data points!
       - Capture the columns exactly as keys inside a dynamic "properties" block. Use the column headers as your JSON keys.
       - Example:
         "Material 1": {
            "code": "F-20",
            "properties": {
               "MARK": "F-20",
               "ITEM": "FLOOR DECKING",
               "SIZE": "1x4",
               "MATERIAL": "PAINTED WOOD",
               "NOTES": "TONGUE & GROOVE, UNPAINTED CEDAR PREFERRED",
               "MANUFACTURER / MODEL": "none"
            }
         }
        Here, the "properties" block must be dynamically generated i.e. the table columns must be the keys and their respective values are the values in the table.

    ### FILTERING & EXTRACTION RULES:
    
    - If a note row or cell is blank or contains a dash ("-"), represent its value as "none" inside the properties object. Do NOT skip the row!
    - If the image is crossed out, ignore it entirely.
    - Collect all layout table columns dynamically inside the "properties" object for schedules.

    ### EXPECTED OUTPUT STRUCTURE
    Strictly match this JSON format:
    [
      {
        "page": "String",
        "views": [
          {
            "view_name": "String (e.g., Kitchen / Water Closet - Floor Plan Detail)",
            "materials": {
              "Material 1": {
                "name": "Concrete Foundation Wall",
                "notes": "8', 4000 PSI"
              },
              "Material 2": {
                "code": "F-60",
                "notes": "Mapping Required"
              },
              "Material 3": {
                "code": "X-02",
                "notes": "Mapping Required"
              }
            }
          },
          {
            "view_name": "Schedules",
            "materials": {
              "Material 1": {
                "code": "F-60",
                "properties": {
                  "MARK": "F-20",
                  "ITEM": "HARDWOOD FLOORING",
                  "SIZE": "1x4",
                  "MATERIAL": "Select Red Oak",
                  "NOTES": "TONGUE & GROOVE",
                  "MANUFACTURER / MODEL": "none"
                }
              }
            }
          }
        ]
      }
    ]

    Do not output any introductory or concluding text. Return ONLY the valid JSON object.
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