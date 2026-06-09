import os
import json
import re
from google import genai
from google.genai import types
from state.graph_state import AgenticState
from tools.pdf_helpers import pdf_to_image

def ingestion_agent_node(state: AgenticState):
    print(f"\n[Agent 1: Ingestion] Extracting Blueprint Views")
    client = genai.Client()
    
    valid_pages = pdf_to_image(state["pdf_path"], state["output_base"])
    results = []
    
    prompt = """Your role is a professional construction material estimator. Analyze this architectural drawing and extract building materials used in CIVIL ENGINEERING and structural construction materials, specifications, and schedule references. 

    🚨 EXTRACT ONLY CIVIL ENGINEERING MATERIALS

    What NOT to extract for "materials":
    DO NOT extract names of rooms, spatial zones, structural spaces, or architectural assemblies (e.g., "Front Porch", "Open Deck", "Balcony", "Primary Bedroom", "Staircase", "Living Room", "Walk-in Closet"). in materials feild. Do not extract the furnitures (refrigerators, sink, etc), equiptments, Cabinets and Decorative architectual elements. If there are no civil materials, then skip the view.

    WHAT TO EXTRACT INSTEAD:
    Only extract actual physical specification materials or products. For example:
    - If a note mentions an exterior wall made of "8' Concrete Foundation Wall, 4000 PSI", extract "Concrete" or "Foundation Wall Assembly Specs". 
    - Instead of extracting "Front Porch", look for specific material callouts inside that porch zone (e.g., "Pressure Treated Southern Yellow Pine", "CMU Block foundation", "Cast-in-place Concrete Slab").
    - Instead of extracting "Interior Partition Walls", look for the actual materials: "5/8" Type X Gypsum Board", "2x4 Wood Studs", or "Light-Gauge Metal Stud Framing".

   🚨 REGEX RULE FOR CODES (CRITICAL)
    For codes (e.g., X-02, F-60, F-62, W1, F1, R2, etc), go automatically to category B.
   
   🚨CRITICAL🚨: LEGEND EXCLUSION RULE:
   - IF YOU SEE STANDALONE KEY NOTES, GENERAL LEGENDS, OR INDEX KEYS LOCATED ON THE SAME PAGE AS PLAN VIEWS, YOU MUST IGNORE THEM. Do not parse these static master legends as views or schedules.
   - If there are no marks, then do not add them. It is not compulsary.

    #CATEGORY CLASSIFICATION RULES

    ## CATEGORY A: STANDARD MATERIALS (No Codes Present)
    Use this formatting ONLY if there is absolutely no schedule code (like F-60 or X-02) associated with the material.
    - Format: Provide "name" and "notes". Do NOT include a "code" key.
    - Example:
      "Material 1": {
         "name": "OSB Board",
         "notes": "2' 4x5 in size"
      }

    ## CATEGORY B: CODED MATERIALS & SCHEDULES (Codes Present)
    Use this formatting if a code (e.g., F-60, X-02) is detected anywhere on the drawing or inside a schedule layout.
    
    1. If it's on a plan view/detail pointing to a layout area:
       - You MUST strip out the "name" key completely. 
       - Provide ONLY the "code" key and set "notes" strictly to "Mapping Required".
       - Example (If you see "Hardwood Floor F-60" or just "F-60"):
         "Material 2": {
            "code": "F-60",
            "notes": "Mapping Required"
         },
         "Material 3": {
            "code": "R1",
            "notes": "Mapping Required"
         }
       - But is you just see numbers like 2036 or 1 then do not consider it as material. Ignore that.

    2. DETECTING FULL SCHEDULES & TABLES (e.g., MATERIALS SCHEDULE, FIXTURE & EQUIPMENT SCHEDULE):
       - If the page contains large master index tables ("MATERIALS SCHEDULE" or "FIXTURE & EQUIPMENT SCHEDULE"), you MUST extract EVERY single row systematically.
       - Write the colums of the table as keys for schedule and their respctive values in the values.
       - Do NOT ignore tables just because columns are empty or contain dashes ("-"). Empty values or dashes are structurally valid data points!
       - Map row column values directly to matching flat lowercase keys (e.g., "MARK" becomes "code", "ITEM" becomes "item", "MATERIAL" becomes "material", "NOTES" becomes "notes").
       - ⚠️ CRITICAL TOKEN SAVING FILTER RULE: If a table cell column value is blank, empty, contains a dash ("-"), or equals "none", completely OMIT that key from the item object entirely. Do not generate empty metadata fields.
       - Example row map:
         {
            "code": "F-20",
            "item": "FLOOR DECKING",
            "size": "1x4",
            "material": "PAINTED WOOD",
            "notes": "TONGUE & GROOVE, UNPAINTED CEDAR PREFERRED"
         }
        Here, the keys must be dynamically generated i.e. the table columns must be the keys and their respective values are the values in the table.

    ### FILTERING & EXTRACTION RULES:
    
    - If a note row or cell is blank or contains a dash ("-"), represent its value as "none" inside the properties object. Do NOT skip the row!
    - If the image is crossed out, ignore it entirely.
    - Multi-ply framing notation MUST preserve the dash prefix. If text says "3-2x12" (meaning three plies of 2x12), it is a fatal error to omit the "3-" or rewrite it as "5x12". Extract it exactly as "3-2x12".
    - Final Output must be a JSON

    ### EXPECTED OUTPUT STRUCTURE
    
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
                "code": "F1",
                "notes": "Mapping Required"
              }
            }
          },
          {
            "view_name": "Schedules",
            "materials": [
              {
                "code": "F-20",
                "item": "HARDWOOD FLOORING",
                "size": "1x4",
                "material": "Select Red Oak",
                "notes": "TONGUE & GROOVE"
              },
              {
                "code": "F-64",
                "item": "WALL TILE",
                "size": "4' square",
                "material": "Ceramic",
                "notes": "white or light blue"
              },
              ...
            ]
          }
          {
          "view_name": "Construction Assembly Notes",
            "materials": [
              {
                "code": "F1",
                "notes": "4" CONCRETE SLAB 6 MIL POLY AIR BARRIER MIN 6" COMPACTED GRAVEL BASE"
              },
              ...
            ]
           }
          }
        ]
      }
    ]

   Return ONLY the valid JSON object.
    """

    chosen_model = "gemini-3-flash-preview"

    for page in valid_pages:
        print(f"Processing blueprint page {page['page_no']} through Gemini Multimodal API...")
        try:
            image_part = {
                'inline_data': {
                    'mime_type': 'image/png',
                    'data': page['image_b64']
                }
            }

            response = client.models.generate_content(
                model=chosen_model,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    system_instruction="You are a strict technical drawing extraction engine. You must output valid raw JSON data blocks only.",
                    response_mime_type="application/json"
                ),
            )

            raw_text = response.text.strip()

            if not raw_text:
                print(f"[Page {page['page_no']}] WARNING: Empty response received from API. Skipping.")
                continue

            data = json.loads(raw_text)
              
            if isinstance(data, dict) and "views" in data:
                data["page_no"] = page["page_no"]
                results.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item["page_no"] = page["page_no"]
                        results.append(item)
                          
        except json.JSONDecodeError as e:
            print(f"[Page {page['page_no']}] JSON parse error: {e}")
            print(f"[Page {page['page_no']}] Raw response was:\n{raw_text[:500]}\n")
        except Exception as e:
            print(f"[Page {page['page_no']}] Unexpected error: {e}")

    pdf_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    raw_json_path = os.path.join(state["output_base"], "data", f"{pdf_name}_materials.json")
    os.makedirs(os.path.dirname(raw_json_path), exist_ok=True)
    
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Raw extracted JSON successfully saved to: {raw_json_path}")

    return {"valid_pages": valid_pages, "extracted_materials": results}