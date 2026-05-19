import json
import os
import re
from state.graph_state import AgenticState

def blueprint_mapper_node(state: AgenticState):
    print(f"\nMapping Blueprint...")
    
    materials_list = state["extracted_materials"]
    code_registry = {}

    for page in materials_list:
        for view in page.get("views", []):
            materials = view.get("materials", {})
            for mat_key, mat_value in materials.items():
                if isinstance(mat_value, dict) and "code" in mat_value and "properties" in mat_value:
                    # Capture the structural code identifier (e.g., "X-02" or "F-20")
                    schedule_code = mat_value["code"]
                    code_registry[schedule_code] = mat_value["properties"]

    print(f"Discovered schedule reference codes: {list(code_registry.keys())}")

    mapped_count = 0

    def find_code(text):
        if not isinstance(text, str):
            return None
        matches = re.findall(r'\b[A-Z]+-\d+\b', text)
        for match in matches:
            if match in code_registry:
                return match

        for key in code_registry:
            if key in text:
                return key
        return None

    for page in materials_list:
        for view in page.get("views", []):
            materials = view.get("materials", {})
            for mat_name in list(materials.keys()):
                current_value = materials[mat_name]
                
                code = None
                is_req = False

                # Scenario 1: The raw value is a flat string "Mapping Required"
                if isinstance(current_value, str) and current_value == "Mapping Required":
                    is_req = True
                    code = find_code(mat_name)
                
                # Scenario 2: The value is a structured object matching your prompt rules
                elif isinstance(current_value, dict):
                    notes_val = current_value.get("notes", "")
                    code_val = current_value.get("code", "")
                    name_val = current_value.get("name", "")
                    
                    if notes_val == "Mapping Required":
                        is_req = True
                        code = find_code(code_val) or find_code(name_val) or find_code(mat_name)
                    else:
                        code = find_code(notes_val) or find_code(code_val)

                if code and code in code_registry:
                    if isinstance(current_value, str):
                        materials[mat_name] = {
                            "code": code,
                            "notes": "Mapped from Schedule",
                            "properties": code_registry[code]
                        }
                    else:
                        materials[mat_name]["properties"] = code_registry[code]
                        materials[mat_name]["notes"] = "Mapped from Schedule"
                        if "name" in materials[mat_name]:
                            del materials[mat_name]["name"] 
                    
                    mapped_count += 1
                elif is_req:
                    target_identifier = mat_name
                    if isinstance(current_value, dict):
                        target_identifier = current_value.get("code") or current_value.get("name") or mat_name
                    print(f"Notice: Item '{target_identifier}' requires mapping but has no definition in schedules.")


    pdf_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    mapped_json_path = os.path.join(state["output_base"], "data", f"{pdf_name}_mapped_materials.json")
    os.makedirs(os.path.dirname(mapped_json_path), exist_ok=True)
    
    with open(mapped_json_path, "w", encoding="utf-8") as f:
        json.dump(materials_list, f, indent=4, ensure_ascii=False)

    return {"mapped_materials": materials_list}