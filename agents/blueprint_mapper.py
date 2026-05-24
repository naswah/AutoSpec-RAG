import json
import os
import re
from state.graph_state import AgenticState

def blueprint_mapper_node(state: AgenticState):
    print(f"\nMapping Blueprint...")
    
    materials_list = state["extracted_materials"]
    code_registry = {}

    for page in materials_list:
        views = page.get("views", []) or page.get("view", [])
        for view in views:
            materials = view.get("materials", [])
            
            if isinstance(materials, list):
                for mat_value in materials:
                    if isinstance(mat_value, dict) and "code" in mat_value:
                        schedule_code = mat_value["code"]
                        if "material" in mat_value or "item" in mat_value or ("notes" in mat_value and mat_value["notes"] != "Mapping Required"):
                            code_registry[schedule_code] = mat_value
            elif isinstance(materials, dict):
                for mat_key, mat_value in materials.items():
                    if isinstance(mat_value, dict) and "code" in mat_value:
                        schedule_code = mat_value["code"]
                        if "material" in mat_value or "item" in mat_value or ("notes" in mat_value and mat_value["notes"] != "Mapping Required"):
                            code_registry[schedule_code] = mat_value

    print(f"Discovered schedule reference codes: {list(code_registry.keys())}")

    mapped_count = 0

    def find_code(text):
        if not isinstance(text, str):
            return None
        matches = re.findall(r'\b[A-Za-z0-9]+-\d+\b', text)
        for match in matches:
            if match in code_registry:
                return match
        for key in code_registry:
            if key in text:
                return key
        return None

    for page in materials_list:
        views = page.get("views", []) or page.get("view", [])
        for view in views:
            materials = view.get("materials", [])
            
            if isinstance(materials, list):
                for mat_value in materials:
                    if isinstance(mat_value, dict):
                        notes_val = mat_value.get("notes", "")
                        code_val = mat_value.get("code", "")
                        name_val = mat_value.get("name", "")

                        if str(notes_val).strip().lower() == "mapping required":
                            code = find_code(code_val) or find_code(name_val)
                            if code and code in code_registry:
                                registry_entry = code_registry[code]
                                components = []
                                for key in ["item", "size", "material", "notes"]:
                                    val = registry_entry.get(key, "")
                                    if val and str(val).strip() not in ["", "none", "-", "Mapping Required"]:
                                        components.append(str(val).strip())
                                
                                mat_value["notes"] = ", ".join(components) if components else "Mapped from Schedule"
                                mat_value["code"] = code
                                if "name" in mat_value:
                                    del mat_value["name"]
                                mapped_count += 1

            elif isinstance(materials, dict):
                for mat_key, mat_value in materials.items():
                    if isinstance(mat_value, dict):
                        notes_val = mat_value.get("notes", "")
                        code_val = mat_value.get("code", "")
                        
                        if str(notes_val).strip().lower() == "mapping required":
                            code = find_code(code_val) or find_code(mat_key)
                            if code and code in code_registry:
                                registry_entry = code_registry[code]
                                components = []
                                for key in ["item", "size", "material", "notes"]:
                                    val = registry_entry.get(key, "")
                                    if val and str(val).strip() not in ["", "none", "-", "Mapping Required"]:
                                        components.append(str(val).strip())
                                
                                mat_value["notes"] = ", ".join(components) if components else "Mapped from Schedule"
                                mat_value["code"] = code
                                mapped_count += 1

    print(f"Successfully mapped {mapped_count} blueprint elements.")

    if mapped_count > 0:
        print("Schedules matched successfully. Cleaning up output view blocks...")
        for page in materials_list:
            if "views" in page and isinstance(page["views"], list):
                page["views"] = [
                    v for v in page["views"]
                    if "schedule" not in str(v.get("view_name", v.get("view", ""))).lower()
                ]
    else:
        print("Warning: No elements were mapped. Preserving schedule logs intact for debugger evaluation.")

    pdf_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    mapped_json_path = os.path.join(state["output_base"], "data", f"{pdf_name}_mapped_materials.json")
    os.makedirs(os.path.dirname(mapped_json_path), exist_ok=True)
    
    with open(mapped_json_path, "w", encoding="utf-8") as f:
        json.dump(materials_list, f, indent=4, ensure_ascii=False)

    return {"mapped_materials": materials_list}