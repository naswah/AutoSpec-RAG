import json
import os
import re
from state.graph_state import AgenticState

def blueprint_mapper_node(state: AgenticState):
    print(f"\nMapping Blueprint Reference Codes...")
    
    materials_list = state["extracted_materials"]
    code_registry = {}

    # Step 1: Discover and register all schedules and assembly types (F-02, W1, R1, etc.)
    for page in materials_list:
        views = page.get("views", []) or page.get("view", [])
        for view in views:
            materials = view.get("materials", [])
            
            # If materials is structured as a list of rows/assemblies
            if isinstance(materials, list):
                for mat_value in materials:
                    if isinstance(mat_value, dict) and "code" in mat_value:
                        schedule_code = str(mat_value["code"]).strip()
                        # Ensure it's a source entry and not a target placeholder requiring mapping
                        if str(mat_value.get("notes", "")).strip().lower() != "mapping required":
                            code_registry[schedule_code] = mat_value
                            
            # If materials is structured as a dictionary object map
            elif isinstance(materials, dict):
                for mat_key, mat_value in materials.items():
                    if isinstance(mat_value, dict) and "code" in mat_value:
                        schedule_code = str(mat_value["code"]).strip()
                        if str(mat_value.get("notes", "")).strip().lower() != "mapping required":
                            code_registry[schedule_code] = mat_value

    print(f"Discovered schedule reference codes: {list(code_registry.keys())}")

    mapped_count = 0

    def find_code(text):
        if not isinstance(text, str):
            return None
        # Robust regex capturing both alphanumeric hyphen entries (F-02) and letter-digit codes (W1, R2, F12)
        matches = re.findall(r'\b[A-Za-z]+-\d+\b|\b[A-Za-z]+\d+\b', text)
        for match in matches:
            if match in code_registry:
                return match
        for key in code_registry:
            if key in text:
                return key
        return None

    # Step 2: Resolve layout callouts marked with "Mapping Required"
    for page in materials_list:
        views = page.get("views", []) or page.get("view", [])
        for view in views:
            materials = view.get("materials", [])
            
            # Case A: Materials collection structured as a list
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
                                
                                # Uniformly collect data from structural assembly blocks or tabular schedules
                                for key in ["item", "size", "material", "notes", "specification"]:
                                    val = registry_entry.get(key, "")
                                    if val and str(val).strip() not in ["", "none", "-", "Mapping Required"]:
                                        components.append(str(val).strip())
                                
                                mat_value["notes"] = "; ".join(components) if components else "Mapped from Assembly Schedule"
                                mat_value["code"] = code
                                if "name" in mat_value:
                                    del mat_value["name"]
                                mapped_count += 1

            # Case B: Materials collection structured as a key-value dictionary object
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
                                
                                for key in ["item", "size", "material", "notes", "specification"]:
                                    val = registry_entry.get(key, "")
                                    if val and str(val).strip() not in ["", "none", "-", "Mapping Required"]:
                                        components.append(str(val).strip())
                                
                                mat_value["notes"] = "; ".join(components) if components else "Mapped from Assembly Schedule"
                                mat_value["code"] = code
                                mapped_count += 1

    print(f"Successfully mapped {mapped_count} plan layout elements.")

    # Step 3: Strip original index views out of final payload to optimize token delivery window
    if mapped_count > 0:
        print("Schedules matched successfully.")
    else:
        print("Warning: No elements were mapped. Preserving raw view logs intact for code debugging.")

    # Step 4: Export mapped results payload back to standard tracking location
    pdf_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    mapped_json_path = os.path.join(state["output_base"], "data", f"{pdf_name}_mapped_materials.json")
    os.makedirs(os.path.dirname(mapped_json_path), exist_ok=True)
    
    with open(mapped_json_path, "w", encoding="utf-8") as f:
        json.dump(materials_list, f, indent=4, ensure_ascii=False)

    return {"mapped_materials": materials_list}