import json
import os
import re
from state.graph_state import AgenticState

def blueprint_mapper_node(state: AgenticState):
    print(f"\nMapping Blueprint Reference Codes...")
    
    materials_list = state.get("extracted_materials", [])
    code_registry = {}

    # Step 1: Discover and register all master schedules, tables, and notes (W1, F1, R1, etc.)
    for page in materials_list:
        views = page.get("views", []) or page.get("view", [])
        for view in views:
            materials = view.get("materials", [])
            
            # If materials is structured as a list of rows/assemblies (e.g., General Notes, Schedules)
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

    # Step 2: Map placeholder views ("Mapping Required") to their matching registry data
    mapped_count = 0
    for page in materials_list:
        views = page.get("views", []) or page.get("view", [])
        for view in views:
            materials = view.get("materials", [])
            
            if isinstance(materials, dict):
                for mat_key, mat_value in materials.items():
                    if not isinstance(mat_value, dict):
                        continue
                        
                    notes_str = str(mat_value.get("notes", "")).strip().lower()
                    code = str(mat_value.get("code", "")).strip()
                    
                    if notes_str == "mapping required" and code in code_registry:
                        registry_entry = code_registry[code]
                        reg_notes = registry_entry.get("notes")
                        
                        # --- DYNAMIC STRUCTURE HANDLING FOR CATEGORY C ---
                        if isinstance(reg_notes, dict) and "submaterials" in reg_notes:
                            mat_value["notes"] = reg_notes
                            mat_value["code"] = code
                            mapped_count += 1
                            
                        # Fallback: if it's a structural dictionary row entry with flat values
                        elif isinstance(registry_entry, dict) and ("item" in registry_entry or "material" in registry_entry):
                            components = []
                            for key in ["item", "size", "material", "notes", "specification"]:
                                val = registry_entry.get(key, "")
                                if val and str(val).strip() not in ["", "none", "-", "Mapping Required"]:
                                    components.append(str(val).strip())
                            
                            mat_value["notes"] = "; ".join(components) if components else "Mapped from Assembly Schedule"
                            mat_value["code"] = code
                            mapped_count += 1
                            
                        # Safe fallback for simple textual descriptions
                        else:
                            mat_value["notes"] = str(reg_notes) if reg_notes else "Mapped from Assembly Schedule"
                            mat_value["code"] = code
                            mapped_count += 1

    print(f"Successfully mapped {mapped_count} plan layout elements.")

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