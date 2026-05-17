import json
import os
from state.graph_state import AgenticState

def blueprint_mapper_node(state: AgenticState):
    print(f"\n=== [Agent 2: Blueprint Mapper] Processing Saved Raw JSON ===")
    
    # Load the extracted materials list from state
    materials_list = state["extracted_materials"]
    code_registry = {}

    # --- PASS 1: Build a registry of code properties from schedules ---
    for page in materials_list:
        for view in page.get("views", []):
            materials = view.get("materials", {})
            for mat_key, mat_value in materials.items():
                # Check if this item defines a schedule property block
                if isinstance(mat_value, dict) and "properties" in mat_value:
                    # Registry tracks the properties block (e.g., code_registry["F-78"] = { ... })
                    code_registry[mat_key] = mat_value["properties"]

    print(f"Discovered schedule reference codes: {list(code_registry.keys())}")

    # --- PASS 2: Replace 'Mapping Required' with target properties ---
    mapped_count = 0
    for page in materials_list:
        for view in page.get("views", []):
            materials = view.get("materials", {})
            for mat_name in list(materials.keys()):
                current_value = materials[mat_name]
                is_req = False

                # Scenario A: The raw value is just the string "Mapping Required"
                if isinstance(current_value, str) and current_value == "Mapping Required":
                    is_req = True
                # Scenario B: The value is a dict, and its notes field says "Mapping Required"
                elif isinstance(current_value, dict) and current_value.get("notes") == "Mapping Required":
                    is_req = True

                if is_req:
                    if mat_name in code_registry:
                        # Replace/inject the full properties block structure
                        materials[mat_name] = {
                            "notes": "Mapped from Schedule",
                            "properties": code_registry[mat_name]
                        }
                        print(f"Resolved code assignment: '{mat_name}' -> Injected properties successfully.")
                        mapped_count += 1
                    else:
                        print(f"Notice: Code '{mat_name}' requires mapping but has no definition in schedules.")

    print(f"Mapping phase completed. Resolved {mapped_count} items.")

    # Save output to REQUIRED_mapped_materials.json style path
    pdf_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    mapped_json_path = os.path.join(state["output_base"], "data", f"{pdf_name}_mapped_materials.json")
    os.makedirs(os.path.dirname(mapped_json_path), exist_ok=True)
    
    with open(mapped_json_path, "w", encoding="utf-8") as f:
        json.dump(materials_list, f, indent=4, ensure_ascii=False)
    print(f"Mapped JSON structure saved to: {mapped_json_path}")

    return {"mapped_materials": materials_list}