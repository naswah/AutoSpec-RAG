import re
from state.graph_state import AgenticState

def validator_agent_node(state: AgenticState):
    print(f"\n=== [Agent 4: Quality Validator] Checking Format Compliance ===")
    specifications = state["final_specifications"]
    errors = []
    
    # Strict pattern for valid codes: 'XX XX XX' or 'XX XX XX.XX'
    csi_pattern = re.compile(r"^\d{2}\s\d{2}\s\d{2}(\.\d{2})?$")
    
    pages = specifications if isinstance(specifications, list) else specifications.get("materials", [])
    if isinstance(specifications, dict) and not pages and "views" in specifications:
        pages = [specifications]
    elif isinstance(specifications, dict) and "page_no" in specifications:
         pages = [specifications]

    for p in pages:
        if not isinstance(p, dict): continue
        for view in p.get("views", []):
            for mat, info in view.get("materials", {}).items():
                if isinstance(info, dict):
                    csi = str(info.get("csi_division", "")).strip()
                    
                    if csi == "" or csi.lower() == "none":
                        continue  
                        
                    if not csi_pattern.match(csi):
                        errors.append(f"Material '{mat}' has invalid 'csi_division': '{csi}'. Standard pattern template 'XX XX XX' required.")
                else:
                    errors.append(f"Material '{mat}' lacks internal dictionary configuration holding 'csi_division'.")

    if errors:
        print(f"Validation finished. Noted {len(errors)} formatting anomalies, but proceeding anyway to save API costs.")
        return {"error_log": errors}
    
    print("Validation checklist complete. Content format complies with standards.")
    return {"error_log": []}