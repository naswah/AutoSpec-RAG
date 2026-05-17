import re
from state.graph_state import AgenticState

def validator_agent_node(state: AgenticState):
    print(f"\n=== [Agent 4: Quality Validator] Checking Format Compliance ===")
    specifications = state["final_specifications"]
    errors = []
    
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
                    csi = info.get("csi_division", "")
                    if not csi or csi == "none" or not csi_pattern.match(str(csi).strip()):
                        errors.append(f"Material '{mat}' has invalid or missing 'csi_division': '{csi}'. Standard pattern template 'XX XX XX' required.")
                else:
                    errors.append(f"Material '{mat}' lacks internal dictionary configuration holding 'csi_division'.")

    if errors:
        print(f"Validation failure found. Logged {len(errors)} formatting defects.")
        return {"error_log": errors, "retry_count": state.get("retry_count", 0) + 1}
    
    print("Validation checklist complete. Content format complies with standards.")
    return {"error_log": []}