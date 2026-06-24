import re
from state.graph_state import AgenticState


def validator_agent_node(state: AgenticState):
    print(f"\n=== [Agent 3: Quality Validator] Checking Format Compliance ===")
    specifications = state["final_specifications"]
    errors = []

    csi_pattern = re.compile(r"^\d{2}\s\d{2}\s\d{2}(\.\d{2})?$")

    materials = specifications if isinstance(specifications, list) else specifications.get("materials", [])

    for info in materials:
        if not isinstance(info, dict):
            continue

        csi = str(info.get("csi_division", "")).strip()
        if csi == "" or csi.lower() == "none":
            continue

        if not csi_pattern.match(csi):
            identifier = info.get("code") or info.get("name") or "Unknown Material"
            errors.append(f"Material '{identifier}' has invalid 'csi_division': '{csi}'. Use template pattern 'XX XX XX'.")

    if errors:
        print(f"Validation finished. Noted {len(errors)} formatting anomalies, but proceeding anyway to save API costs.")
        return {"error_log": errors}

    print("Validation passed completely with zero structural schema defects.")
    return {"error_log": []}