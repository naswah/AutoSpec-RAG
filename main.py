import os
import json
import re
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from state.graph_state import AgenticState
from agents.ingestion_agent import ingestion_agent_node
from agents.csi_classifier import csi_classifier_node
from agents.validator_agent import validator_agent_node
from agents.summary_agent import summary_agent_node

load_dotenv(override=True)

MAX_RETRIES = 2


def increment_retry_node(state: AgenticState):
    current = state.get("retry_count", 0)
    print(f"\n[Retry Controller] Validation flagged issues. Retry attempt {current + 1}/{MAX_RETRIES}...")
    return {"retry_count": current + 1}


def route_after_validation(state: AgenticState) -> Literal["retry", "proceed"]:
    errors = state.get("error_log", [])
    retry_count = state.get("retry_count", 0)

    if errors and retry_count < MAX_RETRIES:
        return "retry"
    return "proceed"


def normalize_material_names(materials: list) -> list:
    """Kunai kunai materials ma 'name' ko thau ma 'code' cha so repace it by 'name' for backend processing"""
    normalized = []
    for item in materials:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        name = item.get("name")
        has_name = isinstance(name, str) and name.strip()

        code = item.get("code")
        has_code = isinstance(code, str) and code.strip()

        if has_name or not has_code:
            
            normalized.append(item)
            continue

        new_item = {}
        for k, v in item.items():
            if k == "code":
                new_item["name"] = code
            else:
                new_item[k] = v
        normalized.append(new_item)

    return normalized


# def build_workflow():

#     workflow = StateGraph(AgenticState)

#     workflow.add_node("ingestion", ingestion_agent_node)
#     workflow.add_node("csi_classifier", csi_classifier_node)
#     workflow.add_node("validator", validator_agent_node)
#     workflow.add_node("increment_retry", increment_retry_node)
#     workflow.add_node("summary", summary_agent_node)

#     workflow.set_entry_point("ingestion")

#     workflow.add_edge("ingestion", "csi_classifier")
#     workflow.add_edge("csi_classifier", "validator")

#     workflow.add_conditional_edges(
#         "validator",
#         route_after_validation,
#         {
#             "retry": "increment_retry",
#             "proceed": "summary",
#         },
#     )

#     workflow.add_edge("increment_retry", "csi_classifier")

#     workflow.add_edge("summary", END)

#     return workflow.compile()


def build_workflow():

    workflow = StateGraph(AgenticState)
    workflow.add_node("ingestion", ingestion_agent_node)
    workflow.add_node("csi_classifier", csi_classifier_node)
    workflow.add_node("validator", validator_agent_node)
    workflow.add_node("increment_retry", increment_retry_node)

    workflow.set_entry_point("ingestion")

    workflow.add_edge("ingestion", "csi_classifier")
    workflow.add_edge("csi_classifier", "validator")

    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "retry": "increment_retry",
            "proceed": END,  
        },
    )

    workflow.add_edge("increment_retry", "csi_classifier")

    return workflow.compile()


if __name__ == "__main__":
    inputs = {
        "pdf_path": r"D:\qtakeoffai-AI\qtakeoff-ai-AI\Example Plans\CR-574_HousePlans.pdf",
        "output_base": r"D:\qtakeoffai-AI\qtakeoff-ai-AI\outputs",
        "retry_count": 0,
        "error_log": []
    }

    print("Launching agentic AutoSpec RAG...")

    app = build_workflow()
    final_state = app.invoke(inputs)

    print("\n=== Workflow Complete ===")
    print(f"Total retries used: {final_state.get('retry_count', 0)}")
    if final_state.get("error_log"):
        print(f"Remaining validation notes ({len(final_state['error_log'])}):")
        for err in final_state["error_log"]:
            print(f"  - {err}")
    else:
        print("No outstanding validation issues.")

    results_folder = r"D:\qtakeoffai-AI\qtakeoff-ai-AI\Results"
    os.makedirs(results_folder, exist_ok=True)
    
    pdf_name = os.path.splitext(os.path.basename(final_state.get("pdf_path", "blueprint.pdf")))[0]
    output_file = os.path.join(results_folder, f"{pdf_name}_Final.json")
    
    final_data = final_state.get("extracted_materials", [])
    final_data = normalize_material_names(final_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(f"Final validated JSON saved securely to standalone folder: {output_file}")