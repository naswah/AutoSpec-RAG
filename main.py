import os
import json
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from config import PDF_PATH, OUTPUT_BASE, RESULTS

from state.graph_state import AgenticState
from agents.ingestion_agent import ingestion_agent_node
from agents.dedupe_agent import dedupe_agent_node
from agents.csi_classifier import csi_classifier_node
from agents.validator_agent import validator_agent_node
from agents.summary_agent import summary_agent_node
from tools.pdf_helpers import is_valid_architectural_pdf  

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


#stop the workflow if the scale report is empty, which indicates that the PDF was not a valid blueprint
def router_after_ingestion(state: AgenticState)-> Literal["proceed", "terminate"]:
    if state.get("status") == "terminated_empty_scale":
        return "terminate"

    return "proceed"

def pre_validation_node(state: AgenticState):
    is_valid, result = is_valid_architectural_pdf(state["pdf_path"])
    if not is_valid:
        print(f"\n[Pre-Validation] Only {result['total']} floor/elevation plan(s) found "
              f"(need at least 2). Terminating.")
        return {"status": "terminated_invalid_pdf"}
    print("\n[Pre-Validation] PDF is valid. Proceeding to ingestion.")
    return {"status": "pre_validated"}


def router_after_pre_validation(state: AgenticState) -> Literal["proceed", "terminate"]:
    if state.get("status") == "terminated_invalid_pdf":
        return "terminate"
    return "proceed"


def build_workflow():

    workflow = StateGraph(AgenticState)
    workflow.add_node("pre_validation", pre_validation_node)
    workflow.add_node("ingestion", ingestion_agent_node)
    workflow.add_node("dedupe", dedupe_agent_node)
    workflow.add_node("csi_classifier", csi_classifier_node)
    workflow.add_node("validator", validator_agent_node)
    workflow.add_node("increment_retry", increment_retry_node)
    
    workflow.set_entry_point("pre_validation")
    workflow.add_conditional_edges(
        "pre_validation",
        router_after_pre_validation,
        {
            "proceed": "ingestion",
            "terminate": END,
        },
    )
    workflow.add_conditional_edges(
    "ingestion",
    router_after_ingestion,
    {
        "proceed": "dedupe",
        "terminate": END,
    },
)
    workflow.add_edge("dedupe", "csi_classifier")
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
    app =workflow.compile()
    return app


# IF SUMMARY NODE IS ALSO REQUIRED, swap "proceed": END for "proceed": "summary"
# and add:
#   workflow.add_node("summary", summary_agent_node)
#   workflow.add_edge("summary", END)


if __name__ == "__main__":
    inputs = {
        "pdf_path": PDF_PATH,
        "output_base": OUTPUT_BASE,
        "retry_count": 0,
        "error_log": []
    }

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

    results_folder = RESULTS
    os.makedirs(results_folder, exist_ok=True)

    pdf_name = os.path.splitext(os.path.basename(final_state.get("pdf_path", "blueprint.pdf")))[0]
    output_file = os.path.join(results_folder, f"{pdf_name}_Final.json")

    final_data = final_state.get("final_specifications") or final_state.get("extracted_materials", [])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(f"\nJSON saved securely to folder: {output_file}")