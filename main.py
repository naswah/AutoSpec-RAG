# main_workflow.py
import os
import json
import re
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from state.graph_state import AgenticState
from agents.ingestion_agent import ingestion_agent_node
from agents.blueprint_mapper import blueprint_mapper_node
from agents.csi_classifier import csi_classifier_node
from agents.validator_agent import validator_agent_node

load_dotenv(override=True)


def check_for_blueprint_codes(state: AgenticState) -> Literal["run_mapper", "skip_mapper"]:

    print("\nDecision Hub: Scanning extracted JSON for structural codes (e.g., X-30, F-78)...")
    extracted_data = state.get("extracted_materials", [])
    json_str = json.dumps(extracted_data)
    
    code_pattern = re.compile(r"\b[A-Za-z]-\d+\b")
    
    if code_pattern.search(json_str) or "Mapping Required" in json_str:
        print("Structural codes found! Routing to Agent 2 (Blueprint Mapper).")
        return "run_mapper"
    
    print("No blueprint codes detected. Skipping Agent 2 step entirely.")
    return "skip_mapper"


def evaluation_router(state: AgenticState) -> Literal["back_to_classifier", "save_and_exit"]:
    if state.get("error_log") and state.get("retry_count", 0) < 3:
        print(f"Routing back to Agent 3 (CSI Classifier) for correction attempt #{state.get('retry_count', 0)}")
        return "back_to_classifier"
    return "save_and_exit"


def save_results_node(state: AgenticState):
    plan_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    save_path = os.path.join(os.getcwd(), "Results", f"{plan_name}_Final.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    final_output = state.get("final_specifications")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\nFinal output successfully saved to: {save_path}")
    return {}


workflow = StateGraph(AgenticState)

workflow.add_node("agent_ingestion", ingestion_agent_node)
workflow.add_node("agent_mapper", blueprint_mapper_node)
workflow.add_node("agent_classifier", csi_classifier_node)
workflow.add_node("agent_validator", validator_agent_node)
workflow.add_node("node_save", save_results_node)

workflow.set_entry_point("agent_ingestion")

workflow.add_conditional_edges(
    "agent_ingestion",
    check_for_blueprint_codes,
    {
        "run_mapper": "agent_mapper",      # Code found -> map it
        "skip_mapper": "agent_classifier"  # Code missing -> bypass mapper node completely
    }
)

workflow.add_edge("agent_mapper", "agent_classifier")
workflow.add_edge("agent_classifier", "agent_validator")

workflow.add_conditional_edges(
    "agent_validator",
    evaluation_router,
    {
        "back_to_classifier": "agent_classifier",
        "save_and_exit": "node_save"
    }
)
workflow.add_edge("node_save", END)

app = workflow.compile()

try:
    graph_image_bytes = app.get_graph().draw_mermaid_png()
    
    with open("workflow_graph.png", "wb") as f:
        f.write(graph_image_bytes)
except Exception as e:
    print(f"Could not generate graph image: {e}")

if __name__ == "__main__":
    inputs = {
        "pdf_path": r"D:\AutoSpec RAG\Example Plans\REQUIRED.pdf",
        "output_base": r"D:\AutoSpec RAG\output",
        "retry_count": 0,
        "error_log": []
    }
    print("Launching agentic AutoSpec RAG...")
    app.invoke(inputs)