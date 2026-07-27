import os
import json
import re
import copy
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from config import PDF_PATH, OUTPUT_BASE, RESULTS

from state.graph_state import AgenticState
from agents.ingestion_agent import ingestion_agent_node
from agents.csi_classifier import csi_classifier_node
from agents.validator_agent import validator_agent_node
from agents.summary_agent import summary_agent_node

load_dotenv(override=True)

MAX_RETRIES = 2
CATEGORY_OVERRIDE_PATTERN = re.compile(r"^\s*(Door|Window)\s*-\s*\S+", re.IGNORECASE)
CODE_NAME_PATTERN = re.compile(r"^\s*[A-Za-z]+\s*-?\s*\d+\s*$")


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


def flatten_nested_categories(materials: list)->list:
    flattened = []
    for item in materials:
        if not isinstance(item, dict):
            flattened.append(item)
            continue
 
        category = item.get("category")
 
        if isinstance(category, dict):
            for key in sorted(category.keys()):
                new_item = copy.deepcopy(item)
                new_item["category"] = category[key]
                flattened.append(new_item)
        else:
            flattened.append(item)
 
    return flattened


def deduplicate_materials(materials: list) -> list:
    """
    Sadharan materials ko lagi: same name + category + csi_division + notes bhayeko items duplicate huncha, ra sabai bhanda dherai 'mentions' bhayeko item matra rakhcha.
    Code-style names (F-22, Door-1, W1, Window-2, jasto) ko lagi: name + category matra hercha.
    - Same code + same category bhayo bhane -> duplicate ho, sabai bhanda dherai mentions bhayeko ekutamatra rakhcha.
    - Same code tara different category bhayo bhane -> duplicate hoina, dubai rakhcha, as it is.
    """
    groups: dict = {}
    non_dict_items = []
    order = []

    for item in materials:
        if not isinstance(item, dict):
            non_dict_items.append(item)
            continue

        name = str(item.get("name") or "").strip()
        category = str(item.get("category") or "").strip().lower()

        name_lower = name.lower()

        if CODE_NAME_PATTERN.match(name):
            # Code-like name (F-22, Door-1, W1, Window-2, etc.): key on name + category only
            key = (name_lower, category)
        else:
            csi_division = str(item.get("csi_division") or "").strip().lower()
            note = str(item.get("notes") or "").strip().lower()
            key = (name_lower, category, csi_division, note)

        mentions = item.get("mentions")
        mention_count = len(mentions) if isinstance(mentions, list) else 0

        if key not in groups:
            groups[key] = item
            order.append(key)
        else:
            existing = groups[key]
            existing_mentions = existing.get("mentions")
            existing_count = len(existing_mentions) if isinstance(existing_mentions, list) else 0

            if mention_count > existing_count:
                groups[key] = item

    deduplicated = [groups[key] for key in order] + non_dict_items

    removed = len(materials) - len(deduplicated)
    print(f"[Deduplication] {removed} duplicate(s) removed (kept the entry with most mentions per group). {len(deduplicated)} unique material(s) retained.")

    return deduplicated


def override_category_for_names(materials: list) -> list:
    """
    Kunai material ko 'name' ma prefix-code pattern cha bhane (e.g. 'Door- D2', 'Window- A', 'Door- 101'), 'category' lai 'name' le replace garcha so each unique door/window code stays distinct .
    """
    updated = []
    for item in materials:
        if not isinstance(item, dict):
            updated.append(item)
            continue

        name = item.get("name")
        if isinstance(name, str) and CATEGORY_OVERRIDE_PATTERN.match(name):
            new_item = dict(item)
            new_item["category"] = name
            updated.append(new_item)
        else:
            updated.append(item)

    return updated


# IF SUMMARY NODE IS REQUIRED, WITHOUT SCALE AGENT

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


# WITHOIT SCALE AND SUMMARY NODE

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
        "pdf_path": PDF_PATH,
        "output_base": OUTPUT_BASE,
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

    results_folder = RESULTS
    os.makedirs(results_folder, exist_ok=True)
    
    pdf_name = os.path.splitext(os.path.basename(final_state.get("pdf_path", "blueprint.pdf")))[0]
    output_file = os.path.join(results_folder, f"{pdf_name}_Final.json")
    
    final_data = final_state.get("extracted_materials", [])
    final_data = normalize_material_names(final_data)
    final_data = flatten_nested_categories(final_data)
    final_data = override_category_for_names(final_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(f"JSON saved securely to folder: {output_file}")

    source_file = os.path.join(results_folder, f"{pdf_name}_Final.json")
 
    print(f"\n[Deduplication] Reading: {source_file}")
    with open(source_file, "r", encoding="utf-8") as f:
        source_data= json.load(f)
 
    deduplicated_data = deduplicate_materials(source_data)
 
    dedup_output_file= os.path.join(results_folder, f"{pdf_name}_Final_2.json")
    with open(dedup_output_file, "w", encoding="utf-8") as f:
        json.dump(deduplicated_data, f, indent=4, ensure_ascii=False)
 
    print(f"Deduplicated JSON saved to: {dedup_output_file}")