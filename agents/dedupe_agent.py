import copy
import re

from state.graph_state import AgenticState
from remove_duplicates import (
    normalize_code_to_name,
    dedupe_materials,
    dedupe_code_names,
    dedupe_paraphrases,
    add_notes,
)

CATEGORY_OVERRIDE_PATTERN = re.compile(r"^\s*(Door|Window)\s*-\s*\S+", re.IGNORECASE)


def flatten_nested_categories(materials: list) -> list:
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


def override_category_for_names(materials: list) -> list:
    """
    Kunai material ko 'name' ma prefix-code pattern cha bhane (e.g. 'Door- D2', 'Window- A', 'Door- 101'), 'category' lai 'name' le replace garcha so each unique door/window code stays distinct.
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


def dedupe_agent_node(state: AgenticState):
  
    print(f"\n=== [Agent 2: Dedupe / Post-Processing Agent] Cleaning & Deduplicating Materials ===")

    materials = state.get("extracted_materials", [])
    if not isinstance(materials, list):
        materials = [materials] if materials else []

    final_data = flatten_nested_categories(materials)
    final_data = override_category_for_names(final_data)

    print("Converting 'code'-only entries to 'name' (via Claude Haiku)...")
    normalize_code_to_name(final_data)
    print("Done converting 'code' -> 'name'.")

    to_remove, duplicate_group_count = dedupe_materials(final_data)
    print(f"Found {duplicate_group_count} exact duplicate group(s).")
    print(f"Exact duplicate entries to remove: {len(to_remove)}")

    code_to_remove, code_group_count = dedupe_code_names(final_data)
    print(f"Found {code_group_count} duplicate code/mark group(s) (e.g. Door-X, Window-X).")
    print(f"Duplicate code/mark entries to remove: {len(code_to_remove)}")
    to_remove |= code_to_remove

    paraphrase_to_remove, paraphrase_group_count = dedupe_paraphrases(final_data, to_remove)
    print(f"Found {paraphrase_group_count} paraphrase duplicate group(s).")
    print(f"Paraphrase duplicate entries to remove: {len(paraphrase_to_remove)}")
    to_remove |= paraphrase_to_remove

    print(f"Total duplicate entries to remove: {len(to_remove)}")

    final_data = [item for idx, item in enumerate(final_data) if idx not in to_remove]

    print("Generating shortened 'notes' for each material (via Claude Haiku)...")
    add_notes(final_data)
    print("Done generating 'notes'.")

    print(f"Post-processing complete. {len(final_data)} material(s) remain for CSI classification.")

    return {"extracted_materials": final_data}