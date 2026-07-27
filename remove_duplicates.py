import os
import re
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

INPUT_PATH = r"D:\qtakeoffai-AI\qtakeoff-ai-AI\local\results\American Farmhouse 201225 full_Final_2.json"

RESULTS_FOLDER = os.path.join("local", "results")

BATCH_MAX_ITEMS = 40

client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

_SUFFIX_RE = re.compile(r"^[A-Za-z0-9]{1,4}$")
_NO_DASH_CODE_RE = re.compile(r"^[A-Za-z]{1,2}\d{1,3}[A-Za-z]?$")


def is_code_name(name: str) -> bool:
    """Return True if `name` looks like a mark/tag code (e.g. Door-D1, Window-01, F-XX, W1) rather than a real material description."""
    name = (name or "").strip()
    if not name:
        return False

    if "-" in name:
        prefix, _, suffix = name.partition("-")
        prefix = prefix.strip()
        suffix = suffix.strip()
        if prefix.isalpha() and _SUFFIX_RE.match(suffix):
            return True
        return False

    # No dash: short alpha+digit tags like "W1", "D1"
    if len(name) <= 4 and _NO_DASH_CODE_RE.match(name):
        return True

    return False


def get_pdf_name(input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    base = re.sub(r"(_Final_2|_Final)$", "", base, flags=re.IGNORECASE)
    return base


def mention_count(item: dict) -> int:
    mentions = item.get("mentions")
    return len(mentions) if isinstance(mentions, list) else 0


def normalize(value) -> str:
    return str(value or "").strip().lower()


_NUMBER_RE = re.compile(r"\d")


def has_numeric_info(notes: str) -> bool:
    """True if the notes contain any digit (dimensions, thickness, sizes, etc)."""
    return bool(_NUMBER_RE.search(notes or ""))


def merge_mentions(materials: list, indices: list) -> list:
    seen = set()
    merged = []
    for i in indices:
        mentions = materials[i].get("mentions")
        if not isinstance(mentions, list):
            continue
        for m in mentions:
            if isinstance(m, dict):
                key = (normalize(m.get("page_label")), normalize(m.get("view")))
            else:
                key = normalize(m)
            if key in seen:
                continue
            seen.add(key)
            merged.append(m)
    return merged


def ask_claude_for_paraphrase_duplicates(items_with_idx: list) -> list:

    if len(items_with_idx) < 2:
        return []

    local_list = []
    for local_pos, (orig_idx, item) in enumerate(items_with_idx):
        local_list.append({
            "id": local_pos,
            "name": item.get("name", ""),
            "notes": item.get("notes", "")
        })

    prompt = f"""You are comparing construction material entries that all share the same category.

Here is a list of entries (id, name, notes):

{json.dumps(local_list, indent=2, ensure_ascii=False)}

Your task: identify which entries describe the SAME underlying material, where the "name" and/or "notes" are just paraphrases, synonyms, or reworded versions of each other (not genuinely different materials).

Rules:
- Only group entries together if they clearly refer to the same material, just worded differently.
- Do NOT group entries that describe genuinely different materials, even if related.
- Singletons (materials with no duplicate) should NOT appear in your output at all.
- Each id can belong to at most one group.
- For each group, decide which single id should be KEPT: prefer the entry whose "notes" contain concrete numerical information (dimensions, thickness, sizes, etc). If more than one entry has numerical info, or none of them do, keep whichever entry has the more complete/detailed information overall.

Respond with ONLY a JSON array of objects, no other text, no markdown formatting, no code fences, in this exact format:
[{{"ids": [<id>, <id>, ...], "keep_id": <id>}}, ...]

If there are no duplicates, respond with: []"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()

    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip()).strip()

    try:
        local_groups = json.loads(text)
    except json.JSONDecodeError:
        print(f"  [Warning] Could not parse Claude response, skipping this group. Raw response: {text[:200]}")
        return []

    orig_groups = []
    for g in local_groups:
        try:
            local_ids = g["ids"]
            local_keep = g["keep_id"]
        except (KeyError, TypeError):
            continue

        orig_ids = [items_with_idx[i][0] for i in local_ids if 0 <= i < len(items_with_idx)]
        if len(orig_ids) < 2:
            continue

        if 0 <= local_keep < len(items_with_idx):
            keep_idx = items_with_idx[local_keep][0]
        else:
            keep_idx = None

        if keep_idx is None or keep_idx not in orig_ids:
          
            item_by_orig_idx = {oi: it for oi, it in items_with_idx}
            numeric_ids = [i for i in orig_ids if has_numeric_info(item_by_orig_idx[i].get("notes", ""))]
            candidates = numeric_ids if numeric_ids else orig_ids
            keep_idx = max(candidates, key=lambda i: len(item_by_orig_idx[i].get("notes", "") or ""))

        orig_groups.append({"ids": orig_ids, "keep_idx": keep_idx})

    return orig_groups


def dedupe_paraphrases(materials: list, exclude_indices: set):

    groups: dict = {}
    order = []

    for idx, item in enumerate(materials):
        if idx in exclude_indices or not isinstance(item, dict):
            continue

        name = item.get("name", "")
        if is_code_name(name):
            continue

        key = normalize(item.get("category"))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(idx)

    to_remove = set()
    merge_group_count = 0

    for key in order:
        indices = groups[key]
        if len(indices) < 2:
            continue

        for start in range(0, len(indices), BATCH_MAX_ITEMS):
            chunk_indices = indices[start:start + BATCH_MAX_ITEMS]
            items_with_idx = [(i, materials[i]) for i in chunk_indices]

            dup_groups = ask_claude_for_paraphrase_duplicates(items_with_idx)
            for g in dup_groups:
                ids = g["ids"]
                keep_idx = g["keep_idx"]

                merged = merge_mentions(materials, ids)
                materials[keep_idx]["mentions"] = merged

                merge_group_count += 1
                for i in ids:
                    if i != keep_idx:
                        to_remove.add(i)

    return to_remove, merge_group_count


def dedupe_code_names(materials: list):
    """Materials whose name is a mark/tag code (e.g. Door-D1, Window-01,
    F-XX, W1) are normally left alone by the other passes. But if the same
    code appears more than once WITHIN THE SAME CATEGORY, only one
    occurrence should survive - duplicates are removed, keeping the one
    with the most mentions (ties broken by first occurrence).

    Note: codes are keyed by (name, category), not name alone - a code
    like F-60 can legitimately repeat across different categories/rooms
    (e.g. the same flooring material listed for Living Room, Dining,
    Bedroom, etc.), and those are distinct entries that must NOT be
    collapsed together."""

    groups: dict = {}
    order = []

    for idx, item in enumerate(materials):
        if not isinstance(item, dict):
            continue

        name = item.get("name", "")
        if not is_code_name(name):
            continue  # only handle code/mark names here

        key = (normalize(name), normalize(item.get("category")))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(idx)

    to_remove = set()
    duplicate_group_count = 0

    for key in order:
        indices = groups[key]
        if len(indices) < 2:
            continue  # only one occurrence, nothing to remove

        duplicate_group_count += 1
        best_idx = max(indices, key=lambda i: mention_count(materials[i]))
        for i in indices:
            if i != best_idx:
                to_remove.add(i)

    return to_remove, duplicate_group_count


def dedupe_materials(materials: list):

    groups: dict = {}
    order = []

    for idx, item in enumerate(materials):
        if not isinstance(item, dict):
            continue

        name = item.get("name", "")
        if is_code_name(name):
            continue  # skip marks/tags entirely - never group these

        key = (normalize(name), normalize(item.get("category")))

        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(idx)

    to_remove = set()
    duplicate_group_count = 0

    for key in order:
        indices = groups[key]
        if len(indices) < 2:
            continue  # nothing to compare, unique material

        duplicate_group_count += 1
        best_idx = max(indices, key=lambda i: mention_count(materials[i]))
        for i in indices:
            if i != best_idx:
                to_remove.add(i)

    return to_remove, duplicate_group_count


def main():
    if not INPUT_PATH:
        raise ValueError("Please set INPUT_PATH at the top of check.py to the *_Final_2.json file path.")

    print(f"Reading: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        materials = json.load(f)

    print(f"Loaded {len(materials)} material(s).")

    to_remove, duplicate_group_count = dedupe_materials(materials)

    print(f"Found {duplicate_group_count} exact duplicate group(s).")
    print(f"Exact duplicate entries to remove: {len(to_remove)}")

    code_to_remove, code_group_count = dedupe_code_names(materials)

    print(f"Found {code_group_count} duplicate code/mark group(s) (e.g. Door-X, Window-X).")
    print(f"Duplicate code/mark entries to remove: {len(code_to_remove)}")

    to_remove |= code_to_remove

    paraphrase_to_remove, paraphrase_group_count = dedupe_paraphrases(materials, to_remove)

    print(f"Found {paraphrase_group_count} paraphrase duplicate group(s).")
    print(f"Paraphrase duplicate entries to remove: {len(paraphrase_to_remove)}")

    to_remove |= paraphrase_to_remove

    print(f"Total duplicate entries to remove: {len(to_remove)}")

    final_materials = [item for idx, item in enumerate(materials) if idx not in to_remove]

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    pdf_name = get_pdf_name(INPUT_PATH)
    output_file = os.path.join(RESULTS_FOLDER, f"{pdf_name}_3.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_materials, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(final_materials)} material(s) to: {output_file}")


if __name__ == "__main__":
    main()