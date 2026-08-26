import os
import re
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

INPUT_PATH = r"D:\qtakeoffai-AI\qtakeoff-ai-AI\local\results\Till Aug 24\CMT Temple_Final_2.json"

RESULTS_FOLDER = os.path.join("local", "results")

BATCH_MAX_ITEMS = 40
NOTES2_BATCH_MAX_ITEMS = 40

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


def mention_views(item: dict) -> frozenset:

    mentions = item.get("mentions")
    if not isinstance(mentions, list):
        return frozenset()
    views = set()
    for m in mentions:
        if isinstance(m, dict):
            views.add(normalize(m.get("view")))
        else:
            views.add(normalize(m))
    return frozenset(views)


def dedupe_code_names(materials: list):

    groups: dict = {}
    order = []

    for idx, item in enumerate(materials):
        if not isinstance(item, dict):
            continue

        name = item.get("name", "")
        if not is_code_name(name):
            continue  # only handle code/mark names here

        key = (normalize(name), normalize(item.get("category")), mention_views(item))
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


_MEASUREMENT_RE = re.compile(
    r"""
    \d+(?:/\d+)?(?:-\d+/\d+)?\s*(?:in\.?|"|'|ft\.?|mm|cm|mil\b|ga\.?|gauge)   # 3/4", 1/2 in, 6 mil, 16 ga
    |
    \bR-\d+(?:\.\d+)?\b                                                       # R-30, R-13
    |
    \d+(?:,\d{3})*(?:\.\d+)?\s*(?:psi|ksi)\b                                  # 3000 psi, 4 ksi
    |
    \bGrade\s*\d+\b                                                          # Grade 60
    |
    \d+\s*(?:o\.c\.|oc)\b                                                    # 16" o.c. (backup, catches trailing oc)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _extract_measurements(text: str) -> set:
    return {m.strip().lower() for m in _MEASUREMENT_RE.findall(text or "")}


def measurements_dropped(orig_notes: str, notes2: str) -> bool:
  
    orig_tokens = _extract_measurements(orig_notes)
    if not orig_tokens:
        return False
    notes2_lower = (notes2 or "").lower()
    for token in orig_tokens:
        if token not in notes2_lower:
            return True
    return False


_STRIP_WORDS_RE = re.compile(r"\b(assembly|assemblies|detail|details)\b", re.IGNORECASE)


def clean_notes2(notes2: str) -> str:

    if not notes2:
        return notes2

    cleaned = _STRIP_WORDS_RE.sub("", notes2)

    # Tidy up leftover punctuation/whitespace from removed words
    cleaned = re.sub(r"\s{2,}", " ", cleaned)                # collapse double spaces
    cleaned = re.sub(r"\s+,", ",", cleaned)                  # space before comma
    cleaned = re.sub(r",\s*,", ",", cleaned)                 # double commas
    cleaned = re.sub(r",\s*$", "", cleaned)                  # trailing comma
    cleaned = re.sub(r"^\s*,\s*", "", cleaned)                # leading comma
    cleaned = cleaned.strip()

    return cleaned


def ask_claude_for_notes2(items_with_idx: list) -> dict:
   
    if not items_with_idx:
        return {}

    local_list = []
    for local_pos, (orig_idx, item) in enumerate(items_with_idx):
        local_list.append({
            "id": local_pos,
            "notes": item.get("notes", "")
        })

    prompt = f"""You are cleaning up "notes" fields for construction material entries.

Here is a list of entries (id, notes):

{json.dumps(local_list, indent=2, ensure_ascii=False)}

Your task: for each entry, produce a shorter version of "notes" called "notes2" by:
- Removing any reference to a code, mark, or tag (e.g. "Door-D1", "Window-02", "per mark W1", "type A3", "ref: F-12").
- Removing any mention of the count/quantity/number of that material (e.g. "3 units", "qty: 5", "x4", "5 pieces", "count: 2", "each 12.5 ft.").
- KEEPING model numbers exactly as written if present but if the model name is present, capitalize the name of model. Example if notes has: AMANA MODEL as model name then the notes2 must be Amana Model.(e.g. "AMANA MODEL 7184596" should be Amana Model 7184596, "SERIN WIRE MODEL HS10-OMP" should be Serin Wire Model HS10-OMP) — do NOT remove these. You may drop a trailing "OR EQUAL" / "or approved equal" qualifier since it adds no information on its own. make sure that the model code is untouched.

  ***CRITICAL***: If the note comes with locations at exterior wall, doors and windows then just add at exterior wall in the note. Exclude doors and windows. Example:
    - "Fiber cement lap siding at exterior wall assembly, at window and door head/sill and jamb details" is in notes sections so the "notes2" must have: "Fiber cement lap siding at exterior wall" 
    but if doors and windows come alone without exterior wall, then keep it
    - "Fiber cement lap siding at window and door head/sill and jamb details" → drop only the drafting-callout part ("head/sill and jamb details") → "Fiber cement lap siding, at window and door"
- Removing any reference to construction drawing details/callouts rather than the material itself — e.g. "sill detail", "jamb detail", "head detail", "lintel detail", "window and door head/sill and jamb details", "elevation detail", "plan detail", "pipe penetration detail", "to follow corner boards", "window lintel and jamb details", "niche detail conditions". These are drafting references, not material specs, and should be dropped entirely (along with any connecting words like "at", "per", "see" that only existed to introduce them).
Example: 
 - "Fiber cement lap siding at window and door head/sill and jamb details" → drop only the drafting-callout part ("head/sill and jamb details") → "Fiber cement lap siding, at window and door"

  * IMPORTANT — do NOT over-strip: only drop the drafting-callout phrase itself. If the same sentence also names a real building location/component (e.g. "exterior wall", "window and door", "slab-on-grade", "roof edge", "metal stud-brick wall"), KEEP that location and only remove the callout part. The word "assembly"/"assemblies" attached to a location is handled separately (by code, not you) — leave it in notes2 exactly as written; do not delete the location just because "assembly" follows it.

    - "EXT SHEATHING W/ BUILDING WRAP at window and door details, exterior wall assembly" → keep only "exterior wall assembly"  → "Exterior sheathing with building wrap, at  exterior wall"
    - "Batt insulation used at exterior wall assemblies at window lintel and window jamb details at metal stud-brick wall" → keep "exterior wall" AND "metal stud-brick wall" (both are real locations), drop only "window lintel and window jamb details" → "Batt insulation, at exterior wall assemblies, metal stud-brick wall"
    - "2X6 SURROUND at window and door head/sill and jamb details" → "window and door" IS a real location (where the surround sits); drop only "head/sill and jamb details", keep "window and door" → "2x6 surround, at window and door"

  * Rule of thumb: words like "head", "head details", "sill", "sill details", "jamb", "jamb details", "lintel", "elevation", "plan", describe a drawing VIEW/callout and should always be dropped. Words like "exterior wall", "window and door" / "doors and windows", "slab-on-grade", "roof edge", "metal stud-brick wall" name a physical LOCATION/component and should always be kept, even when a drawing-view word or "detail(s)" immediately follows them. When SEVERAL such locations/components appear together in one entry (not a "Location:" room list — see below), keep ALL of them; they describe different parts of the same assembly the material touches, not interchangeable alternatives.

- Removing any field/segment of "notes" whose value is empty, blank, "-", "N/A", "NA", or similar (e.g. if notes contains "Glazing: -" or "Glazing: N/A", drop that whole "Glazing: ..." segment — don't write "Glazing:" with nothing after it).
- Location handling: this rule applies ONLY to an explicit "Location:" field that lists a long comma-separated set of ROOMS/SPACES (e.g. "Location: LOBBY, FOYER PERIPHERY, PRAYER HALL, DEITY PEDESTALS, NICHES", or plain "kitchen, bedroom, hallway") — if that field lists more than one room/space, drop the whole location field from "notes2" entirely; if it lists only one room/space, keep it.
  * This does NOT apply to structural components/locations mentioned in ordinary prose (as opposed to a room list) — see the rule of thumb above. Keep every such structural location, no matter how many appear.
- For doors and windows specifically, do NOT include frame type/frame material or other framing construction details in "notes2" (e.g. drop "Frame Type: HM", "Frame Material: Steel frame") — keep the door/window's own size, material, finish, and hardware instead.
- Keeping all other meaningful spec information intact: size, thickness, type, and strength (as well as spacing, grade, and finish if present).

 "notes": "Qty: 1, Description: REFRIGERATOR, Item Specification: AMANA MODEL 7184596, OR EQUAL",
 "notes2": "Refrigerator, Amana Model 7184596",

STYLE — write "notes2" as a terse, comma-separated catalog phrase, in the same compact style used by RSMeans-type cost-database descriptions. NOT a full sentence: no "The", no subject/verb narrative, no trailing period. Lead with the core item/material type, then add comma-separated modifiers (size, thickness, type, strength, single location if present) in natural left-to-right order. Keep inch marks as the " symbol exactly as written in the original "notes" — do NOT spell out the word "inch". Examples of the target style:
  "Welded wire mesh, below 4\" slab"
  "Vapor barrier, 6 mil"
  "Wood framing, 2x10 @ 16\" o.c., SPF"
  "Plywood subfloor, 3/4\" thick"
  "Insulation, R-30, at floor"
  "Fiberglass batt insulation, R-13, at 4\" wall"
  "Gypsum board, 1/2\""
  "Wall cove base, 4\""
  "Top plate, double, with bottom plate, 2x4"
  "Wood studs, 2x4, @ 16\" o.c."
  "Anchor bolts, 1/2\" dia."
  "OSB board sheathing, 3/4\""
  "Exterior siding, vertical"
  "Circuit breaker lock out device, multi-pole, 15 to 225 Amp"
  "Excavator, diesel hydraulic, crawler mounted, 1-1/2 CY capacity"
  "Refrigerator, Amana Model 7184596"
  "Hand sink, Serin Wire Model HS10-OMP"
  "Steel door, 3'-3\" W x 8'-0\" H x 0'-1 3/4\" T, steel material, painted finish, satin chrome hardware"

- If the original "notes" is already terse and matches this style, "notes2" should be identical (or nearly identical) to "notes" — just trimmed of any code/count/reference-to-detail/empty-field/multi-location if present.
- Do NOT invent or add any new information that isn't already in "notes".
- If "notes" is empty, "notes2" should also be an empty string.

CRITICAL — never drop measurements or strength values. Any dimension, thickness, spacing, or size expressed in inches, feet, mil, mm, cm, gauge, or fraction form (e.g. 3/4", 1/2" dia., 4", 16" o.c., 6 mil, R-30, R-13) and any strength/grade value (e.g. 3000 psi, Grade 60, #SPF, 15 Amp, 1-1/2 CY) MUST be carried over into "notes2" exactly as written. These are never "counts" — only remove an actual quantity-of-items count (e.g. "3 units", "qty: 5", "x4 doors") and only remove a code/mark/tag reference (e.g. "Door-D1", "per mark W1"). When in doubt about whether a number is a count vs. a measurement, treat it as a measurement and keep it.

Respond with ONLY a JSON array of objects, no other text, no markdown formatting, no code fences, in this exact format:
[{{"id": <id>, "notes2": "<shortened notes>"}}, ...]

You must include every id from the input list exactly once."""

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

    result_map = {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(f"  [Warning] Could not parse Claude response for notes2, keeping original notes. Raw response: {text[:200]}")
        return result_map

    orig_notes_by_idx = {orig_idx: item.get("notes", "") for orig_idx, item in items_with_idx}

    for entry in parsed:
        try:
            local_id = entry["id"]
            notes2 = entry["notes2"]
        except (KeyError, TypeError):
            continue
        if 0 <= local_id < len(items_with_idx):
            orig_idx = items_with_idx[local_id][0]
            orig_notes = orig_notes_by_idx.get(orig_idx, "")
            if measurements_dropped(orig_notes, notes2):
                print(f"  [Safety] Measurement/strength token dropped for idx {orig_idx}; keeping original notes.")
                notes2 = orig_notes
            result_map[orig_idx] = clean_notes2(notes2)

    return result_map


def add_notes2(materials: list) -> None:
    """ Adds a "notes2" key immediately after "notes" for every material item, using Claude Haiku to strip out code/mark references and counts from the original "notes". Falls back to a copy of "notes" if Claude fails to return a value for a given item. """
    indices = [i for i, item in enumerate(materials) if isinstance(item, dict)]

    notes2_map = {}
    for start in range(0, len(indices), NOTES2_BATCH_MAX_ITEMS):
        chunk_indices = indices[start:start + NOTES2_BATCH_MAX_ITEMS]
        items_with_idx = [(i, materials[i]) for i in chunk_indices]
        notes2_map.update(ask_claude_for_notes2(items_with_idx))

    for idx in indices:
        item = materials[idx]
        original_notes = item.get("notes", "")
        notes2_value = notes2_map.get(idx, clean_notes2(original_notes))

        # Rebuild the dict so "notes2" is placed immediately after "notes", skipping any pre-existing "notes2" key already in the input item (e.g. if this file was already run through this script before) —
        # otherwise that stale key gets copied later in this same loop and silently overwrites the freshly computed notes2_value.
        new_item = {}
        inserted = False
        for k, v in item.items():
            if k == "notes2":
                continue
            new_item[k] = v
            if k == "notes":
                new_item["notes2"] = notes2_value
                inserted = True
        if not inserted:
            # "notes" key wasn't present at all; just append both
            new_item["notes"] = original_notes
            new_item["notes2"] = notes2_value

        materials[idx] = new_item


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

    print("Generating shortened 'notes2' for each material (via Claude Haiku)...")
    add_notes2(final_materials)
    print("Done generating 'notes2'.")

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    pdf_name = get_pdf_name(INPUT_PATH)
    output_file = os.path.join(RESULTS_FOLDER, f"{pdf_name}_Final_3.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_materials, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(final_materials)} material(s) to: {output_file}")


if __name__ == "__main__":
    main()