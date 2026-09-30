import os
import re
import json
import sys
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_INPUT_PATH = r"D:\qtakeoffai-AI\qtakeoff-ai-AI\local\results\REQUIRED_Final.json"

RESULTS_FOLDER = os.path.join("local", "results")

BATCH_MAX_ITEMS = 40
NOTES_BATCH_MAX_ITEMS = 40

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


def ask_claude_for_material_names(items_with_idx: list) -> dict:
    if not items_with_idx:
        return {}

    local_list = []
    for local_pos, (orig_idx, item) in enumerate(items_with_idx):
        local_list.append({
            "id": local_pos,
            "code": item.get("code", ""),
            "estimation_notes": item.get("estimation_notes", "")
        })

    prompt = f"""You are extracting product/material names from construction schedule entries.

    Here is a list of entries (id, code, estimation_notes). The "code" field is just a mark/tag (e.g. "B3", "W1", "FL-1", "Door-D1") and NOT a usable name.

    {json.dumps(local_list, indent=2, ensure_ascii=False)}

    "estimation_notes" is built from a schedule table row and generally contains several labeled fields separated by commas (e.g. "ITEM: ..., SIZE: ..., MATERIAL: ..., NOTES: ..., MANUFACTURER/MODEL: ..."
    or "TYPE: ..., BRAND/MANUFACTURER: ..., STYLE/COLOR/SIZE/FINISH: ..."). The exact field labels vary between schedules/PDFs, but the underlying structure is always the same:

    - ONE field tells you WHAT KIND OF PRODUCT this is — the core noun / category of the thing itself. This field is commonly labeled "ITEM", "TYPE", "PRODUCT", "DESCRIPTION", or similar (e.g. "NATURAL STONE - GRANITE", "CERAMIC TILE", "RUBBER COVE BASE", "BRICK", "EXT OPENING CASING"). This is the anchor noun of the name — NEVER drop or replace it.
    - The OTHER fields (commonly labeled "MATERIAL", "BRAND/MANUFACTURER", "STYLE/COLOR/SIZE/FINISH", "NOTES", "MANUFACTURER/MODEL", "SIZE", etc.) describe or qualify that product — a material composition, brand, finish, color, or style.

    First identify which labeled field in each entry's "estimation_notes" is the "what kind of product" field (the anchor noun), then build a short, clean name in the form "<anchor noun>".

    Examples:
    - "ITEM: BRICK, MATERIAL: SMOOTH BRICK" -> "Brick (F-30)" #Here the code reference is F-30, so we add it in the name to make it unique. If there is no code reference, then we do not add it in the name.
    - "ITEM: EXT OPENING CASING, MATERIAL: FIBER CEMENT" -> "Ext Opening Casing"
    - "TYPE: NATURAL STONE - GRANITE, STYLE/COLOR/SIZE/FINISH: UBATUBA GRANITE - 24\\" X 24\\" POLISHED" -> "Ubatuba Granite"
    - "TYPE: CERAMIC TILE, STYLE/COLOR/SIZE/FINISH: PORTLAND STONE BEIGE - ANTI SLIP" -> "Ceramic Tile"
    - "TYPE: RUBBER COVE BASE, STYLE/COLOR/SIZE/FINISH: TAUPE - 6\\" HIGH" -> "Rubber Cove Base"
    - "ITEM: 01, DESCRRPTION: 56mm wide Wooden Door, MATERIAL: SOLID WOOD, STYLE/COLOR/SIZE/FINISH: OAK - NATURAL FINISH" -> "Wooden Door" # If they are just numbers, do not put them in name
    - "ITEM: 05, Description: Sealing Gasket- 6mm THK Rubber Pipe, DRAWAING NUMBER: AAD-M 22-30-003" -> "Sealing Gasket"
    - "NOTE: C1, DESCRIPTION: Wall Cabinet, 12" depp, Plywood with adjustable shelves" -> "Wall Cabinet (C1)"  # If code is present, put them in name
    - "TYPE MARK: B4, SIZE: 3-2x14, MATERIAL: SPRUCE PINE FIR" -> "Spruce Pine Fir Beam (B4)"
    - "NOTE: GB, DESCRIPTION: Optional Accessible Grab Bar" -> "Grab Bar (GB)"
    - "NOTE: C7, Description:Toe Kick" -> "Toe Kick (C7)"
    - "Foundation Size: F2.0, Length: 2'-0\", Width: 2'-0\", Thickness: 12\", Reinforcement: (3) #4 BOTT. EW" -> "Foundation (F2.0)" 

    Remember this does not apply for Door and Window Schedule. If door and Window schedule are present, the names must be Door-Doorname and Window-Windowname.
    Do not add the code two/ three times in name. Just add it once.

    Rules:
    - The name should be a short noun phrase (a few words), NOT a full sentence.
    - The anchor "what kind of product" noun MUST appear in the name — never output just a material/brand/style/finish value alone with the product type dropped.
    - Do NOT include the mark/tag/code itself in the name.
    - If no field can be identified as the "what kind of product" field, fall back to whatever descriptive text is available (do not fabricate). If nothing usable can be extracted at all, use the "code" value.

    Respond with ONLY a JSON array of objects, no other text, no markdown formatting, no code fences,
    in this exact format:
    [{{"id": <id>, "name": "<extracted name>"}}, ...]

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
        print(f"  [Warning] Could not parse Claude response for code->name, falling back to 'code' value. Raw response: {text[:200]}")
        return result_map

    for entry in parsed:
        try:
            local_id = entry["id"]
            name_value = entry["name"]
        except (KeyError, TypeError):
            continue
        if 0 <= local_id < len(items_with_idx):
            orig_idx = items_with_idx[local_id][0]
            result_map[orig_idx] = str(name_value).strip()

    return result_map


def normalize_code_to_name(materials: list) -> None:
    """For every item whose schema has 'code' instead of 'name':
    1. Extract the material name from 'estimation_notes' via Claude.
    2. Put that extracted name as the value of the 'code' key.
    3. Append 'REFERENCE: <initial code>' to 'estimation_notes' so the original code isn't lost.
    4. Rename the 'code' key to 'name'.
    Modifies `materials` in place.
    """

    indices = [
        i for i, item in enumerate(materials)
        if isinstance(item, dict) and "code" in item and not str(item.get("name", "")).strip()
    ]

    if not indices:
        return

    name_map = {}
    for start in range(0, len(indices), BATCH_MAX_ITEMS):
        chunk_indices = indices[start:start + BATCH_MAX_ITEMS]
        items_with_idx = [(i, materials[i]) for i in chunk_indices]
        name_map.update(ask_claude_for_material_names(items_with_idx))

    for idx in indices:
        item = materials[idx]
        original_code = str(item.get("code", "")).strip()
        extracted_name = name_map.get(idx, "").strip() or original_code

        # Bake the original mark/tag into the name itself, e.g. "Astragal (F-30)".
        # dedupe_materials() groups purely on (name, category) -- without the
        # code baked in, two schedule rows that happen to get the same generic
        # descriptive name look identical to the dedup logic and one gets
        # silently deleted. Baking the code in guarantees every row's name is
        # unique per mark.
        display_name = f"{extracted_name} ({original_code})" if original_code else extracted_name

        original_notes = str(item.get("estimation_notes", "") or "").strip()
        reference_tag = f"referenced from {original_code}" if original_code else ""
        if reference_tag:
            new_notes = f"{original_notes}, {reference_tag}".strip(", ").strip() if original_notes else reference_tag
        else:
            new_notes = original_notes

        new_item = {}
        for k, v in item.items():
            if k == "code":
                new_item["name"] = display_name
            elif k == "estimation_notes":
                new_item["estimation_notes"] = new_notes
            else:
                new_item[k] = v

        if "estimation_notes" not in new_item and new_notes:
            new_item["estimation_notes"] = new_notes

        materials[idx] = new_item


def ask_claude_for_paraphrase_duplicates(items_with_idx: list) -> list:

    if len(items_with_idx) < 2:
        return []

    local_list = []
    for local_pos, (orig_idx, item) in enumerate(items_with_idx):
        local_list.append({
            "id": local_pos,
            "name": item.get("name", ""),
            "estimation_notes": item.get("estimation_notes", "")
        })

    prompt = f"""You are comparing construction material entries that all share the same category.

    Here is a list of entries (id, name, notes):

    {json.dumps(local_list, indent=2, ensure_ascii=False)}

    Your task: identify which entries describe the SAME underlying material, where the "name" and/or "estimation_notes" are just paraphrases, synonyms, or reworded versions of each other (not genuinely different materials).

    Rules:
    - Only group entries together if they clearly refer to the same material, just worded differently.
    - Do NOT group entries that describe genuinely different materials, even if related.
    - Some "name" values end with a mark/tag code in parentheses, e.g. "Beam (B1)", "Astragal (F-30)". If two entries have DIFFERENT codes in parentheses, NEVER group them together -- even if the rest of the name and estimation_notes look identical or near-identical. Each distinct code represents a separate schedule entry that must be preserved on its own, regardless of how similar the wording is.
    - Singletons (materials with no duplicate) should NOT appear in your output at all.
    - Each id can belong to at most one group.
    - For each group, decide which single id should be KEPT: prefer the entry whose "estimation_notes" contain concrete numerical information (dimensions, thickness, sizes, etc). If more than one entry has numerical info, or none of them do, keep whichever entry has the more complete/detailed information overall.

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
            numeric_ids = [i for i in orig_ids if has_numeric_info(item_by_orig_idx[i].get("estimation_notes", ""))]
            candidates = numeric_ids if numeric_ids else orig_ids
            keep_idx = max(candidates, key=lambda i: len(item_by_orig_idx[i].get("estimation_notes", "") or ""))

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

        # Combine every mention from every duplicate in this group onto the
        # survivor before deleting the rest, so location/view/page data from
        # the removed rows isn't silently lost -- only the row itself goes,
        # not the evidence of where it appeared.
        materials[best_idx]["mentions"] = merge_mentions(materials, indices)

        for i in indices:
            if i != best_idx:
                to_remove.add(i)

    return to_remove, duplicate_group_count


_MEASUREMENT_RE = re.compile(
    r"""
    \d+(?:/\d+)?(?:-\d+/\d+)?\s*(?:inch(?:es)?\b|in\.?(?!\w)|"|'|ft\.?\b|mm\b|cm\b|mil\b|ga\.?\b|gauge\b)   # 3/4", 1/2 in, 6 mil, 16 ga
    |
    \bR-\d+(?:\.\d+)?\b # R-30, R-13
    |
    \d+(?:,\d{3})*(?:\.\d+)?\s*(?:psi|ksi)\b # 3000 psi, 4 ksi
    |
    \bGrade\s*\d+\b# Grade 60
    |
    \d+\s*(?:o\.c\.|oc)\b # 16" o.c. (backup, catches trailing oc)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NON_DIMENSION_LABELS = re.compile(
    r"\b(?:volts\s*/\s*phase|phase|amps?|voltage)\s*:\s*[^,]*", re.IGNORECASE
)

_ROOM_DIMENSION_RE = re.compile(
    r"[^.,;]*\bceiling height\b[^.,;]*[.,;]?", re.IGNORECASE
)

# should drop (e.g. "30\" to ground", "90\" above ceiling", "20m along wall").
_LOCATION_DIMENSION_RE = re.compile(
    r"""[^.,;]*\b(?:
        to\s+ground|to\s+(?:the\s+)?floor|
        above\s+(?:the\s+)?ceiling|above\s+(?:the\s+)?grade|above\s+grade|
        below\s+(?:the\s+)?(?:floor|grade|ceiling|slab)|
        under(?:neath)?\s+(?:the\s+)?ground|
        from\s+(?:the\s+)?floor|off\s+(?:the\s+)?floor|
        along\s+(?:the\s+)?(?:length\s+of\s+)?wall|
        in\s+length\s+of\s+wall|
        above\s+finish(?:ed)?\s+floor|below\s+finish(?:ed)?\s+floor|
        a\.?f\.?f\.?|
        height\s+to\s+(?:the\s+)?(?:ground|floor)
    )\b[^.,;]*[.,;]?""",
    re.IGNORECASE | re.VERBOSE,
)


def _strip_non_dimension_segments(text: str) -> str:
    text = _NON_DIMENSION_LABELS.sub("", text or "")
    text = _ROOM_DIMENSION_RE.sub("", text)
    text = _LOCATION_DIMENSION_RE.sub("", text)
    return text


_WS_RE = re.compile(r"\s+")


def _normalize_measurement_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u201d", '"').replace("\u2019", "'")# curly quotes -> straight
    text = text.replace("\u2032", "'").replace("\u2033", '"') # prime marks -> straight
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _extract_measurements(text: str) -> set:
    text = _normalize_measurement_text(_strip_non_dimension_segments(text))
    return {_normalize_measurement_text(m).lower() for m in _MEASUREMENT_RE.findall(text)}


def measurements_dropped(orig_notes: str, notes_value: str) -> bool:
  
    orig_tokens = _extract_measurements(orig_notes)
    if not orig_tokens:
        return False
    notes_lower = _normalize_measurement_text(notes_value).lower()
    for token in orig_tokens:
        if token not in notes_lower:
            return True
    return False


_STRIP_WORDS_RE = re.compile(r"\b(assembly|assemblies|detail|details)\b", re.IGNORECASE)


def clean_notes(notes_value: str) -> str:

    if not notes_value:
        return notes_value

    cleaned = _STRIP_WORDS_RE.sub("", notes_value)

    # Tidy up leftover punctuation/whitespace from removed words
    cleaned = re.sub(r"\s{2,}", " ", cleaned) # collapse double spaces
    cleaned = re.sub(r"\s+,", ",", cleaned)# space before comma
    cleaned = re.sub(r",\s*,", ",", cleaned)#double commas
    cleaned = re.sub(r",\s*$", "", cleaned) #trailing comma
    cleaned = re.sub(r"^\s*,\s*", "", cleaned)#leading comma
    cleaned = cleaned.strip()

    return cleaned


_HEIGHT_SEGMENT_RE = re.compile(r",\s*[^,]*\bheight\b[^,]*", re.IGNORECASE)
_LEADING_HEIGHT_SEGMENT_RE = re.compile(r"^\s*[^,]*\bheight\b[^,]*,?\s*", re.IGNORECASE)


def _strip_ceiling_height_from_notes(orig_notes: str, notes_value: str) -> str:
    """Ceiling height is a room dimension, never a material spec — even for a ceiling material itself. If the original notes mention "ceiling height", make sure no leftover height segment survives in notes_value, regardless of what Claude decided to keep."""
    if "ceiling height" not in (orig_notes or "").lower():
        return notes_value
    if not notes_value:
        return notes_value

    cleaned = _HEIGHT_SEGMENT_RE.sub("", notes_value)
    cleaned = _LEADING_HEIGHT_SEGMENT_RE.sub("", cleaned)
    cleaned = cleaned.strip().rstrip(",").strip()
    return cleaned or notes_value


def ask_claude_for_notes(items_with_idx: list) -> dict:
   
    if not items_with_idx:
        return {}

    local_list = []
    for local_pos, (orig_idx, item) in enumerate(items_with_idx):
        local_list.append({
            "id": local_pos,
            "estimation_notes": item.get("estimation_notes", "")
        })

    prompt = f"""You are cleaning up "estimation_notes" fields for construction material entries.

    Here is a list of entries (id, notes):
    {json.dumps(local_list, indent=2, ensure_ascii=False)}

    Your task: for each entry, produce a shorter version of "estimation_notes" called "notes" by:
    - Removing any reference to a code, mark, or tag (e.g. "Door-D1", "Window-02", "per mark W1", "type A3", "ref: F-12", "wall types B1 and B2", "wall type B1"). These codes identify a specific drawing element, not a material property — drop them entirely along with connecting words like "of" that only existed to introduce them.
    - "Brick veneer up to 4' height on exterior side of wall types B1 and B2." → drop "wall types B1 and B2" (a code reference), rewrite as terse catalog phrase → "Brick veneer, up to 4' height, exterior side"
    - Removing any mention of the count/quantity/number of that material (e.g. "3 units", "qty: 5", "x4", "5 pieces", "count: 2", "each 12.5 ft.").
    - KEEPING model numbers exactly as written if present but if the model name is present, capitalize the name of model. Example if notes has: AMANA MODEL as model name then the notes_value must be Amana Model.(e.g. "AMANA MODEL 7184596" should be Amana Model 7184596, "SERIN WIRE MODEL HS10-OMP" should be Serin Wire Model HS10-OMP) — do NOT remove these. You may drop a trailing "OR EQUAL" / "or approved equal" qualifier since it adds no information on its own.
    Example:
    "estimation_notes": "Qty: 1, Description: REFRIGERATOR, Item Specification: AMANA MODEL 7184596, OR EQUAL",
    "notes": "Refrigerator, Amana Model 7184596",

    ***CRITICAL***: If the note comes with locations at exterior wall, doors and windows then just add at exterior wall in the note. Exclude doors and windows. Example:
        - "Fiber cement lap siding at exterior wall assembly, at window and door head/sill and jamb details" is in notes sections so the "notes" must have: "Fiber cement lap siding at exterior wall" 
        but if doors and windows come alone without exterior wall, then keep it
        - "Fiber cement lap siding at window and door head/sill and jamb details" → drop only the drafting-callout part ("head/sill and jamb details") → "Fiber cement lap siding, at window and door"
    - Removing any reference to construction drawing details/callouts rather than the material itself — e.g. "sill detail", "jamb detail", "head detail", "lintel detail", "window and door head/sill and jamb details", "elevation detail", "plan detail", "pipe penetration detail", "to follow corner boards", "window lintel and jamb details", "niche detail conditions". These are drafting references, not material specs, and should be dropped entirely (along with any connecting words like "at", "per", "see" that only existed to introduce them).
    Example: 
    - "Fiber cement lap siding at window and door head/sill and jamb details" → drop only the drafting-callout part ("head/sill and jamb details") → "Fiber cement lap siding, at window and door"

    * IMPORTANT — do NOT over-strip: only drop the drafting-callout phrase itself. If the same sentence also names a real building location/component (e.g. "exterior wall", "window and door", "slab-on-grade", "roof edge", "metal stud-brick wall"), KEEP that location and only remove the callout part. The word "assembly"/"assemblies" attached to a location is handled separately (by code, not you) — leave it in notes_value exactly as written; do not delete the location just because "assembly" follows it.

        - "EXT SHEATHING W/ BUILDING WRAP at window and door details, exterior wall assembly" → keep only "exterior wall assembly"  → "Exterior sheathing with building wrap, at  exterior wall"
        - "Batt insulation used at exterior wall assemblies at window lintel and window jamb details at metal stud-brick wall" → keep "exterior wall" AND "metal stud-brick wall" (both are real locations), drop only "window lintel and window jamb details" → "Batt insulation, at exterior wall assemblies, metal stud-brick wall"
        - "2X6 SURROUND at window and door head/sill and jamb details" → "window and door" IS a real location (where the surround sits); drop only "head/sill and jamb details", keep "window and door" → "2x6 surround, at window and door"

    * Rule of thumb: words like "head", "head details", "sill", "sill details", "jamb", "jamb details", "lintel", "elevation", "plan", describe a drawing VIEW/callout and should always be dropped. Words like "exterior wall", "window and door" / "doors and windows", "slab-on-grade", "roof edge", "metal stud-brick wall" name a physical LOCATION/component and should always be kept, even when a drawing-view word or "detail(s)" immediately follows them. When SEVERAL such locations/components appear together in one entry (not a "Location:" room list — see below), keep ALL of them; they describe different parts of the same assembly the material touches, not interchangeable alternatives.

    - Removing any field/segment of "estimation_notes" whose value is empty, blank, "-", "N/A", "NA", or similar (e.g. if notes contains "Glazing: -" or "Glazing: N/A", drop that whole "Glazing: ..." segment — don't write "Glazing:" with nothing after it).
    - Location handling: If explicit "Location:" field is provided in ROOMS/SPACES (e.g. "Location: LOBBY, FOYER PERIPHERY, PRAYER HALL, DEITY PEDESTALS, NICHES", or plain "kitchen, bedroom, hallway") — if that field lists more than one room/space, drop the whole location field from "notes" entirely; if it lists only one room/space, keep it.
    * This does NOT apply to structural components/locations mentioned in ordinary prose (as opposed to a room list) — see the rule of thumb above. Keep every such structural location, no matter how many appear.
    - For doors and windows specifically, do NOT include frame type/frame material or other framing construction details in "notes" (e.g. drop "Frame Type: HM", "Frame Material: Steel frame") — keep the door/window's own size, material, finish, and hardware instead.
    - Remove all the reference phrases "from legend", "as per legend", "per legend", "as per plan", "as per detail", "as per schedule", "as per manufactuing plan", "installed per manufacturer instructions", etc from notes_value as they are not related to the material itself. Also, do not user a key inside the notes_value eg:
    
    "notes": "Description: water sizk, blue colour at bathroom"
    should be
    "notes": "Water sink, blue color, at bathroom"    #bathroom is kept as there is only a single location in the notes section.
    Here, the descrption key is removed from notes_value.

    - Do not include any drawing numbers in the notes_value key.
    - Keeping all other meaningful spec information intact: size, thickness, type, and strength (as well as spacing, grade, and finish if present).
    - Do not include any other mesaurements except Size and thickness of the material. If data comes such that "20m above the wall" or '90" above ceiling' or '10" below the floor' 'under the ground', etc then remove these measurements as they are not related to the material itself. Remember to just include SIZE of material or/and THICKNESS of material in notes_value.
    Example: "Bottom rail on PT block, porch ornament railing, less than 30\" to ground" → drop the installation-height phrase "less than 30\" to ground" entirely (it describes WHERE the rail sits, not its size) → "Bottom rail on PT block, porch ornament railing"
    -  Remove core/code requirement callouts from notes_value — phrases that state WHY a material is mandated (a regulatory or performance requirement) rather than describing the material itself. These are not a material property, drop them entirely along with connecting words like "per" or "as required by" that only existed to introduce them.
    Trigger phrases include (not exhaustive): "per code", "per code requirement", "code-required", "as required by code", "meets code", "per fire code", "per building code", "required per IBC/IRC/ADA", "to satisfy code requirement".
    Example: "Gypsum board sheathing, fire-rated, below winder stairs, per code fire-blocking requirement" → drop "per code fire-blocking requirement" (states a regulatory reason, not a material spec) → "Fire-rated gypsum board sheathing, below winder stairs"
    * Do NOT drop a real material property just because it's near a code reference — e.g. "fire-rated" and "below winder stairs" both describe the material/its location and must be kept; only the "per code ... requirement" clause itself is removed. 
    - If there is a spcial note inside the 'estimation_notes' then do not include it in notes_value if they are not a part of the material.
    - If there are any clauses related to climate, weather, or environmental conditions (e.g. as per the weather, as per climate, as per the environmental conditions, etc) then remove them from notes_value as they are not related to the material itself.
    - If all the information in notes are references, them copy the name of the material in notes_value and remove all the references. For example, if notes has:
    
    "name": "Water sink",
    "estimation_notes":"Reference: Drawing 1/A-101" then the notes_value should be 
    "notes":"Water sink"(same as name) as notes only have reference.

    - Use the name of table when necessary for notes for example,
    
        "name": "B3",
        "estimation_notes": "Type Mark: B3, Size: 3-2x14, Material: SPRUCE PINE FIR",
        "notes": "Beam B3, 2x14, Spruce Pine Fir", # State marck name as well.

    ⚠️ Do not add "note" inside notes_value. Instead, privide the note in a descriptive way. Eg for incorrect and correct ways:
    "notes": "Bargeboard running trim, Vintage Woodworks (VW), Mariposa 2229. Note: bargeboard shapes can be easy to custom cut.", ❌
    "notes": "Bargeboard running trim, Vintage Woodworks (VW), Mariposa 2229 whose shapes can be easy to custom cut.", ✅

    - If the extra information is provided which is not related to the material itself, then remove it. For example,
    Example 1:
        "name": "Vinyl Plank",
        "estimation_notes": "Floor material: Vinyl Plank, Clean finish. 9'-0\" ceiling height room. Provide wood shoe moulding at vinyl plank flooring; Greenguard certified vinyl plank required.",
        -> here, the material is Vinyl Plank and ceiling height is not related to the material so you can exclude the information related to ceiling height in notes_value
        "notes": "Vinyl plank flooring, clean finish, Greenguard certified, with wood shoe moulding",

    Example 2:
    "name": "Metal Stud (6\")",
    "estimation_notes": "6\" METAL STUD framing. Referenced in Wall Type B1: EXTERIOR WALL 6\" METAL STUD WITH ONE LAYER OF 5/8\" GYPSUM WALLBOARD AND FRP PANEL ON ONE SIDE AND OTHER SIDE WITH EXTERIOR SHEATHING WITH BRICK VENEER UPTO 4' AND REMAINING HEIGHT OF WALL WITH EXTERIOR SIDING PANEL, and Wall Type B2: EXTERIOR WALL 6\" METAL STUD WITH ONE LAYER OF 5/8\" GYPSUM WALLBOARD AND WALL FINISH ON ONE SIDE AND OTHER SIDE WITH EXTERIOR SHEATHING WITH BRICK VENEER UPTO 4' AND REMAINING HEIGHT OF WALL WITH EXTERIOR SIDING PANEL."
    "notes": "Metal stud, 6\"",          #all other infomation is not related to the material itself so it is removed from notes_value. All references are also removed.

    - If nothing can be extracted from the notes section, then use the name of the material as notes_value.
    - If any substitue value comes in the notes section, then remove it from notes_value as it is not related to the material itself. For example,
    {{
        "name": "Threaded Rod",
        "estimation_notes": "5/8\" threaded rods may be substituted for bent rebar and anchor bolts, grout all cells with threaded rods solid",
        "notes": "Threaded rods, 5/8\"",  ✅ The substitute part is removed for notes section.
        "category": "Wall-Foundation",
        "mentions": [
            {{
                "page_label": "S-302 - Typical Details 2",
                "view": "Masonary Stem Wall Details for Walls 48\" Long or Less"
            }}
        ],
    }},

    STYLE — write "notes" as a terse, comma-separated catalog phrase, in the same compact style used by RSMeans-type cost-database descriptions. NOT a full sentence: no "The", no subject/verb narrative, no trailing period. Lead with the core item/material type, then add comma-separated modifiers (size, thickness, type, strength, single location if present) in natural left-to-right order. Keep inch marks as the " symbol exactly as written in the original "estimation_notes" — do NOT spell out the word "inch". Examples of the target style:
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

    - If the original "estimation_notes" is already terse and matches this style, "notes" should be identical (or nearly identical) to "estimation_notes" — just trimmed of any code/count/reference-to-detail/empty-field/multi-location if present. "Already terse" means the original is short comma-separated fragments, NOT a full grammatical sentence — if "estimation_notes" reads as a sentence (has words like "on", "of", "at", verbs, articles like "the"/"a", or ends in a period), it does NOT qualify as already-terse and MUST be rewritten into the comma-separated catalog style, not just have its trailing period removed.
    - Do NOT invent or add any new information that isn't already in "estimation_notes".
    - If "estimation_notes" is empty, "notes" should also be an empty string.

    CRITICAL — never drop measurements or strength values. Any dimension, thickness, spacing, or size expressed in inches, feet, mil, mm, cm, gauge, or fraction form (e.g. 3/4", 1/2" dia., 4", 16" o.c., 6 mil, R-30, R-13) and any strength/grade value (e.g. 3000 psi, Grade 60, #SPF, 15 Amp, 1-1/2 CY) MUST be carried over into "notes" exactly as written. These are never "counts" — only remove an actual quantity-of-items count (e.g. "3 units", "qty: 5", "x4 doors") and only remove a code/mark/tag reference (e.g. "Door-D1", "per mark W1"). When in doubt about whether a number is a count vs. a measurement, treat it as a measurement and keep it.

    Respond with ONLY a JSON array of objects, no other text, no markdown formatting, no code fences, in this exact format:
    [{{"id": <id>, "notes": "<shortened notes>"}}, ...]

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
        print(f"  [Warning] Could not parse Claude response for notes, keeping original notes. Raw response: {text[:200]}")
        return result_map

    orig_notes_by_idx = {orig_idx: item.get("estimation_notes", "") for orig_idx, item in items_with_idx}

    for entry in parsed:
        try:
            local_id = entry["id"]
            notes_value = entry["notes"]
        except (KeyError, TypeError):
            continue
        if 0 <= local_id < len(items_with_idx):
            orig_idx = items_with_idx[local_id][0]
            orig_notes = orig_notes_by_idx.get(orig_idx, "")
            notes_value = _strip_ceiling_height_from_notes(orig_notes, notes_value)
            if measurements_dropped(orig_notes, notes_value):
                print(f"  [Safety] Measurement/strength token dropped for idx {orig_idx}; keeping original notes.")
                notes_value = orig_notes
            result_map[orig_idx] = clean_notes(notes_value)

    return result_map


def add_notes(materials: list) -> None:
    """ Adds a 'notes' key (cleaned/shortened text) right after 'estimation_notes' for each material. """
    indices = [i for i, item in enumerate(materials) if isinstance(item, dict)]

    notes_map = {}
    for start in range(0, len(indices), NOTES_BATCH_MAX_ITEMS):
        chunk_indices = indices[start:start + NOTES_BATCH_MAX_ITEMS]
        items_with_idx = [(i, materials[i]) for i in chunk_indices]
        notes_map.update(ask_claude_for_notes(items_with_idx))

    for idx in indices:
        item = materials[idx]
        original_notes = item.get("estimation_notes", "")
        cleaned_notes_value = notes_map.get(idx, clean_notes(original_notes))

        new_item = {}
        inserted = False
        for k, v in item.items():
            if k == "notes":
                continue
            if k == "estimation_notes":
                # 'estimation_notes' keeps the original text, 'notes' gets the shortened value
                new_item["estimation_notes"] = original_notes
                new_item["notes"] = cleaned_notes_value
                inserted = True
            else:
                new_item[k] = v

        if not inserted:
            new_item["estimation_notes"] = original_notes
            new_item["notes"] = cleaned_notes_value

        materials[idx] = new_item


def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = DEFAULT_INPUT_PATH

    if not input_path:
        raise ValueError(
            "No input file provided. Pass a file path as a command-line argument, e.g. `python remove_duplicates.py path/to/file.json`, or set DEFAULT_INPUT_PATH at the top of this script."
        )

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        materials = json.load(f)

    print(f"Loaded {len(materials)} material(s).")

    print("Converting 'code'-only entries to 'name' (via Claude Haiku)...")
    normalize_code_to_name(materials)
    print("Done converting 'code' -> 'name'.")

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

    print("Generating shortened 'notes' for each material (via Claude Haiku)...")
    add_notes(final_materials)
    print("Done generating 'notes'.")

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    pdf_name = get_pdf_name(input_path)
    output_file = os.path.join(RESULTS_FOLDER, f"{pdf_name}_Final_2.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_materials, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(final_materials)} material(s) to: {output_file}")


if __name__ == "__main__":
    main()