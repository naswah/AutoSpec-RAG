import os
import json
import re
import time
import base64
import cv2
import numpy as np
from anthropic import Anthropic
from state.graph_state import AgenticState
from tools.pdf_helpers import pdf_to_image
from config import SCALE_PATH


def merge_nearby_boxes(boxes, x_gap=15, y_gap=15):

    def expand(b):
        x, y, w, h = b
        return (x - x_gap, y - y_gap, x + w + x_gap, y + h + y_gap)

    def overlaps(a, b):
        ax0, ay0, ax1, ay1 = expand(a)
        bx0, by0, bx1, by1 = expand(b)
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    def union(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x0, y0 = min(ax, bx), min(ay, by)
        x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
        return (x0, y0, x1 - x0, y1 - y0)

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        result = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            current = merged[i]
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if overlaps(current, merged[j]):
                    current = union(current, merged[j])
                    used[j] = True
                    changed = True
            result.append(current)
        merged = result

    return merged


def detect_table_boxes(image_path, min_area_ratio=0.0015, page_no=None):
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ink_mask = cv2.inRange(hsv, (0, 0, 0), (180, 60, 255))
    thresh = cv2.bitwise_and(thresh, ink_mask)

    h, w = thresh.shape

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    grid = cv2.bitwise_or(horiz_lines, vert_lines)
    # Mild closing only -- bridges small gaps WITHIN one table's own grid lines,
    # without reaching far enough to connect separate tables/diagrams together.
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    img_area = h * w

    noise_floor = min_area_ratio * 0.05
    raw_boxes = []
    for i in range(1, n_labels):  # skip label 0 (background)
        x, y, cw, ch, area = stats[i]
        if (cw * ch) / img_area < noise_floor:
            continue
        raw_boxes.append((x, y, cw, ch))

    merged_boxes = merge_nearby_boxes(raw_boxes)

    boxes = []
    dropped = []
    for (x, y, cw, ch) in merged_boxes:
        box_ratio = (cw * ch) / img_area
        if box_ratio < min_area_ratio:
            dropped.append((x, y, cw, ch, box_ratio))
            continue
        boxes.append((x, y, cw, ch))

    boxes.sort(key=lambda b: (b[1], b[0]))  # top to bottom, then left to right

    page_tag = f"page {page_no}" if page_no is not None else "image"
    print(f"[Table OCR] {page_tag}: {len(raw_boxes)} raw fragment(s) -> {len(merged_boxes)} merged region(s) -> "
          f"{len(boxes)} kept, {len(dropped)} dropped below min_area_ratio.")
    for (x, y, cw, ch, box_ratio) in dropped:
        if box_ratio >= min_area_ratio * 0.3:
            print(f"    -> dropped candidate at ({x},{y}) size {cw}x{ch}, area_ratio={box_ratio:.5f} (cutoff={min_area_ratio})")

    return img, boxes


def crop_with_padding(img, box, padding=10):
    x, y, w, h = box
    H, W = img.shape[:2]
    x0, y0 = max(x - padding, 0), max(y - padding, 0)
    x1, y1 = min(x + w + padding, W), min(y + h + padding, H)
    return img[y0:y1, x0:x1]


def remove_color_stamps(cv2_img, sat_thresh=60, val_thresh=50):
    hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    stamp_mask = ((sat > sat_thresh) & (val > val_thresh)).astype(np.uint8) * 255
 
    if cv2.countNonZero(stamp_mask) == 0:
        return cv2_img
 
    # Dilate slightly so anti-aliased edges of the stamp strokes are fully covered.
    stamp_mask = cv2.dilate(stamp_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(cv2_img, stamp_mask, 3, cv2.INPAINT_TELEA)


def upscale_if_small(cv2_img, min_width=1400, min_height=700):
    h, w = cv2_img.shape[:2]
    scale = max(min_width / w, min_height / h, 1.0)
    if w * scale > 7800 or h * scale > 7800:
            scale = min(7800 / w, 7800 / h)
            
    if scale > 1.0:
        cv2_img = cv2.resize(cv2_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2_img


def encode_cv2_image_b64(cv2_img):
    ok, buf = cv2.imencode(".png", cv2_img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8"), "image/png"


def build_table_prompt():
    return '''You are transcribing content from a cropped region of a technical drawing/schedule sheet.

        IMPORTANT: This image may contain ONE OR MORE separate, independent tables (e.g. a Finish Schedule, a Door Schedule, and a Window Schedule placed side by side or stacked on the same sheet). Treat each visually distinct table -- i.e. each one with its own title and/or its own grid/header row -- as a SEPARATE table. Do NOT merge rows from different tables together, and do NOT skip any table just because one or more other tables are also present in the image.
        This also includes small "remark legend" tables (e.g. "FINISH SCHEDULE REMARK LEGEND", "DOOR SCHEDULE REMARK LEGEND") -- transcribe those as their own separate table too.

        FIRST, decide: does this image actually contain at least one data table (rows and/or columns of text/marks meant to be read as structured data)? If it is instead only a technical diagram, illustration, elevation drawing, or dimension callout figure (e.g. a drawing of a toilet with dimension arrows, NOT a table of rows/columns) -- it is NOT a table. In that case, return exactly: []
        Do not invent a table structure out of labels and dimension numbers found in a diagram.

        If it contains one or more genuine data tables, return a JSON array with ONE OBJECT PER TABLE, in this exact shape:
        [
        {
            "table_title": "<the title printed above/near this specific table, e.g. 'FINISH SCHEDULE',
                            'DOOR SCHEDULE', 'WINDOW SCHEDULE REMARK LEGEND'; if genuinely untitled, use a short descriptive label instead>",
            "records": [ <one JSON object per data row of THIS table only -- see structure rules below> ]
        }
        ]

        For each table's "records", decide independently which structure applies to THAT table:
        - If it is a normal row-based table: its FIRST ROW is the header row, containing the column names. Read that header row yourself and derive the JSON keys from it -- do not assume specific columns; the real columns could be anything (e.g. "Item No.", "Description", "Qty", "Unit Price"). Convert each header's text to a snake_case key (lowercase, spaces/punctuation replaced with underscores, e.g. "Item Specification" -> "item_specification", "S.N." -> "sn"). Use the SAME set of keys for every row object within that table.
        - If it is a MATRIX/checklist-style table (e.g. rooms or categories as column headers, item names as row headers, and marks like "X" at the intersections indicating which items apply to which column): represent each marked intersection as one JSON object with keys "row_label" and "column_label" (using the actual row/column header text, including any grouped/parent header if the header spans multiple levels, e.g. "Men's Restroom - Wall North").
          🚨 ROW-BLEED WARNING (this is the single most common mistake on dense matrix tables): it is very easy to accidentally shift a mark UP or DOWN by one row when the rows are thin and closely packed -- e.g. reading a mark that actually belongs to "Paint" as if it belonged to "Exterior Board" on the row below, or reading the last row of one vertical group (e.g. "Sealed Concrete" at the bottom of a "FLOOR" group) as if it were the first row of the NEXT group ("WALL"). To avoid this:
            (a) Process ONE row at a time. First read that row's own printed row_label text, then -- and only then -- scan strictly within that row's own horizontal band for marks. Do not carry a mark over from the row directly above or below.
            (b) When row labels are grouped under a shared vertical parent label spanning multiple rows (e.g. "FLOOR" bracketing both "Ceramic Tile (Anti Slip)" and "Sealed Concrete", with "WALL" bracketing the group below it), determine the parent-group boundary from the actual bracket/merged-cell extent in the image -- the LAST row inside a group still belongs to THAT group, never to the next one.
            (c) After finishing, re-verify each row_label independently: for every row printed on the left side, look back at the marks you assigned to it and confirm they are horizontally aligned with that exact row and no other.

        Instructions (apply to every table found):
        - Transcribe EVERY data row/mark in each table. Do not skip rows, do not skip the last row of any table.
        - Do NOT include header row(s) themselves as data entries -- they define the keys, not a row of data.
        - Use your knowledge of plausible values (e.g. short codes like "AD-3" or "G-5", model numbers being alphanumeric) to resolve any visually ambiguous characters (0 vs O, 1 vs I vs l, etc).
        - Preserve exact spelling, numbers, punctuation, and formatting as shown -- don't paraphrase or normalize wording.
        - Remove stray table-grid artifacts (e.g. leftover "|" pipe characters, stray quote marks) that are not part of the actual text.
        - No markdown code fences, no explanation, no preamble -- output must start with '[' and end with ']'.
        - COMPLETENESS FOR WIDE MATRIX/CHECKLIST TABLES: some matrix tables have many narrow columns (10+) and several rows. A row with only ONE or TWO marks (e.g. a row that is mostly empty except for a single "X" near the far edge) is just as important as a busy row, and is easy to skim past -- you MUST NOT skip it. Before finalizing your answer, mentally re-scan every row label printed down the left side, top to bottom, all the way to the LAST row -- do not stop early just because the remaining rows look sparse or mostly empty. If a printed row has zero marks anywhere, it is fine to omit it; but if it has even a single mark anywhere in the row, that mark must appear in your output.'''


def extract_table_json_array(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find a JSON array in model output:\n{text[:500]}")
    return json.loads(text[start:end + 1])


def call_claude_table_vision(client, image_b64, media_type, prompt, model="claude-haiku-4-5-20251001", max_tokens=8000):
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def extract_tables_from_page(client, page, table_prompt):
    
    try:
        img, table_boxes = detect_table_boxes(page["local_path"], page_no=page.get("page_no"))
    except FileNotFoundError:
        return []

    if not table_boxes:
        return []

    table_crops = [crop_with_padding(img, box) for box in table_boxes]
    table_crops = [remove_color_stamps(crop) for crop in table_crops]
    table_crops = [upscale_if_small(crop) for crop in table_crops]

    page_tables = []
    for i, crop in enumerate(table_crops):
        try:
            image_b64, media_type = encode_cv2_image_b64(crop)
            response_text = call_claude_table_vision(client, image_b64, media_type, table_prompt)
        except Exception as e:
            print(f"[Table OCR] Page {page['page_no']}, region {i}: skipped ({e})")
            continue

        stripped = response_text.strip()
        if stripped and not stripped.rstrip("`").rstrip().endswith("]"):
           
            print(f"[Table OCR] Page {page['page_no']}, region {i}: response appears TRUNCATED "
                  f"(did not end with ']'). Attempting to salvage complete row objects.")

        try:
            parsed = extract_table_json_array(response_text)
        except Exception as e:
            salvaged = salvage_truncated_array(response_text)
            if salvaged:
                print(f"[Table OCR] Page {page['page_no']}, region {i}: JSON parse failed ({e}), "
                      f"salvaged {len(salvaged)} object(s) from the truncated response.")
                parsed = salvaged
            else:
                print(f"[Table OCR] Page {page['page_no']}, region {i}: skipped ({e})")
                continue

        if not parsed:
            continue

        tables_in_region = []
        if all(isinstance(item, dict) and "records" in item for item in parsed):
            tables_in_region = parsed
        else:
            tables_in_region = [{"table_title": None, "records": parsed}]

        for t_idx, table in enumerate(tables_in_region):
            title = table.get("table_title")
            records = table.get("records") or []
            if not records:
                continue

            label = f"'{title}'" if title else "(untitled)"
            print(
                f"[Table OCR] Page {page['page_no']}, region {i}, table {t_idx} {label}: "
                f"transcribed {len(records)} row(s)"
            )
            print(f"[Table OCR] Page {page['page_no']}, region {i}, table {t_idx} {label} -- extracted content:")
            print(json.dumps(records, indent=2, ensure_ascii=False))

            page_tables.append({
                "source_region": i,
                "table_title": title,
                "records": records,
            })

    return page_tables


SCALE_DELIMITER = "===SCALE_DATA==="

SCALE_PROMPT = """
You are a professional construction drawing reviewer. Look at the drawing pages provided and identify the SCALE of every FLOOR PLAN and ELEVATION view only. Ignore schedules, notes pages, cover sheets, and detail-only pages unless a floor plan or elevation view also appears on that page.

WHAT TO LOOK FOR:
- A printed scale note near the title block or under a specific view, e.g. '1/4" = 1\'-0"', '1:100', '3/16" = 1\'-0"', 'SCALE: AS NOTED', 'NTS' (Not to Scale).
- A graphic scale bar (a small ruler-like bar with tick marks and distance labels).
- Some sheets show a DIFFERENT scale per view on the same sheet; report each view's scale separately.

For each floor plan / elevation view found, return one JSON object with EXACTLY these keys:

{
    "page_label": "<exact sheet number/title as printed, e.g. 'Sheet 4 of 23'>",
    "view": "<name of the view, e.g. 'East Elevation (Front)'>",
    "view_type": "FloorPlan" | "Elevation",
    "scale": "<scale exactly as printed, e.g. 1/4\\" = 1'-0\\" , or 'Not specified'>",
}
Example output:
[
    {
        "page_label": "Sheet 4 of 23", 
        "view": "East Elevation (Front)", 
        "view_type": "Elevation", 
        "scale": "1/4\\" = 1'-0\\"", 
    },
    {
        "page_label": "Sheet 16 of 23", 
        "view": "Main Floor Plan Layout", 
        "view_type": "FloorPlan", 
        "scale": "Not specified", 
    }
]

If NO floor plan or elevation view appears anywhere in the pages given, output exactly: NONE
Do not include any other text, headers, or explanations. One line per view only.
"""


def salvage_truncated_array(raw_text):
    objs = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(raw_text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(raw_text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return objs


def merge_duplicate_materials(items):
    merged = {}
    order = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("name", item.get("code", ""))).strip().lower(),
            str(item.get("category", "")).strip().lower(),
            str(item.get("notes", "")).strip().lower(),
        )
        if key not in merged:
            merged[key] = dict(item)
            merged[key]["mentions"] = list(item.get("mentions", []))
            order.append(key)
        else:
            existing_mentions = merged[key].setdefault("mentions", [])
            seen = {json.dumps(m, sort_keys=True) for m in existing_mentions}
            for m in item.get("mentions", []):
                m_key = json.dumps(m, sort_keys=True)
                if m_key not in seen:
                    existing_mentions.append(m)
                    seen.add(m_key)
    return [merged[k] for k in order]


def build_table_reference_text(page_no, page_tables):

    has_matrix_table = any(
        isinstance(r, dict) and ("row_label" in r or "column_label" in r)
        for t in page_tables
        for r in (t.get("records") or [])
    )

    header = f"===== OCR-TRANSCRIBED TABLE DATA FOR PAGE {page_no} (reference only -- use this to resolve exact row values) ====="
    if has_matrix_table:
        header += (
            "\nNOTE: one or more tables below use the row_label/column_label matrix format "
            "(see CATEGORY F). Convert EACH pair into its own separate output object -- do NOT "
            "merge multiple pairs into one combined summary sentence, and do NOT add or omit any "
            "room/surface that isn't explicitly present in the pairs below."
        )

    return f"{header}\n{json.dumps(page_tables, ensure_ascii=False)}"


def ingestion_agent_node(state: AgenticState):
    node_start = time.time()
    print(f"\n[Agent 1: Ingestion] Extracting Blueprint Views")
    client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    
    valid_pages = pdf_to_image(state["pdf_path"], state["output_base"])
    results = []
    scale_results= []
    
    prompt = """
    Your role is a professional construction material estimator. Analyze this architectural drawing and extract building materials used in CIVIL ENGINEERING and structural construction materials and schedule references. Read the rules below and provide only the VALID JSON output strictly following the specified format.

    WHAT TO EXTRACT:
    Only extract actual physical materials or products used in CIVIL ENGINEERING. For example:
    - If a note mentions an exterior wall made of "8' Concrete Foundation Wall, 4000 PSI", extract "Concrete Foundation Wall" for name and "8' Concrete Foundation Wall, 4000 PSI" for notes, "Foundation-Wall" for category. 
    - Instead of extracting "Front Porch", look for specific material callouts inside that porch zone (e.g. "CMU Block foundation", "Cast-in-place Concrete Slab").
    - Instead of extracting "Interior Partition Walls", look for the actual materials: "5/8" Type X Gypsum Board"(For name key, write Gypsum Board and for notes, add the sizes), "2x4 Wood Studs"(For name key, write Wood Studs and for notes, add the size), or "Light-Gauge Metal Stud Framing".
    - Do not take dimensions as codes.
    - If the material name is 'black asphalt shingles' then write the name of material as 'Asphalt Shingles'. Mention the color and other specifications in the 'notes' section.
    - Structural & Framing Materials: E.g., "2x12 Joists", "4x12 Glulam Beam", "Lookout Rafter", "Chamfered 5x5 Post", wood studs, headers, and plates.
    - Exterior Trim & Roof Components: E.g., "Fascia Board", "Frieze Board", "Shed Roof assemblies", gutters, drip edges.
    - Window & Door Details: E.g., "Fiber Cement Subsills", "Exterior Surrounds", "Door Frames", casing, and moldings.
    - Layered Finishes: E.g., "Gypsum Wallboard", "T&G Decking", vapor barriers, and "Air Space" ventilation gaps.
    - Tagged equipment/fixtures (Category E).
    - If a material is mentioned multiple times, write it only once. Strictly avoid duplicates. A material is considered identical if it has the same name, notes, and category. The notes could be paraphrased. Keep attention to those and do not rewrite the duplicates. If the same material is used in different locations (e.g., "Gypsum Board" in both "Room- Kitchen" and "Room- Bathroom"), list it separately for each location with the same name and notes but different category.

    ❗WHAT NOT TO EXTRACT:
    - In drawing labelling, if you see labelled materials that are not actually used in the construction, civil engineering, do not extract them.
    - Do not extract 'Air Space' as a material. It is a gap between two materials.
    Example:
     {
        "name": "3/4\" Air Space",
        "notes": "3/4\" air space between siding and sheathing, at window and door details",
        "category": "Exterior Wall",
        "mentions": [
            {
                "page_label": "Sheet A4.3 - Doors & Windows",
                "view": "Window Detail @ Head & Sill"
            },
            {
                "page_label": "Sheet A4.3 - Doors & Windows",
                "view": "Exterior Door Head & Sill"
            }
        ],
    }, This is not valid so do not extract it.

    ⚠️ Even if a material is not listed in a high-level Schedule (e.g., if it is only labeled on a typical section detail or elevation callout, such as a "2x10 wood mantle" or a "Double Wall Chimney Connector"), you MUST extract it as an individual material.
    
    ❗❗❗EXTRACT ALL CIVIL, STRUCTURAL AND ARCHITECTURAL FINISH MATERIALS❗❗❗
    (Exception: tagged fittings/accessories/fixture schedule rows -- see CATEGORY E below -- ARE in scope and must be extracted even though they are fixtures/hardware rather than structural materials.)

    🚨 CROSS-PAGE CODE RESOLUTION (CRITICAL):
    You are being given MULTIPLE PAGES from the same drawing set in this single request. Each image is preceded by a marker like "===== INTERNAL PAGE INDEX 7 =====". That marker is ONLY for you to keep track of which image you are looking at while reading — for the actual "page_label" field, always read the real sheet number/title printed in the page's own title block, exactly as you did before.
    - If a code (e.g., F-26, W1, X-02) appears on a plan, elevation, or detail page WITHOUT a full material description next to it, you MUST look through the OTHER pages provided in this same request for the schedule, legend, or detail table that actually defines that code (e.g., a "Window Schedule", "Door Schedule", "Materials Schedule", or detail callout table), and copy the FULL description found there into "notes".
    - NEVER write a vague placeholder describing the act of referencing, such as "Window/shutter code referenced in Window Elevation Details" or "See schedule for details." That is not a material description and is useless downstream. "notes" must always contain the actual material/product description — what it IS, not where else it is mentioned.
    - If you genuinely cannot find the defining schedule/table for a code anywhere in the pages provided in this request, fall back to whatever partial description appears directly next to the code on the page itself (dimensions, material hints, etc.). Only if there is truly zero descriptive text anywhere should "notes" be left empty — never fill it with a description of the reference itself.
    
    REMEMBER TO: - Extract where the materials are located for 'category' key into fixed categories: "Interior Wall", "Exterior Wall", "Door", "Window", "Roof", "Room- RoomName", "Room- Typical", "Foundation-Wall", " Foundation-Room" or "Others"  Do not add any other categories by yourself.

    If that information is explicitly provided in the notes or schedules. If the materials applied in room, read the name of the room and provide that as the location context (e.g., "Room- Kitchen Floor", "Room- Bathroom Walls", "Room- Living Room", "Room- Front Porch") in the category key.
    - For category, if the drawing has no clear information, 'notes' could also be read for adding category. For example, if notes section has the descrption: 'Engineered Trusses @ 24 O.C. per layout. Part of Porch Roof Assembly (R2).' Then the category could be 'Roof' because of the mention of porch roof assembly in the notes.
    - For category, if 'notes' section has anything written as 'Typical Room Assembly' then it the category key must has the value 'Room- Typical' because it is generic and is applied to all  rooms.
    - You can see the drawing for category field. Example, if Asphalt Shingles are labelled in roof area of the drawing then the category must be 'Roof'. If the drawing has no clear information, 'notes' could also be read for adding category. For example, if notes section has the descrption: 'Engineered Trusses @ 24 O.C. per layout. Part of Porch Roof Assembly (R2).' Then the category could be 'Roof' because of the mention of porch roof assembly in the notes.
    - CRITICAL: For 'category', if all the mentions have same category then provide the category normally but you must also look at the menstions page. for eg, if mentions has many category like:
        "mentions": [
            {
                "page_label": "Sheet A4.3 - Doors & Windows",
                "view": "Window Detail @ Jamb"
            },
            {
                "page_label": "Sheet A4.3 - Doors & Windows",
                "view": "Exterior Door Detail @ Jamb"
            }
        ],
    then the category key must look like:
    "category": {
            "c1": "Window", 
            "c2":"Door"
    }
    
    ❗IMPORTANT: Never truncate JSON. 
        - Close all { } braces.
        - Close all [ ] brackets.
        - Use double quotes for all keys and string values.

    🚨 REGEX RULE FOR CODES (CRITICAL)
    For codes (e.g., X-02, F-60, F-62, W1, F1, R2, etc), go automatically to category B.

    ⚠️ DO NOT SKIP DETAIL VIEWS:
    You must inspect every plan detail, wall section, and structural elevation. Small label callouts pointing to specific assembly layers (e.g."Double Wall Chimney Connector") contain critical material information and must be extracted. 
   
    🚨CRITICAL🚨: LEGEND EXCLUSION RULE:
    - IF YOU SEE STANDALONE KEY NOTES, GENERAL LEGENDS, OR INDEX KEYS LOCATED ON THE SAME PAGE AS PLAN VIEWS, YOU MUST IGNORE THEM. Do not parse these static master legends as views or schedules.
    - If there are no marks, then do not add them. It is not compulsory.

    ### EXTRACTION & DE-DUPLICATION RULES
    To avoid data duplication, your output must group multiple page or view references of the exact same material together inside a "mentions" array.

    - **Unique Material Criteria**: A material item is considered identical if it shares the exact same `name` (or `code`), `category`, and `notes`. Do not extract them more than once. If the same material appears in multiple locations, combine all page/view references into a single `mentions` array for that material.
    - **Variation Handling**: If the same material `name` appears elsewhere but has a different `category` or different `notes`, it MUST be listed as a completely separate object in the main list.
    
    #### CATEGORY A: STANDARD MATERIALS (No Codes Present)
    Use this formatting if there is absolutely no schedule code (like F-60 or X-02) associated with the material.
    - Provide "name", "notes", "category" and "mentions". Do NOT include a "code" key.

    Example:
    {
        "name": "Gypsum Drywall",
        "notes": "1/3' Gypsum Drywall",
        "category": "Interior Wall",
        "mentions": [
          {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
          {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
        ]
    },
    {
        "name": "Gypsum Drywall",
        "notes": "1/3' Gypsum Drywall",
        "category": "Roof",
        "mentions": [
          {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
          {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
        ]
    },

    Notice that thge same material must be listed twice because the category is different. If same material isused but of different size (e.g., 1/2' vs 5/8' Gypsum Drywall) then they must be listed as separate items because the notes are different.

    #### CATEGORY B: CODED MATERIALS & SCHEDULES (Codes Present)
    Use this formatting if a code (e.g., F-60, X-02) is detected anywhere on the drawing or inside a schedule layout.
    
    1. If it's on a plan view/detail pointing to a layout area:
       - You MUST strip out the "name" key completely. Only use the "code" key with the exact code (e.g., F-60, X-02) as it appears on the drawing.
         - The "notes" key should include the full material description as it appears in the note or schedule.

    2. DETECTING FULL SCHEDULES & TABLES (e.g., MATERIALS SCHEDULE, FIXTURE & EQUIPMENT SCHEDULE):
       - If the page contains large master index tables, extract EVERY single row of the table (schedule) systematically.
       - Map row column values directly to the 'notes' key.
       - Use the schedule title or row category as the 'category' key.
       - The codes and its respective scehdule category may be repeated across multiple pages, so you must group them together as mentioned in the de-duplication rules above.

    Example:
    {
        "code": "F-60",
        "notes": "HARDWOOD FLOOR, 2-3\" WIDE, FINISH WOOD, TONGUE & GROOVE, STAINED"    //THESE NOTES MAY BE IN ANOTHER PAGE. FIND THAT AND EXTRACT IT. DO NOT LEAVE IT EMPTY.
        "category": "Room- MainRoom",
        "mentions": [
          {"page_label": "Sheet 6 of 23", "view": "Main Floor Plan Layout"}
        ]
    },
    {
        "code": "F-62",
        "notes": "FLOOR TILE, ~2\", CERAMIC, HEXAGONAL PATTERN"
        "category": "Room- Kitchen Floor",
        "mentions": [
          {"page_label": "Sheet 6 of 23", "view": "Main Floor Plan Layout"}
        ]
    },

    WHAT NOT TO PROVIDE FOR CATEGORY B:
    {
        "code": "F-26",
        "notes": "Window/shutter code referenced in Window Elevation Details and Optional Window Shutters details",
        "category": {
            "c1": "Window", 
            "c2":"Door"
        }
        "mentions": [
            {
                "page_label": "Sheet A4.3 - Doors & Windows",
                "view": "Door Elevation Details"
            },
            {
                "page_label": "Sheet A4.3 - Doors & Windows",
                "view": "Optional Window Shutters"
            }
        ],
    },
    Here you can see notes says referenced to some page. Read that page and bring the information up in this section.

    ### CATEGORY C: Schedules with numberss
    Table Schedules Processing Rules:
    If a table occurs with numbers in theor 1st column, then put the 'name' key as the notation/number/name of the column 1. The rest information could be aaded to the 'notes' section. The 'category' key must be added in accordance with the title of the table and the 'mentions' key must have the page and the view where the table is loacated and where the codes are present in the user plan.

    EXAMPLE 1:
    NO  |Qty |Width |Height |Matrial Finish |Glazing
    ----|----|------|-------|---------------|--------
    01A | 1  | 5'-0 | 6'-8' | Fibreglass    | -

    If the above is a door schdule then name should be Door- 01A
    Then the JSON must look like:
    {
        "name": "Door-01A",
        "notes": "Qty: 1, Width: 5'-0, Height: 6'-8', Material Finish: Fibreglass, Glazing: -",
        "category": "Door Schedule",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - Schedules", "view": "Door Schedule"},
        ]
    }

    If Window schdule comes Eg:

     NO  |Qty |Width |Height | Volume
     ----|----|------|-------|--------
     0C  | 1  | 5'-0 | 6'-8' | -

     Then the JSON must look like:
    {
        "name": "Window-0C",
        "notes": "Qty: 1, Width: 5'-0, Height: 6'-8', Volume: -",
        "category": "Window Schedule",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - Schedules", "view": "Window Schedule"},
        ]
    }

    EXAMPLE 2:
    Note | Description
    -----|---------------------------------------------------
    E14  | 1/2" Gypsum Board, Type X, 5/8" thick, fire-rated
    -----|---------------------------------------------------
    E32  | 3/8" OSB Board, Type X, 5/8" thick, fire-rated

    
    Then the JSON must look like:
    {
        "name": "E14",
        "notes": "1/2\" Gypsum Board, Type X, 5/8\" thick, fire-rated",
        "category": "Interior Wall",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
            {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
        ]
    },
    {
        "name": "E32",
        "notes": "3/8\" OSB Board, Type X, 5/8\" thick, fire-rated",
        "category": "Interior Wall",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
        ]
    }

    *Do not miss any rows and columns in the table* Properly extract data from each row and colum to display. If the table has null values, also include them. Keep '-' sign to indicate null values. Do not leave them empty. 

     #### CATEGORY D Listed submaterials  inside a code
      Use the format below if submaterials are listed inside a code. 
      Also use this format if submaterials are listed inside a wall type, partition code, or detailed assembly callout (e.g., a detail showing 5 layers of a wall: Siding, Wrap, Sheathing, Studs, Drywall).
    - You MUST split these complex layered assemblies into individual material entries in your JSON output (one object for Siding, one for Wrap, one for Sheathing, etc.).
    - Do NOT dump the entire assembly sentence into a single "notes" key. Parse each material layer separately.

      If W1 has listed VINYL SIDING, TYVEK HOUSE WRAP, 3/8' OSB EXTERIOR SHEATHING, 2X6 STUDS @ 16' O.C., R-25 BATT INSULATION then:
    
        {
            "name": "VINYL SIDING",
            "notes": "Material listed in W1",
            "category": "Exterior Wall",
            "mentions": [
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
            ]
        },
        {
            "name": "TYVEK HOUSE WRAP", 
            "notes": "Material listed in W1",
            "category": "Exterior Wall",
            "mentions": [
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
            ]
        },
        {
            "name": "3/8' OSB EXTERIOR SHEATHING", 
            "notes": "Material listed in W1, 3/8' thickness is mentioned in the notes of W1"
            "category": "Exterior Wall",
            "mentions": [
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
            ]
        },
        {
            "name": "2X6 STUDS @ 16' O.C.", 
            "notes": "Material listed in W1, 2X6 size and 16' O.C. spacing is mentioned in the notes of W1"
            "category": "Foundation-Wall",
            "mentions": [             
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"},
            ]
        },
        {
            "name": "R-25 BATT INSULATION", 
            "notes": "Material listed in W1, R-25 insulation value and batt type is mentioned in the notes of W1"
            "category": "Exterior Wall",
            "mentions": [ 
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"},
            ]
        },
    ]

    Do not add any extra json for codes W1 if the materials are already listed in the schedule. Only add the code W1 with the schedule category and notes if there is no material breakdown listed in the schedule. If there  is a material breakdown then do not add W1 code as a separate item. Only add the materials listed under W1 as separate items with their respective categories and notes. TAKE NOTES OF CATEGORY B AND C properly.

        🚨 This same submaterial-breakdown rule ALSO applies when the code's materials are written as a full PROSE SENTENCE instead of a clean comma-separated list -- this is very common in PARTITION / WALL-TYPE SCHEDULES (columns like "PARTITION WALL TYPE" / "TYPE" and "DESCRIPTION"), where a code such as "A1" or "B2" has a description like: "3 5/8\" Metal Stud with one layer of 5/8\" Cementitious Backer Board and Ceramic Tile upto 72\" from FFL (UNO on interior elevations) and painted finish above, on both sides." Do NOT output this as a single object with the whole sentence dumped into "notes" (e.g. do NOT produce {"name": "Partition Wall Type A1", "notes": "<entire sentence>", ...}). Instead, parse the sentence and extract each distinct material mentioned (stud framing, backer board, tile, paint, sheathing, siding, cladding, insulation, etc.) as its OWN object, same as any other CATEGORY D breakdown, using "Extracted from code" to record which wall/partition type it came from.

    #### CATEGORY E: FITTINGS & ACCESSORIES SCHEDULES (e.g. "Fittings and Accessories Schedule")
    A table listing tagged fixtures/fittings/hardware (e.g. columns like S.N., TAG, ACCESSORY, ITEM SPECIFICATION -- covering things like toilet paper dispensers, soap dispensers, grab bars, mirrors, lavatories, urinals, water closets, hand dryers, partitions, shower heads, water heaters) is STILL IN SCOPE and MUST be extracted. Do NOT skip this table under the general "civil engineering materials only" rule -- plumbing fixtures, toilet accessories, and fit-out hardware scheduled with their own TAG are treated the same as any other coded schedule item (see CATEGORY B/C).
    - Use the TAG (e.g. "AC-1", "G-1", "L-1", "M-1", "U-1", "WC-1") as the "name".
    - Combine the accessory description and item specification/model columns into "notes".
    - Set "category" to the room/location this fixture serves if the drawing or schedule indicates it (e.g. "Room- Restroom", "Room- Bathroom")
    - Every row of this table must be extracted -- do not skip any TAG.
    
    Example:
    {
        "name": "AC-1",
        "notes": "Accessory: Toilet Paper Dispenser. Item Specification: Bobrick Model B2888 or equal.",
        "category": "Room- Restroom",
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Fittings and Accessories Schedule"}
        ]
    },
    {
        "name": "WC-1",
        "notes": "Accessory: Water Closet, Std. Item Specification: Sloan Model 20231001 or equal.",
        "category": "Room- Restroom",
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Fittings and Accessories Schedule"}
        ]
    },

    🚨There might occur a case where table is provided with markings indicating the materials are applied in those areas. Investigate them properly and provide notes such that only the marked areas where materials are used is given properly. Carefully read the markings for notes section. For this, follow category F

    
    #### CATEGORY F: MATRIX / CHECKLIST-STYLE SCHEDULES (e.g. "Material and Finishes Schedule" with rooms/areas as column headers, materials as row headers, and "X" marks at the intersections showing which material applies to which room/surface)

    🚨 DO NOT summarize a matrix table into one prose sentence per material (e.g. do NOT write something like "Applied in: Lobby, Shower Room, Family Restroom..."). That kind of free-text summary is exactly what causes rooms to be added or dropped by mistake. Instead, you MUST treat EVERY marked (X) intersection as its own separate output object, using the "row_label"/"column_label" pairs already provided to you in the "OCR-TRANSCRIBED TABLE DATA" reference block for this page -- do not re-derive them yourself from the image, and do not merge multiple pairs together. Eg: If "Ceramic Tile" is checked for 5 different rooms, there must be 5 distinct JSON objects, each with its own "Room- <RoomName>" category.

    For each row_label/column_label pair in that reference data:
    - "name" = the material's row_label exactly as given (e.g. "Ceramic Tile", "Sealed Concrete").
    - "category" = "Room- <room name>" using the room/area portion of the column_label (e.g. column_label "Shower Room - Floor" -> category "Room- Shower Room").
    - "notes" = the surface portion of the column_label (Floor / Ceiling / Wall-North / Wall-South/ Wall-East / Wall-West, etc.), plus the material's brand/manufacturer/type-color spec if the schedule's legend provides one for that material -- do not invent a spec if none exists.
    - "mentions" = the usual page_label/view for this schedule.
    - One object per intersection -- if "Ceramic Tile" is marked for 4 rooms x 4 walls, that is 5 separate objects (each with a different "category"/"notes", walls are considerd as one), NOT one object with a combined list of rooms in "notes".
    - Do not add a room/surface that is not present in the row_label/column_label reference data, and do not omit one that is present. Match the reference data exactly, one-to-one.
    - 🚨 Watch for adjacent-row bleed in the reference data itself: on dense matrix tables, a room/surface can occasionally be misattributed to the wrong neighboring row (e.g. a mark that should belong to "Paint" instead showing up under "Exterior Board", or a row at the bottom of one group like "Sealed Concrete" (a FLOOR item) being mislabeled with the next group's name, "WALL"). If a row_label's assigned category doesn't semantically match what that material actually is (e.g. "Sealed Concrete" tagged as a wall material, or a ceiling material tagged as a floor material), trust the material's real-world nature over a mismatched group label from the reference data, and use the reference image itself to double check which row the mark truly belongs to before finalizing.

    Example -- given reference data:
    [{"row_label": "Ceramic Tile (Anti Slip)", "column_label": "Shower Room - Floor"},
     {"row_label": "Ceramic Tile (Anti Slip)", "column_label": "Family Restroom - Floor"}]

    Correct output (two separate objects, not one combined summary):
    {
        "name": "Ceramic Tile (Anti Slip)",
        "notes": "Floor. Trafficmaster, Baja Gray - Matte Finish 12\" x 12\" or approved equal.",
        "category": "Room- Shower Room",
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Material and Finishes Schedule"}
        ]
    },
    {
        "name": "Ceramic Tile (Anti Slip)",
        "notes": "Floor. Trafficmaster, Baja Gray - Matte Finish 12\" x 12\" or approved equal.",
        "category": "Room- Family Restroom",
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Material and Finishes Schedule"}
        ]
    }

    ### FINAL EXPECTED OUTPUT STRUCTURE

    The final output must be a single flat array containing unique material objects matching this exact JSON format:

    [
      {
        "name": "Asphalt Shingles",
        "notes": "Black colored roof covering material, referenced as exterior material no. 1",
        "category": "Roof",
        "mentions": [
          {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
          {"page_label": "Sheet 7 of 23 - West Elevation", "view": "Exterior Materials Schedule"},
          {"page_label": "Sheet 8 of 23 - Right Elevation", "view": "Exterior Materials Schedule"},
        ]
      },
      {
        "name": "Door- 01A",
        "notes": "Qty: 1, Width: 5'-0, Height: 6'-8', Material Finish: Fibreglass, Glazing: -",
        "category": "Door Schedule",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "Door Schedule"},
            {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Window Schedule"}
        ]
      },
      {
        "name": "E32",
        "notes": "3/8\" OSB Board, Type X, 5/8\" thick, fire-rated",
        "category": "Interior Wall",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
        ]
      },
      {
        "code": "X-74",
        "notes": "HARDWOOD FLOOR, 2-3\" WIDE, FINISH WOOD, TONGUE & GROOVE, STAINED"
        "category": {
            "c1":"Room- MainRoom",
            "c2": "Room- Kitchen"
        }
        "mentions": [
          {"page_label": "Sheet 6 of 23", "view": "Main Floor Plan Layout"},
          {"page_label": "Sheet 8 of 23", "view": "Kitchen Floor Plan Layout"},
        ]
    },
     {
        "name": "Plywood Subfloor",
        "notes": "3/4\" Plywood Subfloor. Material listed in F2 - Typical Floor Assembly.",
        "category": "Room- Typical",
        "mentions": [
            {
                "page_label": "Sheet 17 of 23 - General Notes & Construction Assemblies",
                "view": "Construction Assembly Notes - Floor Types",
                "Extracted from code": "F2"
            },
        ]
    },
      {
        "name": "VINYL SIDING",
        "notes": "Material listed in W1",
        "category": "Exterior Wall",
        "mentions": [
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
        ]
      },
     {
        "name": "TYVEK HOUSE WRAP", 
        "notes": "Material listed in W1",
        "category": "Exterior Wall",
        "mentions": [
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
        ]
     },
     {
        "name": "3/8' OSB EXTERIOR SHEATHING", 
        "notes": "Material listed in W1, 3/8' thickness is mentioned in the notes of W1",
        "category": "Exterior Wall",
        "mentions": [
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
        ]
     },
     {
        "name": "2X6 STUDS @ 16' O.C.", 
        "notes": "Material listed in W1, 2X6 size and 16' O.C. spacing is mentioned in the notes of W1",
        "category": "Exterior Wall",
        "mentions": [             
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"},
        ]
     },
     {
        "name": "R-25 BATT INSULATION", 
        "notes": "Material listed in W1, R-25 insulation value and batt type is mentioned in the notes of W1"
        "category": "Exterior Wall",
        "mentions": [ 
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"},
       ]
     },
     {
        "name": "Ceramic Tile (Anti Slip)",
        "notes": "Floor. Trafficmaster, Baja Gray - Matte Finish 12\" x 12\" or approved equal.",
        "category": "Room- Shower Room",
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Material and Finishes Schedule"}
        ]
     },
     {
        "name": "Ceramic Tile (Anti Slip)",
        "notes": "Floor. Trafficmaster, Baja Gray - Matte Finish 12\" x 12\" or approved equal.",
        "category": "Room- Family Restroom",
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Material and Finishes Schedule"}
        ]
     }
   ]

    Return ONLY VALID JSON. DO NOT MISS ANY MATERIALS.
    """

    schedule_pages = [p for p in valid_pages if p.get("is_schedule")]
    regular_pages = [p for p in valid_pages if not p.get("is_schedule")]

    if schedule_pages:
        print(
            f"Detected {len(schedule_pages)} schedule page(s): "
            f"{[p['page_no'] for p in schedule_pages]}. These will be included"
            f"in every batch so codes can always be resolved against them."
        )

    MAX_IMAGES_PER_REQUEST = 10
    regular_pages_per_batch = max(1, MAX_IMAGES_PER_REQUEST - len(schedule_pages))

    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    table_prompt = build_table_prompt()
    print("\n[Table OCR] Starting OpenCV table-region detection + Claude-vision transcription for all pages...\n")

    anchor_blocks = []
    anchor_file_ids = []

    try:
        for page in schedule_pages:
            with open(page["local_path"], "rb") as f:
                uploaded = client.beta.files.upload(
                    file=(f"page_{page['page_no']}_schedule.png", f, "image/png"),
                    betas=["files-api-2025-04-14"],
                )
            anchor_file_ids.append(uploaded.id)
            anchor_blocks.append({
                "type": "text",
                "text": f"===== INTERNAL PAGE INDEX {page['page_no']} (SCHEDULE REFERENCE PAGE) ====="
            })
            anchor_blocks.append({
                "type": "image",
                "source": {"type": "file", "file_id": uploaded.id}
            })
            
            page_tables = extract_tables_from_page(client, page, table_prompt)
            if page_tables:
                anchor_blocks.append({
                    "type": "text",
                    "text": build_table_reference_text(page["page_no"], page_tables)
                })

        if anchor_blocks:
            anchor_blocks[-1] = dict(anchor_blocks[-1])
            anchor_blocks[-1]["cache_control"] = {"type": "ephemeral"}

        batches = list(chunk(regular_pages, regular_pages_per_batch)) or [[]]

        for batch_no, page_batch in enumerate(batches, start=1):
            print(
                f"Processing batch {batch_no} ({len(page_batch)} pages + "
                f"{len(schedule_pages)} schedule anchors) through Claude Multimodal API..."
            )

            content_blocks = list(anchor_blocks)  # schedule anchors go in every batch
            batch_file_ids = []
            try:
                for page in page_batch:
                    with open(page["local_path"], "rb") as f:
                        uploaded = client.beta.files.upload(
                            file=(f"page_{page['page_no']}.png", f, "image/png"),
                            betas=["files-api-2025-04-14"],
                        )
                    batch_file_ids.append(uploaded.id)

                    content_blocks.append({
                        "type": "text",
                        "text": f"===== INTERNAL PAGE INDEX {page['page_no']} ====="
                    })
                    content_blocks.append({
                        "type": "image",
                        "source": {"type": "file", "file_id": uploaded.id}
                    })

                    
                    page_tables = extract_tables_from_page(client, page, table_prompt)
                    if page_tables:
                        content_blocks.append({
                            "type": "text",
                            "text": build_table_reference_text(page["page_no"], page_tables)
                        })

                combined_prompt = (
                    f"{prompt}\n\n"
                    "AFTER the materials array above, on its own line print exactly:\n"
                    f"{SCALE_DELIMITER}\n"
                    "Then, using the SAME pages you were just given, also perform this SECOND, separate task "
                    "and output ITS result as its own JSON array (or the literal word NONE) immediately after "
                    f"the delimiter line:\n\n{SCALE_PROMPT}"
                )
                content_blocks.append({"type": "text", "text": combined_prompt})

                text_chunks = []
                with client.beta.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=100000,
                    temperature=0,
                    system=(
                        "You are a strict technical drawing extraction engine. You must output valid "
                        "raw JSON data blocks only. Do not speak or include explanations, preamble, or "
                        "trailing markdown wrappers. Start your response directly with '[' and end the "
                        f"materials array with ']', then print the delimiter line '{SCALE_DELIMITER}', "
                        "then the scale JSON array (or NONE)."
                    ),
                    messages=[{"role": "user", "content": content_blocks}],
                    betas=["files-api-2025-04-14"],
                ) as stream:
                    for text in stream.text_stream:
                        text_chunks.append(text)

                full_raw = "".join(text_chunks).strip()

                if not full_raw:
                    print(f"[Batch {batch_no}] WARNING: Empty response received from API. Skipping.")
                    continue

                if SCALE_DELIMITER in full_raw:
                    raw_text, scale_raw = full_raw.split(SCALE_DELIMITER, 1)
                    raw_text = raw_text.strip()
                    scale_raw = scale_raw.strip()
                else:
                    raw_text, scale_raw = full_raw, ""

                if not raw_text.startswith(("[", "{")):
                    print(f"[Batch {batch_no}] WARNING: Unexpected non-JSON response. Preview:\n{raw_text[:500]}\n")
                else:
                    if raw_text.startswith("```"):
                        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                        raw_text = re.sub(r"\s*```$", "", raw_text).strip()

                    try:
                        data = json.loads(raw_text)

                        if isinstance(data, dict) and "views" in data:
                            results.append(data)
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    results.append(item)
                    except json.JSONDecodeError as e:
                        print(f"[Batch {batch_no}] JSON parse error: {e}")
                        salvaged = salvage_truncated_array(raw_text)
                        if salvaged:
                            print(f"[Batch {batch_no}] Salvaged {len(salvaged)} complete material object(s) from the truncated response.")
                            results.extend(salvaged)
                        else:
                            print(f"[Batch {batch_no}] Raw response was:\n{raw_text[:500]}\n")

                if scale_raw and scale_raw.upper() != "NONE":
                    if scale_raw.startswith("```"):
                        scale_raw = re.sub(r"^```(?:json)?\s*", "", scale_raw)
                        scale_raw = re.sub(r"\s*```$", "", scale_raw).strip()

                    try:
                        scale_data = json.loads(scale_raw)
                        if isinstance(scale_data, list):
                            for item in scale_data:
                                if isinstance(item, dict):
                                    scale_results.append(item)
                    except json.JSONDecodeError as e:
                        print(f"[Batch {batch_no}] Scale JSON parse error: {e}")
                        salvaged_scale = salvage_truncated_array(scale_raw)
                        if salvaged_scale:
                            print(f"[Batch {batch_no}] Salvaged {len(salvaged_scale)} scale object(s) from the truncated response.")
                            scale_results.extend(salvaged_scale)
                        else:
                            print(f"[Batch {batch_no}] Scale raw response was:\n{scale_raw[:500]}\n")

            except Exception as e:
                print(f"[Batch {batch_no}] Unexpected error: {e}")
            finally:
                for fid in batch_file_ids:
                    try:
                        client.beta.files.delete(fid, betas=["files-api-2025-04-14"])
                    except Exception:
                        pass
    finally:
        for fid in anchor_file_ids:
            try:
                client.beta.files.delete(fid, betas=["files-api-2025-04-14"])
            except Exception:
                pass

    results = merge_duplicate_materials(results)

    pdf_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    raw_json_path = os.path.join(state["output_base"], "data", f"{pdf_name}_materials.json")
    os.makedirs(os.path.dirname(raw_json_path), exist_ok=True)
    
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Raw extracted JSON successfully saved to: {raw_json_path}")

    os.makedirs(SCALE_PATH, exist_ok=True)
    scale_json_path = os.path.join(SCALE_PATH, f"{pdf_name}_scale.json")
    with open(scale_json_path, "w", encoding="utf-8") as f:
        json.dump(scale_results, f, indent=4, ensure_ascii=False)
    print(f"Scale report saved to: {scale_json_path}")

    return {"valid_pages": valid_pages, "extracted_materials": results, "scale_report_path": scale_json_path}