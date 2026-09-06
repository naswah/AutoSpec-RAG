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
 
    stamp_mask = cv2.dilate(stamp_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(cv2_img, stamp_mask, 3, cv2.INPAINT_TELEA)


def count_horizontal_grid_lines(cv2_img):

    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    w = thresh.shape[1]
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.5), 25), 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(horiz_lines, connectivity=8)
    row_lines = sum(1 for i in range(1, n_labels) if stats[i][2] > w * 0.4)
    return max(row_lines - 1, 0)


def upscale_if_small(cv2_img, min_width=1400, min_height=700, min_row_px=42):
   
    h, w = cv2_img.shape[:2]
    scale = max(min_width / w, min_height / h, 1.0)

    estimated_rows = count_horizontal_grid_lines(cv2_img)
    if estimated_rows > 0:
        required_height = estimated_rows * min_row_px
        if required_height > h * scale:
            scale = max(scale, required_height / h)

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

        🚨 EXCEPTION -- WALL/PARTITION ASSEMBLY LEGENDS ARE TABLES, NOT DIAGRAMS: a "WALL ASSEMBLY", "PARTITION TYPE", or similar legend that has a bordered grid with one row per wall type -- even though each row's cell contains a small cross-section drawing rather than plain text -- IS a genuine data table for this purpose, because each row still pairs a tag/code (often a bare number or letter+number inside a diamond/hexagon/circle symbol, e.g. "0", "0A", "1", "1A") with a list of material layers (readable as text labels pointing at the cross-section, e.g. "8\" CONCRETE WALL", "1/2\" GYPSUM BOARD"). Treat this as a row-based schedule: transcribe each row as one record with keys for at least the tag/code and the material layer list (e.g. "tag": "0A", "layers": "8\" CONCRETE WALL; 1.5\" R7.5 RIGID INSULATION; 1-5/8\" LIGHT GAGE GALVANIZED FURRING CHANNEL @16\" O.C.; 1/2\" STUCCO OR G.W.B."). Do not return [] for this kind of legend just because it is illustrated rather than plain text.

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
          🚨 COLUMN-INTEGRITY WARNING (CRITICAL -- prevents silent data loss/shifting): process each row strictly column-by-column against the header's vertical grid lines, not by reading text left-to-right and assigning it to whichever key "seems to fit." A blank cell in the MIDDLE of a row (e.g. an empty "Manufacturer" or "Style" cell) is common and must be preserved as "-" in its own key -- it must NOT cause every value to its right to shift one column to the left. Before finalizing each row, count the number of populated cells against the number of header columns and re-verify each value sits under its own header, especially for rows where any cell looks empty. If a row has a genuinely blank Location or Notes cell, output "-" for that key rather than omitting the key or letting a neighboring column's value slide into it.
          🚨 ROW-BLEED WARNING (CRITICAL -- applies to EVERY row-based table, not just matrix tables): on tables with many thin, closely-packed rows, it is very easy to accidentally attach a cell's text to the row ABOVE or BELOW where it actually belongs -- e.g. copying a "Location" value from the next row up because the current row's own value was short (like a single word "TYPICAL") and visually less prominent than its neighbor's longer text. This is especially likely when two adjacent rows share the same first few columns (same TYPE, same MANUFACTURER) and differ mainly in one short cell. To avoid this:
            (a) Process rows top-to-bottom, ONE row at a time. For each row, first anchor on that row's own mark/code/first-column value, then read every other cell for THAT row strictly within its own horizontal band -- do not reuse or "carry forward" a value you read for the previous row.
            (b) Before finalizing, re-scan each row's cells against the row directly above and below it. If two adjacent rows would end up with byte-identical values in a cell that is normally distinctive (e.g. Location, Notes) -- that is a strong signal one of them bled from the other. Re-examine the actual pixels of that specific cell for each row individually before accepting the values as correct, rather than assuming a value that "looked right" the first time.
            (c) Do not let a short, terse cell value (a single word like "TYPICAL", or "-") get silently overwritten in your own working memory by a nearby row's longer, more visually prominent text -- terse values are correct data, not incomplete reads.
        - If it is a MATRIX/checklist-style table (e.g. rooms or categories as column headers, item names as row headers, and marks like "X" at the intersections indicating which items apply to which column): represent each marked intersection as one JSON object with keys "row_label" and "column_label" (using the actual row/column header text, including any grouped/parent header if the header spans multiple levels, e.g. "Men's Restroom - Wall North").
          🚨 ROW-BLEED WARNING (this is the single most common mistake on dense matrix tables): it is very easy to accidentally shift a mark UP or DOWN by one row when the rows are thin and closely packed -- e.g. reading a mark that actually belongs to "Paint" as if it belonged to "Exterior Board" on the row below, or reading the last row of one vertical group (e.g. "Sealed Concrete" at the bottom of a "FLOOR" group) as if it were the first row of the NEXT group ("WALL"). To avoid this:
            (a) Process ONE row at a time. First read that row's own printed row_label text, then -- and only then -- scan strictly within that row's own horizontal band for marks. Do not carry a mark over from the row directly above or below.
            (b) When row labels are grouped under a shared vertical parent label spanning multiple rows (e.g. "FLOOR" bracketing both "Ceramic Tile (Anti Slip)" and "Sealed Concrete", with "WALL" bracketing the group below it), determine the parent-group boundary from the actual bracket/merged-cell extent in the image -- the LAST row inside a group still belongs to THAT group, never to the next one.
            (c) After finishing, re-verify each row_label independently: for every row printed on the left side, look back at the marks you assigned to it and confirm they are horizontally aligned with that exact row and no other.

        Instructions (apply to every table found):
        - Transcribe EVERY data row/mark in each table. Do not skip rows, do not skip the last row of any table.
        - Do NOT include header row(s) themselves as data entries -- they define the keys, not a row of data.
        - Use your knowledge of plausible values (e.g. short codes like "AD-3" or "G-5", model numbers being alphanumeric) to resolve any visually ambiguous characters (0 vs O, 1 vs I vs l, etc).
        - 🚨 COMMON FINISH-SCHEDULE ABBREVIATION: "TYP." or "TYP" (meaning "Typical") is extremely common in a Location/Notes column of a finish schedule, often written right after a room name (e.g. "KITCHEN, TYP.", "BATHROOMS, TYP."). This is easy to misread as an unrelated abbreviation (e.g. "TWR.") because of its short, stylized print. Whenever a Location cell contains a short trailing abbreviation after a room name, double-check whether it actually reads "TYP." before transcribing anything else -- this word is functionally significant downstream (it drives room-typical categorization), so it must be read correctly rather than guessed from a similar-looking shape.
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


def call_claude_table_vision(client, image_b64, media_type, prompt, model="claude-sonnet-5", max_tokens=8000):

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


def flag_possible_row_bleed(records, page_no, region_idx, table_label,
                             distinctive_keys=("location", "notes")):
   
    if not records or not isinstance(records, list):
        return

    row_key_candidates = ("mark", "code", "no", "tag", "name", "item", "label")

    for key in distinctive_keys:
        seen = {}
        for rec in records:
            if not isinstance(rec, dict) or key not in rec:
                continue
            value = str(rec.get(key, "")).strip().lower()
            if not value or value in ("-", "n/a", "na", "none"):
                continue

            row_id = None
            for rk in row_key_candidates:
                if rec.get(rk):
                    row_id = str(rec[rk]).strip()
                    break

            if value in seen and seen[value] != row_id:
                print(
                    f"[Table OCR][QA WARNING] Page {page_no}, region {region_idx}, table {table_label}: "
                    f"rows '{seen[value]}' and '{row_id}' have an identical '{key}' value "
                    f"({value!r}). If these rows are not supposed to share this value, this is "
                    f"the signature of a row-bleed misread -- verify against the source image."
                )
            else:
                seen[value] = row_id


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

            flag_possible_row_bleed(records, page["page_no"], i, label)

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

        # Wall-type / partition-type codes (e.g. P1, P2, P3) must never merge with each other even when name/category/notes are textually identical, because each tag can carry its own exterior/interior classification. Pull the set of codes referenced in this item's own mentions and fold them into the key.
        mention_codes = tuple(sorted({
            str(m.get("Extracted from code", "")).strip().lower()
            for m in item.get("mentions", [])
            if isinstance(m, dict) and m.get("Extracted from code")
        }))

        key = (
            str(item.get("name", item.get("code", ""))).strip().lower(),
            str(item.get("category", "")).strip().lower(),
            str(item.get("notes", "")).strip().lower(),
            mention_codes,
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


_WALL_CODE_RE = re.compile(r"^wall-(?!interior$|exterior$|foundation$)\S+$", re.IGNORECASE)


def _category_values(category):
    if isinstance(category, str):
        yield category
    elif isinstance(category, dict):
        for v in category.values():
            if isinstance(v, str):
                yield v


def _rewrite_category_values(category, rewrite_fn):
    if isinstance(category, str):
        return rewrite_fn(category)
    if isinstance(category, dict):
        return {k: (rewrite_fn(v) if isinstance(v, str) else v) for k, v in category.items()}
    return category


def enforce_wall_coding_consistency(items):
    if not items:
        return items

    has_wall_type_code = any(
        isinstance(item, dict) and any(
            _WALL_CODE_RE.match(v.strip()) for v in _category_values(item.get("category"))
        )
        for item in items
    )

    if not has_wall_type_code:
        return items

    downgraded = 0

    def rewrite(value):
        nonlocal downgraded
        if value.strip().lower() in ("wall-interior", "wall-exterior"):
            downgraded += 1
            return "Wall"
        return value

    for item in items:
        if not isinstance(item, dict):
            continue
        item["category"] = _rewrite_category_values(item.get("category"), rewrite)

    if downgraded:
        print(
            f"[Wall Rule QA] Document uses a wall-type coding system (Wall-<code> found). "
            f"Downgraded {downgraded} generic 'Wall-Interior'/'Wall-Exterior' value(s) to plain 'Wall' "
            f"for consistency."
        )

    return items


def build_table_reference_text(page_no, page_tables):

    has_matrix_table = any(
        isinstance(r, dict) and ("row_label" in r or "column_label" in r)
        for t in page_tables
        for r in (t.get("records") or [])
    )

    has_schedule_table = any(
        isinstance(t.get("table_title"), str) and "SCHEDULE" in t["table_title"].upper()
        for t in page_tables
    )

    header = f"===== OCR-TRANSCRIBED TABLE DATA FOR PAGE {page_no} (reference only -- use this to resolve exact row values) ====="
    if has_matrix_table:
        header += (
            "\nNOTE: one or more tables below use the row_label/column_label matrix format "
            "(see CATEGORY F). Convert EACH pair into its own separate output object -- do NOT "
            "merge multiple pairs into one combined summary sentence, and do NOT add or omit any "
            "room/surface that isn't explicitly present in the pairs below."
        )
    if has_schedule_table:
        header += (
            "\nNOTE: one or more tables below has a title containing 'SCHEDULE' -- treat EVERY record in that table as a Category B schedule row. Each record's mark/code column "
            "(e.g. 'mark', 'no', 'tag') becomes the ONLY 'name' for that row's output entry. Every other field in that same record (item, material, size, notes, manufacturer, etc.) must be folded into that ONE entry's 'notes' string -- never emit a second entry using any other field's value as its own 'name', even if it reads like a standalone material."
            "\n🚨 Do NOT drop the record's 'item' field just because a 'material' field is also present in the same record -- they are different questions (what it's called vs. what it's made of, e.g. item='BRICK' vs material='SMOOTH BRICK') and BOTH must appear in 'notes', each labeled with its own field name."
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
    - If a note mentions an exterior wall made of "8' Concrete Foundation Wall, 4000 PSI", extract "Concrete Foundation Wall" for name and "8' Concrete Foundation Wall, 4000 PSI" for notes, "Wall-Foundation" for category. 
    - Instead of extracting "Front Porch", look for specific material callouts inside that porch zone (e.g. "CMU Block foundation", "Cast-in-place Concrete Slab").
    - Instead of extracting "Interior Partition Walls", look for the actual materials: "5/8" Type X Gypsum Board"(For name key, write Gypsum Board and for notes, add the sizes), "2x4 Wood Studs"(For name key, write Wood Studs and for notes, add the size), or "Light-Gauge Metal Stud Framing".
    - Do not take dimensions as codes.
    - If the material name is 'black asphalt shingles' then write the name of material as 'Asphalt Shingles'. Mention the color and other specifications in the 'notes' section.
    - Structural & Framing Materials: E.g., "2x12 Joists", "4x12 Glulam Beam", "Lookout Rafter", "Chamfered 5x5 Post", wood studs, headers, and plates.
    - Exterior Trim & Roof Components: E.g., "Fascia Board", "Frieze Board", "Shed Roof assemblies", gutters, drip edges.
    - Window & Door Details: E.g., "Fiber Cement Subsills", "Exterior Surrounds", "Door Frames", casing, and moldings.
    - Layered Finishes: E.g., "Gypsum Wallboard", "T&G Decking", vapor barriers, and "Air Space" ventilation gaps.
    - Tagged equipment/fixtures (Category E).
    - If a material is mentioned multiple times, write it only once. Strictly avoid duplicates. A material is considered identical if it has the same name, notes, and category. For freeform (non-coded) materials, the notes across separate mentions could be worded slightly differently on the page each time -- treat these as the same material and merge their "mentions" rather than creating two entries; do not invent new wording of your own when merging. For coded/schedule materials, follow the VERBATIM NOTES rule above instead -- these should never need "recognizing as a paraphrase" because they must always be transcribed identically. If the same material is used in different locations (e.g., "Gypsum Board" in both "Room-Kitchen" and "Room-Bathroom"), list it separately for each location with the same name and notes but different category.

    ❗WHAT NOT TO EXTRACT:
    - In drawing labelling, if you see labelled materials that are not actually used in the construction, civil engineering, do not extract them.
    - Do not extract 'Air Space' as a material. It is a gap between two materials.
    Example:
     {
        "name": "3/4\" Air Space",
        "notes": "3/4\" air space between siding and sheathing, at window and door details",
        "category": "Wall-Exterior",
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

    🚨 VERBATIM NOTES FOR CODED/SCHEDULE ROWS (CRITICAL -- PREVENTS DUPLICATE ENTRIES):
    - When a "notes" value is being copied from a schedule row, legend entry, or detail callout for a CODE (Category B/C/E/F), you MUST transcribe that row's text VERBATIM -- same words, same order, same punctuation and capitalization as printed. Do NOT paraphrase, reword, summarize, reorder clauses, or "clean up" the wording, even if it reads awkwardly. Two different passes over the SAME schedule row must always produce the EXACT SAME "notes" string, character for character (aside from trivial whitespace), so that duplicate detection downstream can match them.
    - This verbatim rule applies ONLY to schedule/legend/code-defined "notes" text. It does NOT apply to the "name" field (which should still follow the normalization rules above, e.g. "black asphalt shingles" -> "Asphalt Shingles"), and it does NOT apply to freeform materials with no code (Category A), where notes should still be written in your own words as instructed elsewhere.
    - If the same code's schedule row is visible again in a later page of this same request (e.g. because it was included as a reference anchor), re-extract its "notes" the exact same way you did the first time -- do not vary the phrasing between occurrences.
    
    REMEMBER TO: - Extract where the materials are located for 'category' key into fixed categories: "Wall-Interior", "Wall-Exterior", "Wall-WallName", "Wall", "Door", "Window", "Roof", "Room-RoomName", "Room-Typical", "Wall-Foundation", "Room-Foundation" or "Others"  Do not add any other categories by yourself.

    🚨 IMPORTANT CLARIFICATION ON "Wall" vs "Wall-WallName" vs "Wall-Interior"/"Wall-Exterior":
    - Use "Wall" (plain, no suffix) ONLY when the document follows a wall-type coding/naming  system (e.g. "P1", "W1", "Wall-01", or a bare tag like "0"/"1A" in a diamond/hexagon symbol)  but THIS specific material can't be tied to one of those codes but is a material appled in wall.
    - If the document does NOT use any wall-type coding system anywhere, do NOT use "Wall" —  default to "Wall-Interior" or "Wall-Exterior" based on location/context instead.
    - If a specific wall-type code/name IS identifiable, use "Wall-<WallTypeName>" — never   "Wall" or "Wall-Interior"/"Wall-Exterior" in that case.

    If that information is explicitly provided in the notes or schedules. If the materials applied in room, read the name of the room and provide that as the location context (e.g., "Room-Kitchen Floor", "Room-Storage Area", "Room-Living Room", "Room-Front Porch") in the category key.
    
    🚨 EXCEPTION TO THE ABOVE (apply this FIRST, before defaulting to Room-RoomName): this "read the room name -> Room-X" default does NOT apply to materials that are inherently WALL, FLOOR, WALL-BASE, or CEILING functional elements (e.g. Paint, Rubber Base Cove, Ceramic Tile, Gypsum Board, wall panels) even when their notes say "Applied in: <room names>" or come from a room-by-room Finishes Schedule matrix. Materials on the WALL row of a Finishes Schedule/Room Tag Legend matrix -- regardless of how many rooms they're applied in or how their notes are worded -- must go through the WALL RULE. 

    - For category, if the drawing has no clear information, 'notes' could also be read for adding category. For example, if notes section has the descrption: 'Engineered Trusses @ 24 O.C. per layout. Part of Porch Roof Assembly (R2).' Then the category could be 'Roof' because of the mention of porch roof assembly in the notes.
    - For category, if 'notes' section has anything written as 'Typical Room Assembly' then it the category key must has the value 'Room-Typical' because it is generic and is applied to all  rooms.
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
            "c2": "Door",
    }

    - Try to include the name of material in the 'name' key whenver possible. 
    Excample 1:
    "name": "Asphalt Shingles",
    "notes": "Black asphalt shingles, 25-year warranty", #here, as you can see the name is also included in the notes section. This is important for downstream processing and for clarity.
    If it is already mentioned in the notes section, then you can skip it.

    Example 2:
    "name": "Beadboard Trim",
    "notes": "Beadboard Trim, VW Mariposa 2229",
    Instread of just providing the model number in notes, also include the name of the material in the notes section for clarity and downstream processing.

    - Use the name of table when necessary for report_notes for example,
        (Beam Schedule)
        "name": "B3",
        "notes": "Type Mark: B3, Size: 3-2x14, Material: SPRUCE PINE FIR",
        "report_notes": "Beam B3, 2x14, Spruce Pine Fir", # State marck name as well.

    -  If a material has other material details then do not include them in the notes section. For example, if a material has a note like "Concrete slab on grade, 4" thick, on 4" closed cell XPS, on 6" minimum compacted gravel", then the name of the material is "Concrete Slab" and the notes section should not include the name of the material along with size and thickness if present. Example:
        "name": "Concrete Slab",
        "notes": "Concrete slab on grade, 4\" thick, on 4\" closed cell XPS, on 6\" minimum compacted gravel", ❌ as it includes the name of other materials in the notes section, drop them.
        "notes": "Concrete slab on grade, 4\" thick", ✅
    
    In standards and materials schedule, Looks at the following rules if 'LOCATION' is provided:
    🚨🚨🚨 CRITICAL -- STANDARD MATERIALS & FINISHES SCHEDULE WITH NO 'LOCATION' COLUMN AT ALL (this is a very common layout -- do NOT mishandle it):
    Some "STANDARD MATERIALS & FINISHES SCHEDULE" tables list only TAG / TYPE / BRAND-MANUFACTURER / STYLE-COLOR-SIZE-FINISH, with NO Location/room column whatsoever. When you encounter this:
    - 🚫 DO NOT default every row in that schedule to a single generic category like "Wall" just because Location is missing. This is a critical failure mode: FL-1 (a granite FLOOR tile), CT-1 (a ceiling tile), and WB-1 (a wall base) are physically different elements and must never all collapse into the same "Wall" category.
    - Instead, the room/location information for these tags lives on the FLOOR PLAN itself, inside the small "ROOM TAG" boxes printed next to each room (see the ROOM TAG ROW-TO-CATEGORY MAPPING section below) -- e.g. a box next to "PRIEST ROOM" listing FL-3 / WB-1 / PT-1 / CT-1, a box next to "KITCHEN" listing FL-4 / WB-1 / FRP / PT-1, a box next to "LOBBY" listing FL-1, FL-2 / WB-2 / PT-1 / CT-1, C-4, etc. You MUST scan every page of the floor plan for these Room Tag boxes and use them as the Location source for every tag code that appears in the schedule.
    - For each tag code (e.g. FL-1), find EVERY Room Tag box on the plan that lists that code, and read the room name printed on/above that box. Build the category from those room names using the FLOOR/CEILING -> "Room-<RoomName>" and WALL BASE/WALL -> WALL RULE mapping described in the ROOM TAG ROW-TO-CATEGORY MAPPING section. If the same tag (e.g. FL-1) appears in multiple Room Tag boxes for different rooms, combine them into a multi-value category object (c1, c2, c3...), one per distinct room.
    - Only if a tag code from the schedule genuinely does NOT appear in ANY Room Tag box anywhere on the floor plan (truly unreferenced) should you fall back to identifying it purely by its own type/name (e.g. still route an unreferenced FL- floor tile through Room-Typical/STEP 2 using the type name to know it's floor-functional, and an unreferenced WB-/PT- wall item through the WALL RULE) -- never let "no Location column" become an excuse to output the same category for floor, wall, and ceiling tags alike.
    - Roof-, door-, window-, and countertop/casework-type rows in this same schedule are unaffected by this rule and keep using their own type-based category (Roof, Door, Window, Others, etc.) as described elsewhere in this prompt.

    WORKED EXAMPLE (Location-less schedule + Room Tag boxes on the plan): suppose the schedule only has TAG/TYPE/BRAND/STYLE rows for FL-1, WB-1, PT-1, CT-1 (no Location column), and the floor plan shows Room Tag boxes reading: "LOBBY" -> FL-1, WB-2, PT-1, CT-1, C-4; "PRAYER HALL" -> FL-1, WB-1, PT-1, C-4; "PRIEST ROOM" (appears twice) -> FL-3, WB-1, PT-1, CT-1. Then:
    - FL-1 appears in the Lobby box AND the Prayer Hall box (both FLOOR rows) -> category = {"c1": "Room-Lobby", "c2": "Room-Prayer Hall"}.
    - CT-1 appears in the Lobby box AND both Priest Room boxes (all CEILING rows) -> category = {"c1": "Room-Lobby", "c2": "Room-Priest Room"} (Priest Room only listed once even though its box repeats twice, since it's the same room).
    - WB-1 appears in the Prayer Hall and Priest Room boxes (WALL BASE rows, wall-functional) -> goes through the WALL RULE (e.g. "Wall-Interior" if no wall-type coding system exists anywhere in the document), NOT "Room-Prayer Hall"/"Room-Priest Room".
    - PT-1 appears in all three rooms' boxes (WALL rows) -> same WALL RULE treatment as WB-1, combined across mentions, not Room-based.
    This is the pattern to follow whenever the schedule itself has no Location column: read every Room Tag box on the plan, not just one, and build the category from the union of rooms where that exact tag shows up on the matching row (Floor/Ceiling vs Wall Base/Wall).

    === CATEGORY DECISION ORDER (evaluate top to bottom, stop at first match) ===

STEP 1 — WALL-FUNCTIONAL vs FLOOR/CEILING-FUNCTIONAL CHECK (highest precedence):
    Ask first: is this material inherently a WALL-surface functional element (Paint on a wall, Rubber Base Cove, wall-mounted Ceramic Tile, Gypsum Board, wall panels, FRP, wainscot, wall cladding)? These go through the WALL RULE below (Wall-<WallTypeName> or Wall-Interior/ Wall-Exterior) and STOP -- do not consider Room-Typical/Room-RoomName for these.

    Is it instead inherently a FLOOR or CEILING functional element (Concrete Floor, Floor Tile, Carpet, Ceiling Tile, Acoustic Ceiling Panel, etc.)? These are NEVER routed through the WALL RULE -- the WALL RULE has no floor/ceiling output and routing them there causes them to fall through ungrouped. Instead, send FLOOR and CEILING materials directly to STEP 2(Room-Typical rule) and let that rule's explosion/room-name logic decide the category.

    PAINT SPECIAL CASE: paint applied to a wall -> WALL RULE. Paint explicitly described as ceiling paint -> treat as CEILING-functional -> STEP 2, not the WALL RULE.

    🚨 FLOOR vs CEILING ARE SEPARATE ELEMENTS, EVEN THOUGH THEY SHARE THE SAME CATEGORY FORMAT: Both Floor materials and Ceiling materials are represented using the identical string format "Room-<RoomName>" -- that string alone never tells you whether a given entry is a floor or a ceiling material. Because of this, NEVER cross-infer between the two:
      - If a material is explicitly identified (by its name, type, or notes -- e.g. "Concrete Floor", "Floor Tile", "Carpet") as a FLOOR material with Location "Typical" or "Room-<RoomName>", categorize/explode it for the FLOOR only. Do not also assume it applies to that room's ceiling.
      - If a material is explicitly identified (e.g. "Acoustic Ceiling Panel", "Ceiling Tile", ceiling paint) as a CEILING material, categorize/explode it for the CEILING only. Do not also assume it applies to that room's floor.
      - A floor material and a ceiling material in the same room will end up with the SAME "Room-<RoomName>" category value -- that is expected and correct. Do not merge them into one entry, and do not use one's presence to infer or duplicate the other. Each is extracted, categorized, and exploded independently, based solely on its own identity as a floor or ceiling element.

STEP 2 — ROOM-TYPICAL RULE (for FLOOR/CEILING materials and any non-wall-functional material):
    Category should be "Room-Typical" if and only if the material is applied in all rooms (regardless of location), determined as follows:

    2a. RECOGNIZING A "ROOM-SPECIFIC TYPICAL" ENTRY (CRITICAL -- do not pattern-match on exact spelling/punctuation):
        A Location value counts as a "room-specific Typical" whenever it names ONE room/area AND also carries a typical/generic qualifier, in ANY of these surface forms (this list is illustrative, not exhaustive -- treat all of these as equivalent):
          - "Kitchen-Typical" (hyphenated)
          - "KITCHEN, TYP." or "KITCHEN, TYP" (comma + abbreviated "Typ.")
          - "Kitchen Typical" (space-separated)
          - "TYP. KITCHEN" or "TYPICAL - KITCHEN" (qualifier first)
        The abbreviations "TYP.", "TYP", and "TYPICAL." all mean "Typical" -- normalize them to the word "Typical" when reasoning about this rule. Do not treat "Typ." as a separate, unrelated location token, and do not confuse it with unrelated similar-looking abbreviations on the same sheet (e.g. if you are unsure whether a printed abbreviation reads "TYP." vs something else, re-examine the image at that cell specifically before deciding -- this qualifier is load-bearing for the whole Room-Typical rule and must be transcribed correctly, not guessed). A Location value like "Restroom, Dining" (multiple room names, NO typical qualifier) is a plain multi-room subset, not a "room-specific Typical" -- handle it via the last sentence of 2c instead.

    2b. ROOM LIST AVAILABILITY CHECK (mandatory before exploding):
        Only attempt explosion into per-room categories if you can identify a concrete, enumerable list of ALL individual room names in the building/floor plan from:
          (b) explicit room-name labels printed on the floor plan (actual names like "KITCHEN", "DINING", "BAR", "OFFICE", "JANITOR", etc. -- not bare room numbers). Read all pages of the pdf to find the names of every room shown on the floor plan. THIS is the room list you explode a plain-Typical material into.
        Source (a) below (room-specific Typical rows within the schedule itself, e.g. "KITCHEN, TYP.", "BAR, TYP.") is NEVER sufficient on its own to build the room list for explosion -- 🚨 those rows tell you ONLY which rooms to EXCLUDE (see STEP A below), never which rooms to explode INTO. Using (a) as the room list is self-defeating: the rooms an exploded material should go into and the rooms it should exclude would end up being drawn from the identical set, which either produces zero rooms to explode into or, worse, lets an excluded room slip back in through the "room list" side of the calculation. Always source the actual room list from (b) -- the floor plan itself.
        (a) other rows in this SAME schedule with room-specific Typical Locations as defined in 2a (e.g. "KITCHEN, TYP.", "Kitchen-Typical"), or explicit multi-room subsets (e.g. "Restroom, Dining") -- use this ONLY to build the STEP A excluded-room set below, never as the room list itself.
        If the floor plan's room names (source b) cannot be reliably read, do NOT explode anything. Output the literal single category "Room-Typical" and stop. Do not guess a room list, do not fall back to using source (a) alone as if it were a complete room list, and do not substitute "Others" or a wall category as a workaround.

    2c. EXPLOSION LOGIC (once 2b is satisfied) -- follow this as a two-step ALGORITHM, not per-material free reasoning:

        🚨 FLOOR/CEILING EXCLUDED-ROOM SETS ARE TRACKED SEPARATELY (do this before STEP A): because floor materials and ceiling materials share the identical "Room-<RoomName>" category format, the "excluded rooms" set built in STEP A must be split into TWO independent sets -- one for FLOOR-functional materials and one for CEILING-functional materials -- never a single shared set. A room-specific Typical row for a CEILING material (e.g. "APC-2 | Acoustic Panel; Ceiling | Kitchen-Typical") excludes that room ONLY from the ceiling-exclusion set, and must NOT exclude that room from any FLOOR material's explosion, even though both would render as the same "Room-Kitchen" string. Likewise a floor-specific Typical row only feeds the floor-exclusion set. Materials that are neither floor- nor ceiling-functional (e.g. a plain multi-room fixture) don't feed either exclusion set. Classify each room-specific Typical row's underlying material as floor- or ceiling-functional FIRST (using the same identity check from the WALL-vs-FLOOR/CEILING rule above), then route it into the matching exclusion set.

        STEP A (do this ONCE per schedule, before categorizing any individual material): build the "excluded rooms" set(s) by scanning EVERY row in the schedule and collecting the room name from EVERY row that is a room-specific Typical per 2a (e.g. "KITCHEN, TYP." -> Kitchen; "BAR, TYP." -> Bar; "BATHROOMS, TYP." -> Bathrooms) -- sorting each into the FLOOR set or the CEILING set per the rule immediately above. These sets are a property of the WHOLE schedule, not of any single material -- compute them once and reuse for every plain-Typical material below. This is source (a) from 2b -- it is used ONLY to build these excluded sets, never as the room list to explode into. 🚨 Do not stop after finding the first room-specific Typical row and assume that's the only exclusion -- schedules commonly have several (one per room that has its own override, and possibly split across floor vs ceiling), and EVERY one of them must go into its matching set, not just the first/most memorable one.

        STEP B (apply to each material individually): if a material's own Location is a plain "Typical" (no room name attached), explode it into "Room-<Name>" for every room in the 2b(b) floor-plan room list EXCEPT every room in the STEP A excluded set that MATCHES this material's own floor-or-ceiling identity (a floor material only omits rooms from the floor-exclusion set; a ceiling material only omits rooms from the ceiling-exclusion set -- never the other set). A room stays excluded from a plain-Typical material's explosion regardless of whether the room's own override happens to be a *different* material name than the one being exploded, as long as it is the SAME floor-or-ceiling identity -- exclusion is keyed on the ROOM having its own Typical override for that same surface (floor or ceiling) in this schedule, not on matching material type or name. Rooms named only via a plain multi-room subset with no Typical qualifier (e.g. "Restroom, Dining") are NOT added to either excluded set. If a material's own location explicitly names specific rooms (not "Typical"), just use those rooms directly -- no exclusion logic needed for that material.

        SELF-CHECK before finalizing: list out the full FLOOR excluded-room set and the full CEILING excluded-room set explicitly and separately, then confirm that (a) NONE of the floor-set rooms appear in the exploded category list of any plain-Typical FLOOR material, (b) NONE of the ceiling-set rooms appear in the exploded category list of any plain-Typical CEILING material, and (c) a room excluded from one set (e.g. Kitchen excluded from CEILING because of a ceiling override) still correctly APPEARS in the opposite surface's explosion (e.g. Kitchen still appears in the FLOOR material's exploded list) unless Kitchen also has its own separate floor override. If a schedule has room-specific Typical rows for both, say, Kitchen (ceiling) and Bar (floor), then a plain-Typical ceiling material's explosion must omit Kitchen but still include Bar, and a plain-Typical floor material's explosion must omit Bar but still include Kitchen.

    2c. Separately: if the 'notes' field contains the phrase "Typical Room Assembly" (generic,  applies to all rooms), category = "Room-Typical" as a literal single value, following the same 2a availability check before attempting any explosion.
    Example:
      If there are 5 rooms in a plan: Kitchen, Dining, Restroom, UtilityRoom and Bedroom
    
            Label   |Type                       |Colour     | Location
            APC-1   |Acoustic Panel; Ceiling    |Black      | Typical
            APC-2   |Acoustic Panel; Ceiling    |White      | Kitchen-Typical
            TP-1    |Shutter                    |Natural    | Restroom, Dining
        
            Then JSON should be:
            {
            "name": "APC-1",
                    "notes": "Type: Acoustic Panel; Ceiling,Colour: Black, Location: Typical",
                    "category": {
                        "c1": "Room-Restroom",
                        "c2": "Room-Dining",
                        "c3": "Room-UtilityRoom",
                        "c4": "Room-Bedroom",    // everything except Kitchen typical. 
                    }   
                    "mentions": [
                        {
                            "page_label": "C - 202 - Finishes Plan and Schedule",
                            "view": "Standard Materials & Finishes Schedule"
                        }
                    ]
            },
            {
                "name": "APC-2",
                        "notes": "Type: Acoustic Panel; Ceiling,Colour: White, Location: Kitchen Typical",
                        "category": "Room-Kitchen",     #Since Kitche Typical is given
                        "mentions": [
                            {
                                "page_label": "C - 202 - Finishes Plan and Schedule",
                                "view": "Standard Materials & Finishes Schedule"
                            }
                        ]
            },
            {
                "name": "TP-1",
                        "notes": "Type: Shutter,Colour: Natural, Location: Restroom, Dining, Storage",
                        "category": {
                            "c1": "Room-Restroom",
                            "c2": "Room-Dining"
                        }   
                        "mentions": [
                            {
                                "page_label": "C - 202 - Finishes Plan and Schedule",
                                "view": "Standard Materials & Finishes Schedule"
                            }
                        ]
                },

            MULTI-EXCLUSION EXAMPLE (this is the case that gets missed most often -- a schedule with MORE THAN ONE room-specific Typical row, all of which must be excluded together):
            If the same 5-room plan also has:
                C-1   |Concrete Floor  |Grey  | Typical
                C-2   |Sealed Concrete |Grey  | Bar-Typical
            C-1 and C-2 are FLOOR-functional materials, so they feed and consult only the FLOOR excluded-room set -- they are entirely independent of the CEILING excluded-room set built from APC-2 (Kitchen-Typical) above. The FLOOR excluded-room set for this schedule is {Bar} (from C-2 only) -- Kitchen is NOT in the floor set, because APC-2's "Kitchen-Typical" override was for a ceiling material, not a floor material. So C-1's explosion (a plain-Typical FLOOR material) omits ONLY Bar, and correctly still includes Kitchen (Kitchen has no floor-specific override, only a ceiling one):
            {
                "name": "C-1",
                "notes": "Type: Concrete Floor, Colour: Grey, Location: Typical",
                "category": {
                    "c1": "Room-Kitchen",  // included -- Kitchen's Typical override (APC-2) was for the CEILING, not the floor, so it does not exclude Kitchen from this FLOOR material's explosion
                    "c2": "Room-Restroom",
                    "c3": "Room-Dining",
                    "c4": "Room-UtilityRoom",
                    "c5": "Room-Bedroom"   // Bar omitted -- Bar has its own room-specific Typical row for a FLOOR material (C-2) in this same schedule. Exclusion is keyed on the ROOM having its own override for the SAME surface (floor), not on matching material type or on the room appearing in the (separate) ceiling exclusion set.
                },
                "mentions": [
                    {"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Standard Materials & Finishes Schedule"}
                ]
            },
            {
                "name": "C-2",
                "notes": "Type: Sealed Concrete, Colour: Grey, Location: Bar Typical",
                "category": "Room-Bar",   // room-specific Typical -> single-room category directly, same pattern as APC-2/Kitchen
                "mentions": [
                    {"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Standard Materials & Finishes Schedule"}
                ]
            },
            // Note by contrast: APC-1 (a CEILING material, shown further above) correctly omits Kitchen from ITS explosion, because APC-2's Kitchen-Typical override is a ceiling override and therefore belongs to the CEILING exclusion set that APC-1 consults. APC-1 does not consult the FLOOR exclusion set, so Bar (a floor-only exclusion) does not affect APC-1's explosion at all -- APC-1's explosion still includes Bar.

 — GENERAL FALLBACK (lowest precedence, only if Steps 1-2 don't apply or the drawing gives no clear location/category information): Infer category from context in the 'notes' field or from the drawing itself. E.g. if notes mention "Part of Porch Roof Assembly (R2)" -> category = "Roof". If Asphalt Shingles are labeled in the roof area of the drawing -> category = "Roof".

    🚨🚨🚨WALL RULE (TAKES PRECEDENCE OVER the generic Wall-Interior/Wall-Exterior fallback used elsewhere in this prompt, including the Room Tag Legend override and the MATERIALS/FINISHES schedule wall handling below):
     - If the drawing gives the wall a specific type name/code (e.g. "P1", "P2", "P3", "Wall-01", "W1", or any other wall-type mark/tag), DO NOT categorize it as "Wall-Interior" or "Wall-Exterior". Instead, set the category to: "Wall-<WallTypeName>" using the exact wall-type name/code as it appears on the drawing. Examples:
         - Wall tagged "P1"      -> category: "Wall-P1"
         - Wall tagged "P3"      -> category: "Wall-P3"
         - Wall tagged "Wall-01" -> category: "Wall-01"
         - Wall tagged "W1"      -> category: "Wall-W1"

     - 🚨 WALL-TYPE CODES ARE NOT ALWAYS LETTER-PREFIXED, AND ARE OFTEN SHOWN INSIDE A GRAPHIC TAG SYMBOL, NOT AS PLAIN TEXT (CRITICAL -- easy to miss): a wall-type code can be a bare number or short alphanumeric string with no descriptive prefix at all (e.g. "0", "0A", "0B", "0C", "1", "1A", "1B", "1C", "1D", "2", "2A", "3"), printed INSIDE a small tag shape (a diamond, hexagon, circle, or similar symbol) rather than written as loose text. Check the sheet's own "PLAN SYMBOLS" / legend key -- if it defines a shape as a "WALL TAG" (or similar), then EVERY instance of that shape anywhere on the sheet (including next to each row of a "WALL ASSEMBLY" cross-section schedule/legend, and on the floor/roof plan itself) carries a wall-type code, and the text inside that shape IS the code -- read it as carefully as you would a P1/P2/P3 label. Do not dismiss a bare "0" or "1A" inside a tag shape as "not a real code" just because it lacks a letter prefix like "P" or "W" -- it is exactly as authoritative as "P1" would be. A "WALL ASSEMBLY" legend/schedule that pairs a wall-tag symbol with a cross-section drawing and material list (even though it's laid out as illustrated details rather than a plain text grid) is a wall-type coding system for this document, exactly like a Partition Type legend -- every material listed in one of its rows/cells must be assigned "Wall-<code>" using that row's tag code (e.g. a row tagged "0A" listing "8\" Concrete Wall, 1.5\" R7.5 Rigid Insulation, ... 1/2\" Stucco or G.W.B." -> every one of those materials gets category "Wall-0A", not "Wall-Exterior").
     - 🚨 DOCUMENT-WIDE CODE CHECK (CRITICAL -- READ BEFORE EVER USING "Wall-Interior"/"Wall-Exterior"): before falling back to a generic interior/exterior label for ANY wall-functional material, check whether this drawing SET uses wall-type codes/tags anywhere at all (e.g. a Partition Type legend with P1/P2/P3, a Wall Type schedule with W1/W2, a "WALL ASSEMBLY" legend tagged with diamond/hexagon symbols containing codes like 0/0A/1/1A, etc.), even if that code isn't the one printed next to THIS particular material.
       - If the document DOES use wall-type codes somewhere, then for a wall-functional material where you cannot pin down which specific wall-type code it belongs to, do NOT guess "Wall-Interior" or "Wall-Exterior" from the room/location name. Instead, use the plain category "Wall" (no suffix). "Wall" means "this is a wall-functional material, but this document identifies walls by type code and I could not determine which specific code applies here" -- it is NOT the same as "Others".
       - If the document DOES not use wall-type codes somewhere, then use "Wall-Interior" or "Wall-Exterior" from the room/location name. 
       - Only use "Wall-Interior" / "Wall-Exterior" as a fallback when the document does NOT use wall-type codes anywhere for walls (i.e. walls are referred to only generically, with no P1/P2/P3-style or diamond/hexagon-tag-style coding system at all in the drawing set). In that case, an ordinary interior space (Restroom, Kitchen, Dining, Bar, Corridors, Storage, Office, etc.) still defaults to "Wall-Interior", and a Location naming "Exterior", "Ext.", or an outdoor-facing element still defaults to "Wall-Exterior".
     - If you cannot determine ANY wall-type code AND the document has no wall-type coding system AND the Location text gives no hint of interior vs. exterior at all (truly ambiguous, e.g. a bare code with no location text anywhere), provide category as "Others".
     - Do NOT use "Others" just because you're unsure whether a specific wall-type code applies -- use "Wall" (if the document has a wall-type coding system) or default to "Wall-Interior"/"Wall-Exterior" (if it does not) as described above. "Others" is reserved for cases where you genuinely cannot infer anything about the wall from what's available.
     - Do not mix up "Wall-Wallname", "Wall", and "Wall-Interior/Exterior" -- they are three distinct outcomes of the same decision, not interchangeable.
     - If same materials are provided in different walltype, provide multiple categories in the same material.
          
    - 🚨 MANDATORY FINAL SELF-CHECK (run this once, AFTER you have drafted every material object, right before you output the final JSON array -- this is not optional and applies to the WHOLE document's output, not per-page):
         STEP 1: Scan every material object you have drafted. Does ANY object anywhere in the array have a category (or any "c1"/"c2"/... value inside a category object) matching the pattern "Wall-<code>" where <code> is a wall-type tag (e.g. "Wall-P1", "Wall-0A", "Wall-1", "Wall-W1")?
         STEP 2: If YES (even a single one) -- this document uses a wall-type coding system. You must now find EVERY object anywhere in the array whose category (or any c1/c2/... value) is the literal string "Wall-Interior" or "Wall-Exterior", and change that value to plain "Wall" instead. This includes materials from completely different pages/sheets/schedules than where you found the code -- the rule is document-wide, not page-by-page. There must be ZERO occurrences of "Wall-Interior" or "Wall-Exterior" anywhere in your final output if even one "Wall-<code>" value exists anywhere in that same output.
         STEP 3: If NO "Wall-<code>" value exists anywhere in the whole array, "Wall-Interior"/"Wall-Exterior" values are fine as-is; leave them.
         This self-check OVERRIDES whatever category you assigned a material earlier in your reasoning -- a material you drafted as "Wall-Exterior" while looking at one page must be corrected to "Wall" if you find a "Wall-<code>" value anywhere else in the document, even on an unrelated sheet (e.g. a window-installation detail using generic "stucco"/"siding" callouts must be corrected to "Wall" if a Partition Type or Wall Assembly legend exists anywhere in the set).
     Example (bare numeric/alphanumeric wall-tag codes inside diamond symbols, per a "WALL ASSEMBLY" legend where the sheet's Plan Symbols key defines the diamond shape as "WALL TAG"):
     {
        "name": "Concrete Wall",
        "notes": "8\" CONCRETE WALL. Wall assembly tagged 0A: 8\" concrete wall, 1.5\" R7.5 rigid insulation, 1-5/8\" light gage galvanized furring channel @16\" O.C., 1/2\" stucco or G.W.B.",
        "category": "Wall-0A",
        "mentions": [
            {"page_label": "A006 - Roof Plan", "view": "Wall Assembly Legend"}
        ]
     },
     {
        "name": "Rigid Insulation",
        "notes": "1.5\" R7.5 RIGID INSULATION. Wall assembly tagged 0A.",
        "category": "Wall-0A",
        "mentions": [
            {"page_label": "A006 - Roof Plan", "view": "Wall Assembly Legend"}
        ]
     },
     {
        "name": "Wood Stud",
        "notes": "2x6 @16\" WOOD STUD. Wall assembly tagged 1: siding grey blue, building wrap, 15/32\" OSB sheathing, R-20 insulation, 2x6 @16\" wood stud, 1/2\" gypsum board, sheetrock.",
        "category": "Wall-1",
        "mentions": [
            {"page_label": "A006 - Roof Plan", "view": "Wall Assembly Legend"}
        ]
     },
     Example: If Gypsumboard is provided in both wall type Wall-01 and Wall-02 then, JSON shold look like:
     {
        "name": "Gypsumboard",
        "notes": "Gypsumboard in each side",
        "category": {
            "c1": "Wall-01",
            "c2": "Wall-02",
        }
        "mentions": [
             {
                "page_label": "C - 202 - Details",
                "view": "Wall partitions"
             }
        ]
     }
    
   - Only use the generic categories "Wall-Interior" or "Wall-Exterior" when the drawing SET as a whole has NO wall-type coding system at all -- i.e. no wall in the entire document is ever referred to by a type name/code (no P1/P2/P3, no W1/W2, etc.), only generically (e.g. "existing wall to remain", an unlabeled partition line, or a wall description with no tag/mark at all anywhere in the set).
   - If the drawing SET does use a wall-type coding system somewhere, but THIS particular wall-functional material cannot be tied to one of those specific codes, use the plain category "Wall" instead -- never fall back to guessing "Wall-Interior"/"Wall-Exterior" from the room/location name in that case.
   - This applies regardless of WHICH other rule in this prompt is telling you to assign "Wall-Interior"/"Wall-Exterior" (Room Tag Legend override, inherently-wall-surface materials like FRP, MATERIALS & FINISHES schedule handling, etc.) -- always check for a wall-type name/code FIRST, for every wall-related material, before applying any generic interior/exterior label, and check whether the document uses a coding system at all before ever choosing "Wall-Interior"/"Wall-Exterior". Never output both a "Wall-WallName" value AND the generic "Wall-Interior"/"Wall-Exterior" pair for the same material -- they are mutually exclusive, and wall-type name always wins when one exists.

   🚨🚨🚨CRITICAL🚨🚨🚨
   - 🚫 MUTUAL EXCLUSIVITY (HARD RULE): "Wall-<WallTypeName>", "Wall", and "Wall-Interior"/"Wall-Exterior" can NEVER appear together for the same wall/material:
       - If a wall-type name/code EXISTS for that wall -> you may ONLY use "Wall-<WallTypeName>". You may NEVER also use "Wall-Interior", "Wall-Exterior", or "Wall" for it, whether as a single value, inside a "category" object (c1/c2), or across duplicate/split entries for the same wall.
       - 🚨If NO wall-type name/code exists for that wall, BUT the DOCUMENT USES WALL-TYPE CODE ELSEWHERE -> use "Wall" ONLY. You may NEVER use "Wall-Interior"/"Wall-Exterior" (that would be guessing) or "Wall-<WallTypeName>" (there is no code for this one) for it.
       - If NO wall-type name/code exists for that wall AND the DOCUMENT HAS NO WALL-TYPE CODE ANYWHERE -> you may ONLY use "Wall-Interior" and/or "Wall-Exterior". You may NEVER use "Wall-<WallTypeName>" or "Wall" for it.
     These three forms are mutually exclusive outputs for a given wall - pick exactly one path per wall and never mix them.

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
    - **Variation Handling**: If the same material `name` appears elsewhere but has a different `category` and different `notes`, it MUST be listed as a completely separate object in the main list.

    **If Lighting, Electrical, Plumbing and Mechnical Schedule comes, IGNORE them.***
    
    🚨❗IMPORTANT: If Accessories, fitting schedules or fixture schdules comes in a pdf, then make sure you fill the 'category' field with 'Others'. (See category E)  But if MATERIALS or FINISHES schedule comes, then use location for category if loaction is provided in the table. 

***Also look at the Room Ledgends/Room Tags for category***

🚨 PRIORITY ORDER FOR ROOM-TAG / ROOM-LEGEND MATERIALS (decide this FIRST, before anything else in this section):
    1. Check whether a MATERIALS/FINISHES SCHEDULE table exists somewhere in the document AND that specific table has its own "Location"/"Room"/"Applied In" column with actual room names filled in for that row. Only in that case is the schedule the source of truth for that tag's category -- use the schedule's Location column to build the category (e.g. "Room-Dining", "Room-Storage", ...).
    2. 🚨 CRITICAL -- DO NOT CONFUSE "a schedule exists" WITH "the schedule has a Location column": a "STANDARD MATERIALS & FINISHES SCHEDULE" or similar table that only lists TAG / TYPE / BRAND-MANUFACTURER / STYLE-COLOR-SIZE-FINISH (i.e. no Location/Room column anywhere in that table) does NOT satisfy step 1 for ANY of its rows, even though a schedule is technically present on the page. For every tag in that kind of schedule, you MUST fall through to the Room Tag boxes on the floor plan (the small "FLOOR / WALL BASE / WALL / CEILING" boxes printed next to each room, e.g. next to "PRIEST ROOM", "KITCHEN", "LOBBY", "OFFICE ROOM", "FOYER", "ENTRANCE PORCH", "PRAYER HALL") and use the ROOM TAG ROW-TO-CATEGORY MAPPING below. Read the room's actual name printed on/near each box and use it for the ROOM-based rows as described below. Do this per-tag, not per-document: it is normal for one document to have some tags resolved via a Location column and other tags (from a schedule with no Location column) resolved via Room Tag boxes instead.
    3. If a genuine Location column with room names IS present for a tag (step 1), prefer that column for that tag's category, but you may still cross-check tag identity against the Room Tag boxes on the plan.
    4. Self-check before finalizing every FL-/WB-/PT-/CT- style tag: did I actually find a filled-in "Location" cell for this exact tag in a schedule table, or did I only see TAG/TYPE/BRAND/STYLE columns? If the latter, go scan the floor plan's Room Tag boxes for this tag code -- never leave the category blank, "Others", or guessed from the tag name alone.

    5. 🚨 A "Room Tag box" is NOT limited to boxes printed next to a fully enclosed, four-walled ROOM. Any small bordered box on the floor plan that sits next to a printed zone/area/furniture label and lists one or more finish tags is a Room Tag box, and you MUST scan and use ALL of them -- this includes labels such as "PEDESTAL", "DEITIES PLATFORM", "ENTRANCE PORCH", "ENTRY STAIR & LANDING", or any other named zone, not only conventional rooms like "KITCHEN" or "LOBBY". Do not skip a box just because the word next to it isn't literally a "room" -- treat that printed word/phrase as the <RoomName> exactly like you would for an enclosed room (e.g. a box under "PEDESTAL" listing FL-1 -> "Room-Pedestal"; a box under "DEITIES PLATFORM" listing FL-2/WB-3/PT-2,PT-3/CT-2 -> "Room-Deities Platform" for the Floor and Ceiling rows).

    6. 🚨 Room Tag boxes do NOT always contain all four FLOOR/WALL BASE/WALL/CEILING rows -- many contain only one or two rows (e.g. a "PEDESTAL" box that lists only a single FL-1 row, with no WALL BASE/WALL/CEILING rows at all, because a pedestal has no walls or ceiling of its own). When a box has fewer than 4 rows, do NOT try to force a position-based reading (1st row = Floor, 2nd = Wall Base, etc.) -- instead identify each present row's function from the TAG'S OWN PREFIX/TYPE, exactly as you would for the full 4-row case: a tag starting with FL/FP (or otherwise identifiable as a floor material) is always a FLOOR row -> "Room-<Label>"; WB is always a WALL BASE row -> WALL RULE; PT/wall-panel/wall-tile codes on that row are always a WALL row -> WALL RULE; CT/C- (or otherwise identifiable as a ceiling material) is always a CEILING row -> "Room-<Label>". This prefix-based identification applies regardless of how many rows are actually printed in that specific box, and regardless of whether the box's rows are even labeled "Floor"/"Wall"/etc. at all.

    7. If the same zone label (e.g. "PEDESTAL") appears as multiple separate boxes at different physical locations on the plan (as is common -- multiple freestanding pedestals each with their own box), treat them as multiple mentions of the SAME "Room-Pedestal" category (one mention per box instance, all pointing at the same category string) rather than inventing numbered variants like "Room-Pedestal 1"/"Room-Pedestal 2" -- unless the plan itself prints a distinguishing name/number for each instance, in which case use that printed distinguishing name instead.

Example for materials/finishes schedule WITH a Location column (use schedule per step 1):
TAG     |          Type         | Brand     | Location                                                             
CT-1    |Vinyl Coated Ceiling   |Armstrong  | Dining, Storage, Toilet                       
PT-1    |Interior Latex Paint   | Daltile   | Dining, Storage, Kitchen                      
WB-1    |Rubber base Glove      |Rope       | Lobby, Foyer, Prayer-Hall                    
FP-1    |Floor tile             |Armstrong  | Lobby

Example for a "STANDARD MATERIALS & FINISHES SCHEDULE" WITHOUT a Location column (this does NOT satisfy step 1 -- fall through to step 2 / Room Tag boxes for every one of these tags):
TAG     |          TYPE                | BRAND/MANUFACTURER  | STYLE / COLOR / SIZE / FINISH
FL-3    | CERAMIC TILE                 | TRAFFICMASTER        | Portland Stone Beige - Anti Slip 18"x18" Glazed
PT-1    | INTERIOR LATEX WALL PAINT    | SHERWIN-WILLIAMS     | SW-7551 (Greek Villa) or approved equal
CT-1    | ACOUSTIC CEILING TILE        | ARMSTRONG            | Kitchen Zone 2'-0"x2'-0" Sq Lay-In Ceiling Tile - White
-> None of these rows tell you WHICH room they apply to. You must instead scan the floor plan for the Room Tag boxes (e.g. a box next to "PRIEST ROOM" listing FL-3 / WB-1 / PT-1 / CT-1) and read the room name from there, per the mapping below.

Example for Room ledgend/Room Tag (the small box shown on the floor plan, e.g. next to "PRIEST ROOM"):
RoomName 
_____________                          
|Floor      |                       
|Wall Base  |                       
|Wall       |                       
|Ceiling    |
|___________|                       

In drawings, there might be markings like:
Priest Room
_________
|FL-3   |
|WB-1   |
|PT-1   |
|CT-1   |
|_______|

🚨 ROOM TAG ROW-TO-CATEGORY MAPPING (apply this per row when using the Room Tag Legend, i.e. when NO Materials/Finishes schedule with a Location column exists for that tag):
    - FLOOR row (e.g. FP-1, FL-3, FL-1)      -> the tag on this row gets category "Room-<RoomName>", where <RoomName> is the actual zone/area label printed on/next to that specific Room Tag box on the floor plan (e.g. "PRIEST ROOM" -> "Room-Priest Room"; "PEDESTAL" -> "Room-Pedestal"; "DEITIES PLATFORM" -> "Room-Deities Platform"). This is exactly the same treatment as a floor tile with a known Location.
    - WALL BASE row (e.g. WB-1, WB-3)  -> the tag on this row is a wall-functional material. Do NOT use "Room-<RoomName>" for it -- instead apply the WALL RULE precedence below (Wall-<code> if a wall-type code exists for that wall, else plain "Wall" if the document uses a wall-type coding system elsewhere, else "Wall-Interior"/"Wall-Exterior").
    - WALL row (e.g. PT-1, PT-2, PT-3)       -> same treatment as WALL BASE -- wall-functional, goes through the WALL RULE, never "Room-<RoomName>".
    - CEILING row (e.g. CT-1, CT-2, C-4)    -> the tag on this row gets category "Room-<RoomName>", using the SAME zone/area label read from that Room Tag box, exactly like the FLOOR row. Ceiling tags are treated as room-located, just like floor tags -- do NOT put ceiling tags through the WALL RULE and do NOT default them to "Others".
    In short: FLOOR and CEILING rows of a Room Tag Legend both resolve to "Room-<RoomName>" (using the zone/area label printed at that specific tag instance); WALL BASE and WALL rows both resolve through the WALL RULE instead. When the Room Tag Legend is the category source for a given tag, do NOT use a schedule's own Location text if one exists for a different mention of the same tag with no Location -- read the zone/area label directly off the plan next to that instance of the box.
    🚨 IDENTIFY EACH ROW BY THE TAG ITSELF, NOT BY ITS POSITION IN THE BOX: many Room Tag boxes (e.g. a "PEDESTAL" box) print only ONE row -- just a bare FL-1 with no "Floor"/"Wall"/etc. label at all. Do not assume "the only row in this box must mean something different because it isn't in a 4-row box" -- a lone FL-1 next to "PEDESTAL" is still a FLOOR-row tag by virtue of being an FL-code, so it still resolves to "Room-Pedestal", exactly as it would if printed as row 1 of a full 4-row box. Apply this per tag code (FL-/FP- => Floor; WB- => Wall Base; PT- or other wall-panel/wall-tile code on that row => Wall; CT- or C- => Ceiling) regardless of how many rows the specific box happens to contain.
For Room ledgend, look at the drawing properly. Every separate Room Tag box on the plan -- next to every named zone, whether a fully enclosed room, an open platform, or a single piece of furniture like a pedestal -- is a separate mention with its own zone label. e.g. a floor tag appearing next to "ROOM A" and again next to "ROOM B" produces category values "Room-Room A" and "Room-Room B" respectively; a floor tag appearing next to three separate "PEDESTAL" boxes and nowhere else produces the single category value "Room-Pedestal" (repeated as multiple mentions, not multiple c1/c2/c3 values, since the label is identical each time) (combined into a category object with c1/c2/... only when the SAME tag repeats next to DIFFERENT zone labels, per the multi-category rules elsewhere in this prompt).

🚨🚨 CROSS-BOX CONTAMINATION -- THE SINGLE MOST COMMON ERROR ON DENSE FLOOR PLANS (read carefully, this WILL happen if you are not deliberate about it):
    Floor plans routinely place several DIFFERENT Room Tag boxes physically close together -- e.g. a "LOBBY" box sitting right next to a "FOYER" box, or a "PRIEST ROOM" box sitting near an unrelated "OFFICE ROOM" box. It is very easy to accidentally read one box's tag list while attaching it to the NEIGHBORING box's room name, or to only register a tag under the first nearby room you saw and stop looking. To avoid this:
    (a) For EVERY Room Tag box, re-verify by tracing: which exact tag characters are printed INSIDE this specific box's own border, and which exact room-name text is printed directly above/beside THIS SAME box (not a box one or two boxes over). Do not let a tag "borrow" a room name from an adjacent box just because they are close together on the sheet, and do not let a room name "borrow" tags from a neighboring box.
    (b) Do NOT stop after finding one box that contains a given tag code. The SAME tag code (e.g. FL-1, FL-4, CT-1) commonly appears in the Room Tag boxes of MULTIPLE different rooms across the whole sheet. You must scan every single Room Tag box on every page before finalizing a tag's category -- treat this as a full-sheet search per tag, not a "first match wins" search. If you already found FL-1 in one room's box, keep looking for it in every other room's box too.
    (c) When the SAME room-name text appears MORE THAN ONCE on the plan (e.g. a mirrored/symmetric layout with two separate "PRIEST ROOM" boxes on opposite sides of the sheet), both instances are the SAME room and should normally carry the SAME tag set -- read each occurrence's own box independently to confirm, but do not "invent" a different room name for the second occurrence based on what a nearby, differently-named room happens to contain. If a second box's own printed tags genuinely differ from the first same-named box, re-examine both boxes' room-name text specifically before concluding they differ -- a misread digit or word is far more likely than two genuinely different rooms sharing a name.
    (d) Self-check before finalizing output: for every distinct room-name label visible anywhere on the plan (including repeated/mirrored ones), list out that room's own 4 rows (or however many rows its box has) directly from that box, independently of any other room's box -- then confirm every tag from that box appears in your final category output for that exact room name. Do not skip a room just because a same-shaped box nearby was already processed.
    (e) A tag box on the plan with NO room-name text printed anywhere near it (a bare tag in a corridor, walkway, or step area with only a dashed leader line and no adjacent room label) still belongs to whatever named room/zone's floor area it physically sits inside, based on the wall/boundary lines on the plan -- do not silently drop these mentions, and do not guess a room name from a different, unrelated part of the sheet. If you truly cannot determine which room such a floating tag belongs to after checking the boundary lines, keep the mention but note the uncertainty in "notes" rather than omitting it.
    (f) A dashed "EXTENT OF <tag>" annotation is a boundary/coverage indicator for a tag already captured on that same room's box -- it is NOT a new tag mention and does NOT create a new room category; ignore it for categorization purposes (it only confirms where that tag's material physically extends within the room already identified).

Example when ONLY the Room Tag Legend is present on the plan (no Materials/Finishes schedule Location column for these tags) -- a box reading "PRIEST ROOM" with rows FL-3 / WB-1 / PT-1 / CT-1 next to it:
    {
        "name": "FL-3",
        "notes": "Room Tag: Floor row, Room: Priest Room",  # Add the note from the table
        "category": "Room-Priest Room",   # FLOOR row -> Room-<RoomName>, room name read directly off the plan next to this Room Tag box
        "mentions": [
            {"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}
        ]
    },
    {
        "name": "CT-1",
        "notes": "Room Tag: Ceiling row, Room: Priest Room", # Add note from the table
        "category": "Room-Priest Room",   # CEILING row -> SAME Room-<RoomName> treatment as FLOOR, using the same room name
        "mentions": [
            {"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}
        ]
    },
    {
        "name": "WB-1",
        "notes": "Room Tag: Wall Base row, Room: Priest Room", # Add note from the table
        "category": "Wall",   # WALL BASE row -> goes through the WALL RULE (Wall-<code> / "Wall" / Wall-Interior-Exterior), NOT Room-Priest Room
        "mentions": [
            {"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}
        ]
    },
    {
        "name": "PT-1",
        "notes": "Room Tag: Wall row, Room: Priest Room", # Add note from the table
        "category": "Wall",   # WALL row -> also goes through the WALL RULE, same as Wall Base
        "mentions": [
            {"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}
        ]
    },
 
    * If FL-3 is also presdent in other room (Dining for now), then the category must be given as:
    "category": {
        "c1": "Room-Priest Room",
        "c2": "Room-Dining",
    }

    Example when the Room Tag box is next to a non-room ZONE (platform/furniture label) and only has ONE row -- e.g. three separate boxes on the plan each simply reading "PEDESTAL" with a single "FL-1" line inside (no Wall Base/Wall/Ceiling rows at all), plus a 4-row box reading "DEITIES PLATFORM" listing FL-2 / WB-3 / PT-2,PT-3 / CT-2:
    {
        "name": "FL-1",
        "notes": "Room Tag: Floor row, Zone: Pedestal",
        "category": "Room-Pedestal",   # lone FLOOR-prefixed tag in a 1-row box -> still Room-<Label>, using the printed zone label "PEDESTAL" even though it is not an enclosed room
        "mentions": [
            {"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"},
            {"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"},
            {"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}
        ]   # three separate PEDESTAL boxes on the plan all resolve to the same "Room-Pedestal" category -- one mention per box instance
    },
    {
        "name": "FL-2",
        "notes": "Room Tag: Floor row, Zone: Deities Platform",
        "category": "Room-Deities Platform",   # FLOOR row of the DEITIES PLATFORM box
        "mentions": [{"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}]
    },
    {
        "name": "CT-2",
        "notes": "Room Tag: Ceiling row, Zone: Deities Platform",
        "category": "Room-Deities Platform",   # CEILING row -> same zone label as the Floor row above
        "mentions": [{"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}]
    },
    {
        "name": "WB-3",
        "notes": "Room Tag: Wall Base row, Zone: Deities Platform",
        "category": "Wall",   # WALL BASE row -> WALL RULE, never "Room-Deities Platform"
        "mentions": [{"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}]
    },
    {
        "name": "PT-2",
        "notes": "Room Tag: Wall row, Zone: Deities Platform",
        "category": "Wall",   # WALL row -> WALL RULE
        "mentions": [{"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}]
    },
    {
        "name": "PT-3",
        "notes": "Room Tag: Wall row, Zone: Deities Platform",
        "category": "Wall",   # also listed on the WALL row (multiple tags can share a row, e.g. "PT-2, PT-3") -> WALL RULE
        "mentions": [{"page_label": "C - 202 - Finishes Plan and Schedule", "view": "Finishes Plan"}]
    },

     🚨 IMPORTANT — this Room Tag Legend override is NOT limited to coded materials (WB-, CT-, PT-, etc). It applies equally to plain-named, non-coded materials whenever the material's function matches one of the legend rows (FLOOR / WALL BASE / WALL / CEILING). Judge this by what the material physically IS, not by its code format:
    - If a material is inherently a wall-surface material (e.g. "FRP" / Fiber Reinforced Plastic panels, "Stainless Steel Wall Panels", ceramic wall tile, wainscot, wall cladding, paneling) -- it belongs on the WALL row of the Room Tag Legend regardless of what the "Location" field says (e.g. "Location: Kitchen"). Do NOT fall back to "Room-Kitchen" in this case. The Location text stays in "notes" for reference, but does not drive "category" once the material-type identifies it as a wall item.
      Then apply the WALL RULE precedence to decide the actual category value:
        1. If a specific wall-type name/code (e.g. "P1", "Wall-01") applies to that wall -- use "Wall-WallName" ONLY.
        2. If no wall-type name/code exists for THIS wall, but the drawing set uses a wall-type coding system somewhere -- use plain "Wall" ONLY (do not guess interior/exterior).
        3. Only if the drawing set has NO wall-type coding system anywhere -- fall back to providing BOTH "Wall-Interior" and "Wall-Exterior" as shown.
    - Only fall back to using "Location" for the category when the material's real-world function is ambiguous (i.e., it is not clearly a floor, wall, wall-base, or ceiling material by name/type alone).

    Example A (drawing set has NO wall-type coding system anywhere -- generic interior/exterior fallback applies):
    {
        "name": "FRP",
        "notes": "Type: FIBER REINFORCED PLASTIC, Style/Color/Size/Finish: 4'-0\" X 10'-0\" PANELS - WHITE COMMERCIAL GRADE OR APPROVED EQUAL, Location: KITCHEN",
        "category": {
            "c1": "Wall-Interior",
            "c2": "Wall-Exterior", # FRP is inherently a wall-panel material, so it maps to the WALL row of the Room Tag Legend regardless of Location: Kitchen. No wall-type name/code (like P1, Wall-01) was found ANYWHERE in this ENTIRE drawing set, so both generic categories are given. If this drawing set had a wall-type coding system anywhere (even for other walls), the category would instead be "Wall-WallName" (if this wall has its own code) or plain "Wall" (if it doesn't) -- never this Interior/Exterior guess.
        }   
        "mentions": [
            {
                "page_label": "C - 202 - Finishes Plan and Schedule",
                "view": "Standard Materials & Finishes Schedule"
            }
        ]
    },

    Example B (drawing set DOES use wall-type codes elsewhere in the set, e.g. a P1/P2/P3 partition-type legend, but THIS material's row/mark cannot be tied to a specific one of those codes):
    {
        "name": "Rubber Base Cove",
        "notes": "Type: Rubber Base Cove, Location: Reception",
        "category": "Wall",   # NOT "Wall-Interior". This drawing set has a P1/P2/P3 wall-type coding system elsewhere (e.g. a Rated Partition legend), so guessing interior/exterior from the room name is not allowed. Since this occurrence (from a room-by-room Finishes Schedule matrix) cannot be tied to a specific P1/P2/P3 code, use plain "Wall" instead.
        "mentions": [
            {
                "page_label": "C - 103 - Proposed Floor Plan",
                "view": "Finishes Schedule"
            }
        ]
    },

    #### CATEGORY A: STANDARD MATERIALS (No Codes Present)
    Use this formatting if there is absolutely no schedule code (like F-60 or X-02) associated with the material.
    - Provide "name", "notes", "category" and "mentions". Do NOT include a "code" key.

    Example:
    {
        "name": "Gypsum Drywall",
        "notes": "1/3' Gypsum Drywall",
        "category": {
            "c1": "Wall-Interior", 
            "c2": "Roof",
        },
        "mentions": [
          {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
          {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
        ]
    },

    If same material is used but of different size (e.g., 1/2' vs 5/8' Gypsum Drywall) then they must be listed as separate items because the notes are different.

    
    #### CATEGORY B: CODED MATERIALS & SCHEDULES (Codes Present)
    If the code is a fixture, see CATEGORY E below. Otherwise, if the code is a material or finish, use this formatting.
    Use this formatting if a code (e.g., F-60, X-02) is detected anywhere on the drawing or inside a schedule layout. 🚨REMEMBER: Use this format if it is a schedule. the code should be the name and all the otehr info must be in notes section. Try to inclde name of material in notes.
    
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
        "category": "Room-MainRoom",
        "mentions": [
          {"page_label": "Sheet 6 of 23", "view": "Main Floor Plan Layout"}
        ]
    },
    {
        "code": "F-62",
        "notes": "FLOOR TILE, ~2\", CERAMIC, HEXAGONAL PATTERN"
        "category": "Room-Kitchen Floor",
        "mentions": [
          {"page_label": "Sheet 6 of 23", "view": "Main Floor Plan Layout"}
        ]
    },
    {
        "name": "CT-1",
        "notes": "Type: Vinyl Coated Ceiling,  Brand: Armstrong, Location: Dining, Storage, Toilet",
        "category": "Room-Toilet",
        "mentions": [
            {
             "page_label": "C - 301 - Materials and Finishes Schedule Schedule",
             "view": "Material and Finishes Schedule"
            }
        ],
    },

    🚨 NEVER DROP THE "ITEM" (OR EQUIVALENT NAME/TYPE/DESCRIPTION)or any COLUMN FROM "notes" (CRITICAL):
    - This rule applies to EVERY schedule table regardless of how many columns it has (4, 8, 15+) -- it is not limited to the Item/Material example below, which is only ONE illustration of the general rule.
    - A schedule row commonly has BOTH an "Item"/"Type"/"Description" column (what the thing is called, e.g. "BRICK", "PORCH DECKING", "CROWN MOLDING") AND a separate "Material" column (what it's made of, e.g. "SMOOTH BRICK", "PAINTED WOOD", "ROT-RESISTANT"). These are NOT the same value and NEITHER may be dropped for looking similar to the other -- both must appear in "notes", each labeled with its own original column header.
    - Fold EVERY column from the row into "notes", NO MATTER HOW MANY there are, in the SAME left-to-right order they appear in the table, each one labeled with its own header text exactly as printed (e.g. "ITEM:", "SIZE:", "MATERIAL:", "NOTES:", "MANUFACTURER/MODEL:", or whatever headers that specific table actually has). Do not skip any column just because a later column looks related to it in meaning -- every distinct column answers a distinct question and a downstream reader needs all of them.
    - Before finalizing each row's "notes" string, count the number of columns in that table's header row and count the number of labeled segments you produced for this row -- they must match. If they don't match, you dropped a column; go back and find which one.

    🚨 EXPAND PAGE-LEVEL ABBREVIATION LEGENDS IN "notes" (CRITICAL -- e.g. "VW" = "VINTAGE WOODWORKS"):
    - Pages sometimes print a short legend line near a list, table, or option group defining what an abbreviation/prefix used in that section stands for, e.g. "(NOTE: VW = \"VINTAGE WOODWORKS\")" printed above a list of items like "VW COCKATOO 1194", "VW MARIPOSA 2229", "VW RILEY 1551". Whenever a manufacturer/model value uses an abbreviated prefix like this, you MUST resolve it and write the FULL name into "notes" -- do not leave the bare abbreviation unexpanded.
    - Format it as: "Manufacturer: <Full Name> (<Abbreviation>), Model: <model number/name>". Example: "VW MARIPOSA 2229" with legend "VW = VINTAGE WOODWORKS" becomes "Manufacturer: Vintage Woodworks (VW), Model: Mariposa 2229" in "notes".
    - This legend may appear on a DIFFERENT page than the specific mention of the abbreviated code (e.g. the legend is on the options-list page, but the code also appears on a detail/elevation page). Search ALL pages provided in this same request for a "<ABBR> = <FULL NAME>" style legend before leaving any abbreviation unexpanded, the same way you would cross-reference a schedule code.
    - NEVER invent or guess an expansion. Only expand an abbreviation when its defining legend text is actually visible on one of the pages provided in this request. If no legend is found anywhere in this request, keep the abbreviation as-is in "notes" rather than fabricating a full name.

    Example (MARK | ITEM | SIZE | MATERIAL | NOTES | MANUFACTURER/MODEL row from a Materials Schedule):
    {
        "code": "F-00",
        "notes": "ITEM: BRICK, SIZE: -, MATERIAL: SMOOTH BRICK, NOTES: SOLID RED COLOR, SAND-FACED, AVOID WIRE CUT OR \"EXTRUDED\" LOOK, MANUFACTURER/MODEL: OLD CAROLINA BRICK COMPANY",
        "category": "Wall-Foundation",
        "mentions": [
          {"page_label": "A5.0", "view": "Materials Schedule"}
        ]
    },

    🚨 ONE ROW = ONE ENTRY (CRITICAL -- PREVENTS SPLIT DUPLICATES LIKE "F-26" + "Astragal"):
    - If a code (e.g. F-26) and a row from a schedule table both describe the SAME row -- i.e. the code came from a MARK/TAG/NO column and other fields (item, material, size, notes, manufacturer) came from the same row of that same table -- they must produce exactly ONE material entry per Category B, keyed by the code.
    - Do NOT additionally emit an entry using any other column's value as "name", regardless of whether that value looks like a standalone material (Category A) or a submaterial listed inside a code (Category D). A column value such as an "Item" or "Material" name is notes content for the code's single entry -- it is never its own entry.
    - This rule applies even when the row's fields are presented to you as separate pre-parsed key/value pairs in an OCR-transcribed table reference block -- a pre-split field structure does not change the fact that they belong to one schedule row and must collapse to one entry.

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
    103 | 2  | 7'-0 | 8'-6' | HM            | -
    If the above is a door schdule then name should be Door-01A

    Then the JSON must look like:
    {
        "name": "Door-01A",
        "notes": "Qty: 1, Width: 5'-0, Height: 6'-8', Material Finish: Fibreglass, Glazing: -",
        "category": "Door Schedule",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - Schedules", "view": "Door Schedule"},
        ]
    },
    {
        "name": "Door-103",
        "notes": "Qty: 2, Width:  7'-0, Height:  8'-6', Material Finish: HM, Glazing: -",
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

    For Door and Window, Add the keyword "Door" or "Window" in the name.

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
        "category": "Wall-Interior",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
            {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
        ]
    },
    {
        "name": "E32",
        "notes": "3/8\" OSB Board, Type X, 5/8\" thick, fire-rated",
        "category": "Wall-Interior",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
        ]
    }

    EXAMPLE 3:

    WINDOWS
    1    32x72 DH 2/2 DIVIDED LITES (2)
    2    32x60 DH 2/2 DIVIDED LITES
    3    24x42 DH 2/2 DIVIDED LITES(3)
    DOORS
    A    36x84 9-LITE FRONT DOOR 
    B    36x80 4-LITE BACK DOOR

    Then the JSON must look like:
        {
            "name": "Window-1",
            "notes": "32x72 DH 2/2 DIVIDED LITES, Count 2",
            "category": "Window",
            "mentions": [
                {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
                {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
            ]
        },
        {
            "name": "Window-2",
            "notes": "32x60 DH 2/2 DIVIDED LITES",
            "category": "Window",
            "mentions": [
                {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
                {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
            ]
        },
        {
            "name": "Window-3",
            "notes": "24x42 DH 2/2 DIVIDED LITES, Count 3",
            "category": "Window",
            "mentions": [
                {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
                {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
            ]
        }
        {
            "name": "Door-A",
            "notes": "36x84 9-LITE FRONT DOOR",
            "category": "Door",
            "mentions": [
                {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
                {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
            ]
        },
        {
            "name": "Door-B",
            "notes": "36x80 4-LITE BACK DOOR",
            "category": "Door",
            "mentions": [
                {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
                {"page_label": "Sheet 5 of 23 - West Elevation", "view": "Exterior Materials Schedule"}
            ]
        }
    Here, if any number comes in parenthsis, consider that as the count (number of doors/windows) for Door/Window Schedule
    *Do not miss any rows and columns in the table* Properly extract data from each row and colum to display. If the table has null values, also include them. Keep '-' sign to indicate null values. Do not leave them empty. Try to read the data correctly from the pdf.

    
     #### CATEGORY D Listed submaterials  inside a code
      Use the format below if submaterials are listed inside a code. 🚨IF THE CODE IS A SCHEDULE, LOOK AT CATEGORY B
      Also use this format if submaterials are listed inside a wall type, partition code, or detailed assembly callout (e.g., a detail showing 5 layers of a wall: Siding, Wrap, Sheathing, Studs, Drywall). Try to inclde name of material in notes.
      🚨 This includes FINISH callouts written inline as part of the assembly description, not just physical layers -- e.g. if a wall-type detail says "5/8\" Gypsum Wall Board (both sides) WITH PAINTED FINISH", the phrase "with painted finish" means "Paint" must ALSO be extracted as its own submaterial object for that code (e.g. {"name": "Paint", "category": "Wall-P3", "notes": "Painted finish. Used in P3 Non-Rated Partition assembly.", "mentions": [...code P3...]}), in addition to the Gypsum Wall Board object. Do not drop finish/coating callouts just because they're phrased as an adjective clause ("with painted finish") rather than a listed layer -- if a finish is named as part of that code's assembly text, it is a submaterial of that code just like the studs or insulation are.
    - You MUST split these complex layered assemblies into individual material entries in your JSON output (one object for Siding, one for Wrap, one for Sheathing, etc.).
    - Do NOT dump the entire assembly sentence into a single "notes" key. Parse each material layer separately.
    - Do not use this category if the code is a schedule. Instead, use CATEGORY B for schedules.

    🚨 MANDATORY PRE-CHECK BEFORE APPLYING CATEGORY D:
    Before applying Category D, check: does this code appear as one row of a table with a title (e.g. "MATERIALS SCHEDULE", "DOOR SCHEDULE", "WINDOW SCHEDULE")? If yes, this row has already been captured under Category B -- do NOT create any additional entry from any other column in that same row (item name, material type, etc.), even if that column's value looks like a standalone material name. Category D applies ONLY to codes discovered on plan/detail/elevation views that are NOT part of a tabular schedule -- e.g. a wall-type tag (W1) written directly on a floor plan or detail drawing with its layers listed in a callout sentence, not as a row in a titled schedule table.

      If W1 has listed VINYL SIDING, TYVEK HOUSE WRAP, 3/8' OSB EXTERIOR SHEATHING, 2X6 STUDS @ 16' O.C., R-25 BATT INSULATION and W1 is not a schedule, then:
    
        {
            "name": "VINYL SIDING",
            "notes": "Vinyl Siding, Material listed in W1",
            "category": "Wall-Exterior",
            "mentions": [
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
            ]
        },
        {
            "name": "TYVEK HOUSE WRAP", 
            "notes": "Tyvek House Wrap, Material listed in W1",
            "category": "Wall-Exterior",
            "mentions": [
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
            ]
        },
        {
            "name": "3/8' OSB EXTERIOR SHEATHING", 
            "notes": "3/8' osb exterior sheathing, Material listed in W1",
            "category": "Wall-Exterior",
            "mentions": [
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
            ]
        },
        {
            "name": "2X6 STUDS @ 16' O.C.", 
            "notes": "2X6 size STUDS, 16'o.c. spacing, Material listed in W1",
            "category": "Wall-Foundation",
            "mentions": [             
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"},
            ]
        },
        {
            "name": "R-25 BATT INSULATION", 
            "notes": "R-25 BATT INSULATION, Material listed in W1",
            "category": "Wall-Exterior",
            "mentions": [ 
                {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"},
            ]
        },
    ]

    Do not add any extra json for codes W1 if the materials are already listed in the schedule. Only add the code W1 with the schedule category and notes if there is no material breakdown listed in the schedule. If there  is a material breakdown then do not add W1 code as a separate item. Only add the materials listed under W1 as separate items with their respective categories and notes. TAKE NOTES OF CATEGORY B AND C properly.

    
    🚨 GENERIC-VS-WALL-TYPE DUPLICATE CHECK (CRITICAL):
    Before creating a "Wall-Interior"/"Wall-Exterior" entry for a material sourced from a GENERAL Finish Legend, Materials & Finishes Schedule, or Room finishes table (e.g. a legend code like "RB1 = RUBBER BASE COVE"), check whether that SAME product (by name/material type, e.g. "Rubber Base Cove") ALSO appears inside a wall-type/partition-type assembly detail elsewhere in this same document (e.g. named inside the P1/P2/P3 partition detail callouts). 
    - If it DOES also appear tied to a wall-type code -- do NOT create a separate standalone "Wall-Interior"/"Wall-Exterior" object for it. Instead, treat the wall-type-tagged extraction as authoritative: fold the generic legend/schedule page reference in as an ADDITIONAL mention on the existing "Wall-<WallTypeName>" object(s) for that material (one added mention per applicable code, still following the Multi-Code Mentions Rule above -- do not add the generic mention to every split object indiscriminately if it doesn't apply to all of them).
    - If the SAME product does NOT itself appear tied to a wall-type code, but the document uses a wall-type coding system SOMEWHERE (for other materials/walls), do NOT create a "Wall-Interior"/"Wall-Exterior" object for it either -- use plain category "Wall" instead. Guessing interior vs. exterior from the room/location name is only allowed when the whole document has no wall-type coding system at all.
    - Only create a standalone "Wall-Interior"/"Wall-Exterior" object for a generically-sourced material when NO wall-type/partition coding system exists ANYWHERE in the entire document, for any wall. This is the same "no wall-type name/code exists" condition already described in the WALL RULE -- apply it across the WHOLE document, not just within the single page/table you are currently reading.

        🚨 This same submaterial-breakdown rule ALSO applies when the code's materials are written as a full PROSE SENTENCE instead of a clean comma-separated list -- this is very common in PARTITION / WALL-TYPE SCHEDULES (columns like "PARTITION WALL TYPE" / "TYPE" and "DESCRIPTION"), where a code such as "A1" or "B2" has a description like: "3 5/8\" Metal Stud with one layer of 5/8\" Cementitious Backer Board and Ceramic Tile upto 72\" from FFL (UNO on interior elevations) and painted finish above, on both sides." Do NOT output this as a single object with the whole sentence dumped into "notes" (e.g. do NOT produce {"name": "Partition Wall Type A1", "notes": "<entire sentence>", ...}). Instead, parse the sentence and extract each distinct material mentioned (stud framing, backer board, tile, paint, sheathing, siding, cladding, insulation, etc.) as its OWN object, same as any other CATEGORY D breakdown, using "Extracted from code" to record which wall/partition type it came from.

        
    #### CATEGORY E: FITTINGS, FIXTURES & ACCESSORIES SCHEDULES 
    A table listing tagged fixtures/fittings/hardware (e.g. columns like S.N., TAG, ACCESSORY, ITEM SPECIFICATION -- covering things like toilet paper dispensers, soap dispensers, grab bars, mirrors, lavatories, urinals, water closets, hand dryers, partitions, shower heads, water heaters, refridgerator, etc) is STILL IN SCOPE and MUST be extracted. Do NOT skip this table under the general "civil engineering materials only" rule -- plumbing fixtures, toilet accessories, and fit-out hardware scheduled with their own TAG are treated the same as any other coded schedule item (see CATEGORY B/C).
    - Use the TAG (e.g. "AC-1", "G-1", "L-1", "M-1", "U-1", "WC-1") as the "name".
    - Combine the accessory description and item specification/model columns into "notes".
    - Set "category" to "Others"
    - Every row of this table must be extracted -- do not skip any TAG.
    - FIXTURES SHOULD BE GENERATED ONLY ONCE.
    
    Example:
    {
        "name": "AC-1",
        "notes": "Accessory: Toilet Paper Dispenser. Item Specification: Bobrick Model B2888 or equal.",
        "category": "Others",
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Fittings and Accessories Schedule"}
        ]
    },
    {
        "name": "WC-1",
        "notes": "Accessory: Water Closet, Std. Item Specification: Sloan Model 20231001 or equal.",
        "category": "Others",
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Fittings and Accessories Schedule"}
        ]
    },

    🚨There might occur a case where table is provided with markings indicating the materials are applied in those areas. Investigate them properly and provide notes such that only the marked areas where materials are used is given properly. Carefully read the markings for notes section. For this, follow category F


    #### CATEGORY F: MATRIX / CHECKLIST-STYLE SCHEDULES (e.g. "Material and Finishes Schedule" with rooms/areas as column headers, materials as row headers, and "X" marks at the intersections showing which material applies to which room/surface)

    🚨 DO NOT summarize a matrix table into one prose sentence per material (e.g. do NOT write something like "Applied in: Lobby, Shower Room, Family Restroom..."). That kind of free-text summary is exactly what causes rooms to be added or dropped by mistake. Instead, you MUST treat EVERY marked (X) intersection as its own separate output object, using the "row_label"/"column_label" pairs already provided to you in the "OCR-TRANSCRIBED TABLE DATA" reference block for this page -- do not re-derive them yourself from the image, and do not merge multiple pairs together. Eg: If "Ceramic Tile" is checked for 5 different rooms, there must be 5 distinct JSON objects, each with its own "Room- <RoomName>" category.

    For each row_label/column_label pair in that reference data:
    - "name" = the material's row_label exactly as given (e.g. "Ceramic Tile", "Sealed Concrete").
    - "category" = "Room-<room name>" using the room/area portion of the column_label (e.g. column_label "Shower Room-Floor" -> category "Room-Shower Room").
    - "notes" = the surface portion of the column_label (Floor / Ceiling / Wall-North / Wall-South/ Wall-East / Wall-West, etc.), plus the material's brand/manufacturer/type-color spec if the schedule's legend provides one for that material -- do not invent a spec if none exists.
    - "mentions" = the usual page_label/view for this schedule.
    - One object per intersection -- if "Ceramic Tile" is marked for 4 rooms x 4 walls, that is 5 separate objects (each with a different "category"/"notes", walls are considerd as one), NOT one object with a combined list of rooms in "notes".
    - Do not add a room/surface that is not present in the row_label/column_label reference data, and do not omit one that is present. Match the reference data exactly, one-to-one.
    - 🚨 Watch for adjacent-row bleed in the reference data itself: on dense matrix tables, a room/surface can occasionally be misattributed to the wrong neighboring row (e.g. a mark that should belong to "Paint" instead showing up under "Exterior Board", or a row at the bottom of one group like "Sealed Concrete" (a FLOOR item) being mislabeled with the next group's name, "WALL"). If a row_label's assigned category doesn't semantically match what that material actually is (e.g. "Sealed Concrete" tagged as a wall material, or a ceiling material tagged as a floor material), trust the material's real-world nature over a mismatched group label from the reference data, and use the reference image itself to double check which row the mark truly belongs to before finalizing.
    - Anything related to Wall must be kept inside "wall" category regardless of the location of the room. IF ANY OTHER MATERIAL LIE IN THE WALL ROW OR COLUMN OF THE TABLE, AUTOMATICALLY SET THE CATEGORY TO WALL.

    Example:
    [{"row_label": "Ceramic Tile (Anti Slip)", "column_label": "Shower Room-Floor"},
     {"row_label": "Ceramic Tile (Anti Slip)", "column_label": "Family Restroom-Floor"},
     {"row_label": "Cer amic Tile", "column_label":"Wall"}
    ]

    Correct output (two separate objects, not one combined summary):
    {
        "name": "Ceramic Tile (Anti Slip)",
        "notes": "Floor. Trafficmaster, Baja Gray - Matte Finish 12\" x 12\" or approved equal.",
        "category": {
        "c1": "Room-Family Restroom",
        "c2": "Room-Shower Room",
        }
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Material and Finishes Schedule"}
        ]
    },
    {
        "name": "Ceramic Tile",
        "notes": "Wall. Trafficmaster, Baja Gray - Matte Finish 12\" x 12\" or approved equal.",
        "category":"Wall"   #You may use wall-interior or wall-Exterior if the pdf is follwing that approach.
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Material and Finishes Schedule"}
        ]
    }

    ### FINAL EXPECTED OUTPUT STRUCTURE

    The final output must be a single flat array containing unique material objects matching this exact JSON format:

    [
      {
        "name": "Asphalt Shingles",
        "notes": "Black asphalt shingles, referenced as exterior material no. 1",
        "category": "Roof",
        "mentions": [
          {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
          {"page_label": "Sheet 7 of 23 - West Elevation", "view": "Exterior Materials Schedule"},
          {"page_label": "Sheet 8 of 23 - Right Elevation", "view": "Exterior Materials Schedule"},
        ]
      },
      {
        "name": "Door-01A",
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
        "category": "Wall-Interior",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - East Elevation (Front)", "view": "East Elevation (Front) - Exterior Elevation View"},
        ]
      },
      {
        "code": "X-74",
        "notes": "HARDWOOD FLOOR, 2-3\" WIDE, FINISH WOOD, TONGUE & GROOVE, STAINED"
        "category": {
            "c1": "Room-MainRoom",
            "c2": "Room-Kitchen"
            "c3": "Room-Toilet"
        },
        "mentions": [
          {"page_label": "Sheet 6 of 23", "view": "Main Floor Plan Layout"},
          {"page_label": "Sheet 8 of 23", "view": "Kitchen Floor Plan Layout"},
        ]
    },
     {
        "name": "Plywood Subfloor",
        "notes": "3/4\" Plywood Subfloor. Material listed in F2 - Typical Floor Assembly.",
        "category": "Room-Typical",
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
        "category": {
                    "c1": "Wall-Exterior",
                    "c2": "Wall-Interior",
        },
        "mentions": [
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
        ]
      },
     {
        "name": "TYVEK HOUSE WRAP", 
        "notes": "Tyvek House Wrap, Material listed in W1",
        "category": "Wall-Exterior",
        "mentions": [
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
        ]
     },
     {
        "name": "3/8' OSB EXTERIOR SHEATHING", 
        "notes": "Material listed in W1, 3/8' thickness osb exterior sheathing",
        "category": "Wall-Exterior",
        "mentions": [
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"}
        ]
     },
     {
        "name": "2X6 STUDS @ 16' O.C.", 
        "notes": "Material listed in W1, 2X6 size STUDS and 16' o.c spacing is mentioned in the notes of W1",
        "category": "Wall-Exterior",
        "mentions": [             
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"},
        ]
     },
     {
        "name": "R-25 BATT INSULATION", 
        "notes": "Material listed in W1, R-25 insulation value and batt type is mentioned in the notes of W1"
        "category": "Wall-Exterior",
        "mentions": [ 
            {"page_label": "Sheet 16 of 23", "view": "Main Floor Plan Layout", "Extracted from code": "W1"},
       ]
     },
     {
        "name": "Ceramic Tile (Anti Slip)",
        "notes": "Ceramic Tile (Anti Slip), Floor. Trafficmaster, Baja Gray - Matte Finish 12\" x 12\" or approved equal.",
        "category": "Room-Shower Room",
        "mentions": [
            {"page_label": "C - 301 - Finishes, Fittings and Accessories Schedule", "view": "Material and Finishes Schedule"}
        ]
     },
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

            content_blocks = list(anchor_blocks) # schedule anchors go in every batch
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
                    "AFTER the materials array above, on its own line print exactly:\n" f"{SCALE_DELIMITER}\n" "Then, using the SAME pages you were just given, also perform this SECOND, separate task and output ITS result as its own JSON array (or the literal word NONE) immediately after " f"the delimiter line:\n\n{SCALE_PROMPT}"
                )
                content_blocks.append({"type": "text", "text": combined_prompt})

                text_chunks = []
                with client.beta.messages.stream(
                    model="claude-sonnet-5",
                    max_tokens=100000,
                    system=(
                        "You are a strict technical drawing extraction engine. You must output valid raw JSON data blocks only. Do not speak or include explanations, preamble, or trailing markdown wrappers. Start your response directly with '[' and end the " f"materials array with ']', then print the delimiter line '{SCALE_DELIMITER}', then the scale JSON array (or NONE)."
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
    results = enforce_wall_coding_consistency(results)
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