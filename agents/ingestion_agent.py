import os
import json
import re
import time
from anthropic import Anthropic
from state.graph_state import AgenticState
from tools.pdf_helpers import pdf_to_image
from config import SCALE_PATH


SCALE_PROMPT = """
You are a professional construction drawing reviewer. Look at the drawing pages
provided and identify the SCALE of every FLOOR PLAN and ELEVATION view only.
Ignore schedules, notes pages, cover sheets, and detail-only pages unless a floor
plan or elevation view also appears on that page.

WHAT TO LOOK FOR:
- A printed scale note near the title block or under a specific view, e.g.
  '1/4" = 1\'-0"', '1:100', '3/16" = 1\'-0"', 'SCALE: AS NOTED', 'NTS' (Not to Scale).
- A graphic scale bar (a small ruler-like bar with tick marks and distance labels).
- Some sheets show a DIFFERENT scale per view on the same sheet; report each
  view's scale separately.

For each floor plan / elevation view found, return one JSON object with EXACTLY
these keys:

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

If NO floor plan or elevation view appears anywhere in the pages given, output
exactly: NONE

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
    
    ❗❗❗EXTRACT ONLY CIVIL ENGINEERING MATERIALS, LEAVE OTHERS❗❗❗

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

    ⚠️ If there are no materials in a view/page then you can skip the view or page. Try to fill the 'notes' section with sizes and specifications if possible. If there is no information for notes then you can leave it empty. 
   
    🚨CRITICAL🚨: LEGEND EXCLUSION RULE:
    - IF YOU SEE STANDALONE KEY NOTES, GENERAL LEGENDS, OR INDEX KEYS LOCATED ON THE SAME PAGE AS PLAN VIEWS, YOU MUST IGNORE THEM. Do not parse these static master legends as views or schedules.
    - If there are no marks, then do not add them. It is not compulsory.

    ### EXTRACTION & DE-DUPLICATION RULES
    To avoid data duplication, your output must group multiple page or view references of the exact same material together inside a "mentions" array.

    - **Unique Material Criteria**: A material item is considered identical if it shares the exact same `name` (or `code`), `category`, and `notes`. 
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
    If a table occurs with numbers in theor 1st column, then put the 'name' key as the notation/number/name of the column 1. RThe rest information could be aaded to the 'notes' section. The 'category' key must be added in accordance with the title of the table and the 'mentions' key must have the page and the view where the table is loacated and where the codes are present in the user plan.

    EXAMPLE 1:
    NO  |Qty |Width |Height |Matrial Finish |Glazing
    ----|----|------|-------|---------------|--------
    01A | 1  | 5'-0 | 6'-8' | Fibreglass    | -

    If the above is a door schdule then name should be Door- 01A
    Then the JSON must look like:
    {
        "name": "Door- 01A",
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
        "name": "Window- GC",
        "notes": "Qty: 1, Width: 5'-0, Height: 6'-8', Volume: -",
        "category": "Window Schedule",
        "mentions": [
            {"page_label": "Sheet 4 of 23 - Schedules", "view": "Window Schedule"},
        ]
    }

    If it isn't a schedule, just a normal table then the name must be the value of first column and the rest of the columns must be added to the notes section. The category must be added in accordance with the title of the table, notes or the mentions key must have the page and the view where the table is loacated and where the codes are present in the user plan.

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
      if W1 has listed VINYL SIDING, TYVEK HOUSE WRAP, 3/8' OSB EXTERIOR SHEATHING, 2X6 STUDS @ 16' O.C., R-25 BATT INSULATION then:
    
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
   ]

    Return ONLY VALID JSON.
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

                content_blocks.append({"type": "text", "text": prompt})

                text_chunks = []
                with client.beta.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=100000,
                    temperature=0,
                    system="You are a strict technical drawing extraction engine. You must output valid raw JSON data blocks only. Do not speak or include explanations, preamble, or trailing markdown wrappers. Start your response directly with '[' and end with ']'.",
                    messages=[{"role": "user", "content": content_blocks}],
                    betas=["files-api-2025-04-14"],
                ) as stream:
                    for text in stream.text_stream:
                        text_chunks.append(text)

                raw_text = "".join(text_chunks).strip()

                if not raw_text:
                    print(f"[Batch {batch_no}] WARNING: Empty response received from API. Skipping.")
                    continue

                if not raw_text.startswith(("[", "{")):
                    print(f"[Batch {batch_no}] WARNING: Unexpected non-JSON response. Preview:\n{raw_text[:500]}\n")
                    continue

                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text).strip()

                data = json.loads(raw_text)

                if isinstance(data, dict) and "views" in data:
                    results.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            results.append(item)

                # Reuse the SAME uploaded images for scale extraction, but with a different prompt 
                
                try:
                    scale_content_blocks = content_blocks[:-1]  # drop materials prompt text
                    scale_content_blocks.append({"type": "text", "text": SCALE_PROMPT})

                    scale_text_chunks = []
                    with client.beta.messages.stream(
                        model="claude-sonnet-4-6",
                        max_tokens=5000,
                        temperature=0,
                        system="You are a strict technical drawing extraction engine. Output valid raw JSON only. Do not speak or include explanations, preamble, or markdown fences. Start your response directly with '[' and end with ']'.",
                        messages=[{"role": "user", "content": scale_content_blocks}],
                        betas=["files-api-2025-04-14"],
                    ) as scale_stream:
                        for text in scale_stream.text_stream:
                            scale_text_chunks.append(text)

                    scale_raw = "".join(scale_text_chunks).strip()

                    if scale_raw:
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
                            print(f"[Batch {batch_no}] Scale raw response was:\n{scale_raw[:500]}\n")
                except Exception as e:
                    print(f"[Batch {batch_no}] Scale extraction error (non-fatal): {e}")

            except json.JSONDecodeError as e:
                print(f"[Batch {batch_no}] JSON parse error: {e}")
                salvaged = salvage_truncated_array(raw_text)
                if salvaged:
                    print(f"[Batch {batch_no}] Salvaged {len(salvaged)} complete material object(s) from the truncated response.")
                    results.extend(salvaged)
                else:
                    print(f"[Batch {batch_no}] Raw response was:\n{raw_text[:500]}\n")
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