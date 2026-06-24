import os
import re
import asyncio
from datetime import datetime
from anthropic import AsyncAnthropic
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder
from tools.helpers import safe_parse_json, dense_model, sparse_model, COLLECTION

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", revision="main")

qdrant_client = QdrantClient(url="http://localhost:6333")
async_anthropic_client = AsyncAnthropic(api_key=os.getenv("CLAUDE_API_KEY"))

CSI_CODE_PATTERN = re.compile(r"^\d{2}\s\d{2}\s\d{2}(?:\.\d{2})?$")

REFERENCE_BOILERPLATE_PATTERN = re.compile(
    r"\b(material listed (?:in|under)|referenced in|extracted from code|see schedule(?: on)?)\b.*?(?=$|\.)",
    flags=re.IGNORECASE,
)

TOP_K_CHUNKS = 5


def strip_reference_boilerplate(text: str) -> str:
    if not text:
        return ""
    cleaned = REFERENCE_BOILERPLATE_PATTERN.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .-")
    return cleaned


def clean_material_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\b\d+(?:/\d+)?[\"\']?\s*x\s*\d+(?:/\d+)?[\"\']?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+(?:/\d+)?\s*(?:\"|\'|inch|inches|foot|feet|mm|cm|oz|\-gauge|\s*O\.C\.)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = text.replace('"', '').replace("'", "").strip()
    return text


def build_query_and_context(item: dict):
    raw_name = item.get("name") or item.get("code") or ""
    cleaned_name = clean_material_text(str(raw_name))

    notes = item.get("notes", "")
    notes = strip_reference_boilerplate(notes) if isinstance(notes, str) else ""
    cleaned_notes = clean_material_text(notes) if isinstance(notes, str) else ""

    if cleaned_name:
        query = f"{cleaned_name}. Description: {cleaned_notes}".strip()
    else:
        query = cleaned_notes or "construction material specification"

    category = item.get("category", "")
    location_context = category if (category and str(category).strip().lower() not in ("others", "")) else ""

    return query, location_context


def _get_content(payload: dict) -> str:
    if not payload:
        return ""
    return payload.get("content", "") or payload.get("chunk_text", "")


def rerank_chunks(query: str, docs: list, top_n: int = TOP_K_CHUNKS) -> list:
    if not docs:
        return []

    pairs = []
    valid_docs = []

    for d in docs:
        if hasattr(d, "payload") and d.payload is not None:
            content = _get_content(d.payload)
        elif isinstance(d, dict) and d.get("payload") is not None:
            content = _get_content(d["payload"])
        else:
            content = ""

        if content:
            pairs.append([query, content])
            valid_docs.append(d)

    if not pairs:
        print(f"[RERANKER] No content found in any of {len(docs)} docs for query '{query[:60]}'")
        return []

    scores = reranker.predict(pairs)
    # Flatten in case CrossEncoder returns [[s1], [s2], ...] instead of [s1, s2, ...]
    scores = [s[0] if hasattr(s, "__len__") else s for s in scores]
    ranked = sorted(zip(valid_docs, scores), key=lambda x: x[1], reverse=True)

    result_payloads = []
    for doc, _ in ranked[:top_n]:
        if hasattr(doc, "payload") and doc.payload is not None:
            result_payloads.append(doc.payload)
        elif isinstance(doc, dict) and doc.get("payload") is not None:
            result_payloads.append(doc["payload"])
        
    return result_payloads


def extract_codes_from_context(context_block: str) -> set:
    return set(re.findall(r"\d{2}\s\d{2}\s\d{2}(?:\.\d{2})?", context_block or ""))


def parent_code(code: str) -> str:
    """
      06 15 13.11  ->  06 15 13
    """
    if "." in code:
        return code.split(".")[0].strip()
    return code


async def classify_query_task(
    query: str,
    refs: list,
    context_block: str,
    location_context: str,
) -> None:
    location_line = (
        f"LOCATION ON BUILDING (context only - do NOT use this to pick the division; "
        f"it describes WHERE the item is, not WHAT it is): {location_context}\n"
        if location_context
        else ""
    )

    prompt = f"""You are an expert construction specification cost & CSI classification engine. Your task is to match a MATERIAL/PRODUCT description to its accurate MasterFormat 2018 classification code.

    🚨CRITICAL RULES:
        1. MATCH BY NAME AND TITLE: Look at the names of the Divisions and Titles in the context reference. Match the material name to the text name that describes what the product fundamentally is.
        2. MATERIAL OVER LOCATION: Classify based strictly on WHAT the product is fundamentally built as, NOT where it sits or what macro-structure it serves.
        3. SPECIFICITY OVER GENERALITY: If a code exists that explicitly covers the mechanism or form of the item, you MUST choose it over a broad umbrella code, provided a related section is present in the context. If a specific CSI division is not found in the context, assign the parent category code (e.g., ending in '00' like '06 16 00') rather than an overly granular micro-subdivision.
        4. FUNCTION OVER LOCATION: Words describing the SURFACE or APPLICATION (e.g. "floor", "roof") are functional. Use them to pick sibling sub-codes under the same parent.
        5. NAME IS THE PRIMARY SIGNAL: The "Item Type" shown below is the canonical identity of this material.

    TASK:
        - Classify based on the MATERIAL/PRODUCT itself, NOT based on where it is installed.
        - Return ONLY a single flat JSON object with exactly one key: "csi_division", value formatted as "XX XX XX" or "XX XX XX.XX".
        - If nothing in the context matches perfectly, use a valid parent level category code present in the text. If entirely unresolvable, return "00 00 00".

        MATERIAL/PRODUCT TO CLASSIFY:
        "{query}"
        {location_line}
        MASTERFORMAT SYSTEM CONTEXT REFERENCE (codes + titles retrieved for this material):
        {context_block}
    """

    assigned_code = "00 00 00"
    try:
        response = await async_anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            temperature=0.0,
            system=(
                "You are a strict technical automation engine. "
                "You must output a valid flat raw JSON object with no additional text or markdown decoration. "
                "Do not explain anything. Begin directly with your JSON payload structure."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
        result = safe_parse_json(raw_text)
        candidate = str(result.get("csi_division", "00 00 00")).strip()

        if CSI_CODE_PATTERN.match(candidate):
            assigned_code = candidate
        else:
            print(
                f"[CSI Classifier] REGEX MISMATCH for query '{query[:80]}': "
                f"got '{candidate}'. Falling back to 00 00 00."
            )
            assigned_code = "00 00 00"

        if assigned_code != "00 00 00":
            context_codes = extract_codes_from_context(context_block)
            if context_codes and assigned_code not in context_codes:
                # Only attempt dot-suffix strip (e.g. 06 15 13.11 -> 06 15 13)
                p = parent_code(assigned_code)
                if p != assigned_code and p in context_codes:
                    assigned_code = p
                else:
                    assigned_code = "00 00 00"

    except Exception as e:
        print(
            f"[CSI Classifier] EXCEPTION for query '{query[:80]}': "
            f"{type(e).__name__}: {e}."
        )
        assigned_code = "00 00 00"

    for ref in refs:
        ref["item"]["csi_division"] = str(assigned_code).strip()


def harmonize_consistent_codes(materials: list) -> list:
    groups: dict = {}
    for item in materials:
        if not isinstance(item, dict):
            continue
        key = str(item.get("name") or item.get("code") or "").strip().lower()
        if not key:
            continue
        groups.setdefault(key, []).append(item)

    for key, group in groups.items():
        codes = [g.get("csi_division", "00 00 00") for g in group]
        if len(set(codes)) <= 1:
            continue

        non_unclassified = [c for c in codes if c != "00 00 00"]
        if non_unclassified:
            counts: dict = {}
            for c in non_unclassified:
                counts[c] = counts.get(c, 0) + 1
            winner = max(counts.items(), key=lambda kv: kv[1])[0]
        else:
            winner = "00 00 00"

        for g in group:
            g["csi_division"] = winner

    return materials


def _write_debug_chunks(
    debug_file,
    query: str,
    contexts: list,
    run_label: str = "",
) -> None:
    header = f"Query: '{query}'"
    if run_label:
        header += f"  [{run_label}]"
    debug_file.write(header + "\n")

    if not contexts:
        debug_file.write("   No matching chunks found. LLM will receive no context.\n")
    else:
        for rank, chunk in enumerate(contexts, 1):
            cleaned = chunk.replace("\n", " ")
            debug_file.write(f"   [{rank}] {cleaned}\n")

    debug_file.write("-" * 60 + "\n\n")


async def csi_classifier_node_async(state: dict) -> dict:
    print("\n=== [Agent 2: CSI Classifier] ===")

    materials = state.get("extracted_materials", [])
    if not isinstance(materials, list):
        materials = [materials] if materials else []

    refs = []
    for item in materials:
        if not isinstance(item, dict):
            continue
        query, location_context = build_query_and_context(item)
        refs.append({"item": item, "search_query": query, "location_context": location_context})

    if not refs:
        return {"final_materials": materials, "final_specifications": materials}

    # Deduplicate queries (same text -> one retrieval call)
    query_to_refs: dict = {}
    query_to_location: dict = {}
    for r in refs:
        q = r["search_query"]
        query_to_refs.setdefault(q, []).append(r)
        query_to_location[q] = r["location_context"]

    unique_queries = list(query_to_refs.keys())
    query_contexts: dict = {}

    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = f"retrieved_chunks_debug_{run_label}.txt"

    try:
        dense_vectors = [dense_model.embed_query(q) for q in unique_queries]
        sparse_vectors = list(sparse_model.embed(unique_queries))

        batch_requests = []
        for idx in range(len(unique_queries)):
            s_vec = sparse_vectors[idx]
            batch_requests.append(
                models.QueryRequest(
                    prefetch=[
                        models.Prefetch(query=dense_vectors[idx], using="dense", limit=20),
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=s_vec.indices.tolist(),
                                values=s_vec.values.tolist(),
                            ),
                            using="sparse",
                            limit=20,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=15,
                    with_payload=True,  
                )
            )

        batch_results = qdrant_client.query_batch_points(
            collection_name=COLLECTION, requests=batch_requests
        )

        # Inspect first point to detect payload key ("content" vs "chunk_text")
        if batch_results and batch_results[0].points:
            sample_payload = batch_results[0].points[0].payload or {}
            detected_key = "content" if "content" in sample_payload else "chunk_text" if "chunk_text" in sample_payload else "unknown"
            print(f"[BATCH DEBUG] {len(batch_results[0].points)} points. Payload key detected: '{detected_key}'")

        with open(debug_path, "w", encoding="utf-8") as dbg:
            dbg.write(f"=== CSI Classifier Debug - Run: {run_label} ===\n\n")

            for uq, lookup_response in zip(unique_queries, batch_results):
                ranked_chunks = rerank_chunks(uq, lookup_response.points, top_n=TOP_K_CHUNKS)

                seen_contents: list = []
                for chunk in ranked_chunks:
                    if not isinstance(chunk, dict):
                        continue
                    
                    content = _get_content(chunk)
                    if content and content not in seen_contents:
                        seen_contents.append(content)

                query_contexts[uq] = "\n".join(seen_contents)
                _write_debug_chunks(dbg, uq, seen_contents, run_label=run_label)

        print(f"[CSI Classifier] Debug chunks saved -> {debug_path}")

    except Exception as e:
        print(f"[CSI Classifier] Batch vector search pipeline failed: {e}. Falling back to no-context.")
        for uq in unique_queries:
            query_contexts[uq] = "No context available from MasterFormat Database."

        try:
            with open(debug_path, "w", encoding="utf-8") as dbg:
                dbg.write(f"=== CSI Classifier Debug - Run: {run_label} (RETRIEVAL FAILED) ===\n\n")
                dbg.write(f"Error: {type(e).__name__}: {e}\n\n")
                for uq in unique_queries:
                    _write_debug_chunks(dbg, uq, [], run_label=run_label)
        except Exception:
            pass

    tasks = [
        classify_query_task(
            q,
            r,
            query_contexts.get(q, ""),
            query_to_location.get(q, ""),
        )
        for q, r in query_to_refs.items()
    ]
    await asyncio.gather(*tasks)

    materials = harmonize_consistent_codes(materials)

    return {"final_materials": materials, "final_specifications": materials}


def csi_classifier_node(state: dict) -> dict:
    return asyncio.run(csi_classifier_node_async(state))