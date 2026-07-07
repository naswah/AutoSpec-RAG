import os
import re
import asyncio
import time
from datetime import datetime
from anthropic import AsyncAnthropic
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder
from tools.helpers import safe_parse_json, dense_model, sparse_model, COLLECTION
from config import CHUNKS_PATH, PDF_PATH
import torch

try:
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 8)
except ImportError:
    device = "cpu"

print(f"[Reranker] Loaded on device: {device}")   
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", revision="main", device=device)
# reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

qdrant_client = QdrantClient(url="http://localhost:6333")
async_anthropic_client = AsyncAnthropic(api_key=os.getenv("CLAUDE_API_KEY"))

CSI_CODE_PATTERN = re.compile(r"^\d{2}\s\d{2}\s\d{2}(?:\.\d{2})?$")

REFERENCE_BOILERPLATE_PATTERN = re.compile(
    r"\b(material listed (?:in|under)|referenced in|extracted from code|see schedule(?: on)?)\b.*?(?=$|\.)",
    flags=re.IGNORECASE,
)

TOP_K_CHUNKS = 3

# Rate limiting semaphore to avoid API congestion
MAX_CONCURRENT_LLM_CALLS = 8
llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)


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


def extract_codes_from_context(context_block: str) -> set:
    return set(re.findall(r"\d{2}\s\d{2}\s\d{2}(?:\.\d{2})?", context_block or ""))


def parent_code(code: str) -> str:
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
    
    async with llm_semaphore:
        try:
            response = await async_anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
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


def _write_debug_chunks(debug_file, query: str, contexts: list, run_label: str = "") -> None:
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
    total_start = time.time()

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

    query_to_refs: dict = {}
    query_to_location: dict = {}
    for r in refs:
        q = r["search_query"]
        query_to_refs.setdefault(q, []).append(r)
        query_to_location[q] = r["location_context"]

    unique_queries = list(query_to_refs.keys())
    query_contexts: dict = {}

    raw_pdf_name = state.get("pdf_name") or PDF_PATH
    if raw_pdf_name and isinstance(raw_pdf_name, (str, bytes, os.PathLike)):
        pdf_base_name = os.path.splitext(os.path.basename(raw_pdf_name))[0]
    else:
        pdf_base_name = "unknown_document"
    
    date_today = datetime.now().strftime("%Y%m%d")
    os.makedirs(CHUNKS_PATH, exist_ok=True)

    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = os.path.join(CHUNKS_PATH, f"{pdf_base_name}_{date_today}.txt")

    try:
        embed_start = time.time()
        if hasattr(dense_model, "embed_documents"):
            dense_vectors = dense_model.embed_documents(unique_queries)
        else:
            dense_vectors = [dense_model.embed_query(q) for q in unique_queries]
            
        sparse_vectors = list(sparse_model.embed(unique_queries))
        print(f"[PERF] Vector Embedding Generation Time: {time.time() - embed_start:.4f} seconds")

        db_start = time.time()
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
        print(f"[PERF] Qdrant Batch Retrieval Time: {time.time() - db_start:.4f} seconds")

        if batch_results and batch_results[0].points:
            sample_payload = batch_results[0].points[0].payload or {}
            detected_key = "content" if "content" in sample_payload else "chunk_text" if "chunk_text" in sample_payload else "unknown"
            print(f"[BATCH DEBUG] {len(batch_results[0].points)} points. Payload key: '{detected_key}'")

        rerank_start = time.time()
        
        all_pairs = []
        pair_to_source = []

        # LATENCY OPTIMIZATION: Only evaluate the top 5 high-signal candidates per queryRRF (Reciprocal Rank Fusion) has already consolidated Dense + Sparse signals, so candidates beyond index 5 have diminishing returns and massive CPU costs.
        PRE_RERANK_LIMIT = 5

        for q_idx, (uq, lookup_response) in enumerate(zip(unique_queries, batch_results)):
            for d in lookup_response.points[:PRE_RERANK_LIMIT]:
                if hasattr(d, "payload") and d.payload is not None:
                    content = _get_content(d.payload)
                elif isinstance(d, dict) and d.get("payload") is not None:
                    content = _get_content(d["payload"])
                else:
                    content = ""

                if content:
                    all_pairs.append([uq, content])
                    pair_to_source.append((q_idx, d))

        if all_pairs:
            scores = reranker.predict(all_pairs, batch_size=64, show_progress_bar=False)
            scores = [s[0] if hasattr(s, "__len__") else s for s in scores]
        else:
            scores = []

        query_to_scored_points = {i: [] for i in range(len(unique_queries))}
        for score, (q_idx, doc) in zip(scores, pair_to_source):
            query_to_scored_points[q_idx].append((doc, score))

        with open(debug_path, "w", encoding="utf-8") as dbg:
            dbg.write(f"=== CSI Classifier Debug - Run: {run_label} ===\n\n")

            for q_idx, uq in enumerate(unique_queries):
                scored_points = query_to_scored_points[q_idx]
                ranked = sorted(scored_points, key=lambda x: x[1], reverse=True)

                result_payloads = []
                for doc, _ in ranked[:TOP_K_CHUNKS]:
                    if hasattr(doc, "payload") and doc.payload is not None:
                        result_payloads.append(doc.payload)
                    elif isinstance(doc, dict) and doc.get("payload") is not None:
                        result_payloads.append(doc["payload"])

                seen_contents: list = []
                for chunk in result_payloads:
                    content = _get_content(chunk)
                    if content and content not in seen_contents:
                        seen_contents.append(content)

                query_contexts[uq] = "\n".join(seen_contents)
                _write_debug_chunks(dbg, uq, seen_contents, run_label=run_label)

        print(f"[PERF] Reranking and I/O Time: {time.time() - rerank_start:.4f} seconds")

    except Exception as e:
        print(f"[CSI Classifier] Vector pipeline failed: {e}. Defaulting to no-context.")
        for uq in unique_queries:
            query_contexts[uq] = "No context available from MasterFormat Database."

    llm_start = time.time()
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
    print(f"[PERF] Total Parallel LLM Classification Time: {time.time() - llm_start:.4f} seconds")

    materials = harmonize_consistent_codes(materials)
    print(f"[PERF] Total Node Runtime: {time.time() - total_start:.4f} seconds\n")

    return {"final_materials": materials, "final_specifications": materials}


def csi_classifier_node(state: dict) -> dict:
    return asyncio.run(csi_classifier_node_async(state))