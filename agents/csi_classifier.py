import os
import re
import json
from anthropic import Anthropic
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder
from state.graph_state import AgenticState
from tools.helpers import safe_parse_json, dense_model, sparse_model, COLLECTION

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant_client = QdrantClient(url="http://localhost:6333")

def extract_keywords_from_material(mat_info):
    keywords = []
    if isinstance(mat_info, str):
        keywords.append(mat_info)
    elif isinstance(mat_info, dict):
        for flat_key in ["name", "code", "notes"]:
            val = mat_info.get(flat_key, "")
            if val and str(val).strip() not in ["-", "", "none", "Mapping Required"]:
                keywords.append(str(val).strip())
                
    return " | ".join(keywords) if keywords else ""


def rerank_chunks(query, docs):
    if not docs: 
        return []
    
    pairs = []
    valid_docs = []
    
    for d in docs:
        payload = None
        if hasattr(d, 'payload') and d.payload is not None:
            payload = d.payload
        elif isinstance(d, dict) and d.get('payload') is not None:
            payload = d['payload']
            
        content = payload.get("content", "") if payload else ""
        if content:
            pairs.append([query, content])
            valid_docs.append(d)
            
    if not pairs:
        return []

    scores = reranker.predict(pairs)
    ranked = sorted(zip(valid_docs, scores), key=lambda x: x[1], reverse=True)
    
    cleaned_chunks = []
    for r in ranked[:3]:
        doc = r[0]
        if hasattr(doc, 'payload') and doc.payload:
            cleaned_chunks.append(doc.payload)
        elif isinstance(doc, dict) and doc.get('payload'):
            cleaned_chunks.append(doc['payload'])
        else:
            cleaned_chunks.append(doc)
            
    return cleaned_chunks


def csi_classifier_node(state: AgenticState):
    print("\n=== [Agent 3: CSI Classifier] MasterFormat Classifications (TRUE BATCH) ===")
    client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    
    target_materials = state.get("mapped_materials")
    if not target_materials:
        target_materials = state.get("extracted_materials", [])
        
    material_refs = []  
    full_retrieved_contexts = []
    
    # Track materials in both list formats and layout dictionaries
    for page in target_materials:
        for view in page.get("views", []):
            materials = view.get("materials", [])
            
            if isinstance(materials, list):
                for mat_info in materials:
                    if isinstance(mat_info, dict):
                        search_query = extract_keywords_from_material(mat_info)
                        if search_query:
                            material_refs.append({"search_query": search_query, "mat_dict_ref": mat_info})
            
            elif isinstance(materials, dict):
                for mat_key, mat_info in materials.items():
                    if isinstance(mat_info, dict):
                        search_query = extract_keywords_from_material(mat_info)
                        if search_query:
                            material_refs.append({"search_query": search_query, "mat_dict_ref": mat_info})

    if not material_refs:
        print("No dynamic materials discovered requiring classification.")
        return {"final_specifications": target_materials, "retrieved_context": ""}

    unique_queries = list(set([ref["search_query"] for ref in material_refs]))
    print(f"Aggregated {len(material_refs)} items ({len(unique_queries)} unique queries) for single-batch vector retrieval...")

    # --- TRUE BATCH VECTOR DB LOOKUP (STAYS ATOMIC) ---
    if unique_queries:
        try:
            print("Executing bulk asynchronous vector retrieval over Qdrant...")
            dense_vectors = dense_model.encode(unique_queries, normalize_embeddings=True).tolist()
            sparse_vectors = list(sparse_model.embed(unique_queries))
            
            requests = []
            for idx, query in enumerate(unique_queries):
                s_vec = sparse_vectors[idx]
                
                req = models.QueryRequest(
                    prefetch=[
                        models.Prefetch(query=dense_vectors[idx], using="dense", limit=2),
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=s_vec.indices.tolist(),
                                values=s_vec.values.tolist()
                            ),
                            using="sparse",
                            limit=2
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=2
                )
                requests.append(req)
            
            batch_results = qdrant_client.query_batch_points(collection_name=COLLECTION, requests=requests)
            
            for query, lookup_response in zip(unique_queries, batch_results):
                initial_chunks = lookup_response.points
                ranked_chunks = rerank_chunks(query, initial_chunks)
                
                for chunk in ranked_chunks:
                    if isinstance(chunk, dict) and "content" in chunk:
                        full_retrieved_contexts.append(chunk["content"])
                        
        except Exception as e:
            print(f"Batch Vector search pipeline failed: {e}. Falling back safely to contextual matching.")

    unique_contexts = list(set(full_retrieved_contexts))
    item_context_block = "\n\n".join(unique_contexts) if unique_contexts else "No context available from MasterFormat Database."
    
    feedback = ""
    if state.get("retry_count", 0) > 0 and state.get("error_log"):
        feedback = f"\nCRITICAL CORRECTIONS REQUIRED FROM PREVIOUS ATTEMPT:\n" + "\n".join(state["error_log"])

    # Build tracking map for the full collection
    llm_materials_payload = {}
    for index, ref in enumerate(material_refs):
        item_id = f"item_{index + 1}"
        llm_materials_payload[item_id] = ref["search_query"]

    classification_map = {}
    items_list = list(llm_materials_payload.items())
    sub_batch_size = 100  
    
    print(f"Dividing {len(items_list)} items into {((len(items_list)-1)//sub_batch_size)+1} safe sub-batches for LLM evaluation...")

    for i in range(0, len(items_list), sub_batch_size):
        chunk = items_list[i : i + sub_batch_size]
        chunk_payload = dict(chunk)
        
        prompt = f"""You are an expert construction specification cost & CSI classification engine.
TASK:
- Analyze the chunked slice of materials detailed below identified by their structural item IDs.
- Assign each individual entry its matching 6-digit MasterFormat classification from the provided context records.
- Focus on material specific details: if it is ceramic tile, select the exact specific code (e.g., '09 30 13') rather than general headings.
- Return a single flat JSON object where each key matches the structural item ID (e.g., "item_1"), and its value is strictly its assigned "csi_division" code matching the template pattern 'XX XX XX'.

{feedback}

MATERIALS SUB-COLLECTION TO CLASSIFY ({i + 1} to {min(i + sub_batch_size, len(items_list))}):
{json.dumps(chunk_payload, indent=2)}

MASTERFORMAT SYSTEM CONTEXT:
{item_context_block}"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000, 
                temperature=0.0, 
                system="You are a strict technical automation engine. You must output a valid flat raw JSON object with no additional text or markdown decoration. Do not explain anything. Begin directly with your JSON payload structure.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_text = response.content[0].text.strip()
            chunk_map = safe_parse_json(raw_text)
            classification_map.update(chunk_map)
            
        except Exception as chunk_err:
            print(f"[CSI Chunk Error] Sub-batch window {i//sub_batch_size + 1} failed processing: {chunk_err}")
            for k in chunk_payload.keys():
                classification_map[k] = "00 00 00"

    mapped_count = 0
    for index, ref in enumerate(material_refs):
        item_id = f"item_{index + 1}"
        mat_info = ref["mat_dict_ref"]
        
        if item_id in classification_map:
            mat_info["csi_division"] = str(classification_map[item_id]).strip()
            mapped_count += 1
        else:
            mat_info["csi_division"] = "00 00 00"
    
    print(f"Successfully mapped {mapped_count}/{len(material_refs)} elements into specifications.")

    final_context_log = "\n\n".join(unique_contexts)
    return {
        "final_specifications": target_materials, 
        "retrieved_context": final_context_log
    }