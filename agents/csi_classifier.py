import os
import re
import json
import asyncio
from pydantic import BaseModel
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder
from state.graph_state import AgenticState
from tools.helpers import safe_parse_json, dense_model, sparse_model, COLLECTION

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant_client = QdrantClient(url="http://localhost:6333")

gemini_client = genai.Client()

class CsiOutput(BaseModel):
    csi_division: str

def extract_keywords_from_material(mat_info):
    keywords = []
    if isinstance(mat_info, str):
        keywords.append(mat_info)
    elif isinstance(mat_info, dict):
        for flat_key in ["name", "code", "notes"]:
            val = mat_info.get(flat_key, "")
            if val and str(val).strip() not in ["-", "", "none", "Mapping Required"]:
                keywords.append(str(val).strip())
                
    text = " | ".join(keywords) if keywords else ""
    text = re.sub(r'\b\d{4,6}\b', '', text)
    text = re.sub(r'[\s\t\n]+', ' ', text).strip()
    
    return text


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


async def classify_query_task(query, refs, context_block, feedback):
    prompt = f"""You are an expert construction specification cost & CSI classification engine. You must only map the material to a CSI code that is explicitly listed in the provided reference text. If a code is not present in the document context, do not infer or use external MasterFormat knowledge.

TASK:
- Analyze the single material type specified below.
- If multiple codes are mentioned in the text (such as cross-references or "See Also" lines), pick the primary code that closest matches the material description itself. For example, if evaluating stair treads and the text says "05 55 00 Metal Stair Treads (See 05 51 00 for metal stairs)", choose "05 55 00"
- Assign it its matching MasterFormat csi code from the provided context records.
- Focus on material-specific details: if it is ceramic tile, select the exact specific code (e.g., '09 30 13') rather than general headings.
- Return a single flat JSON object with exactly one key: "csi_division". Its value must strictly be the code matching the template pattern 'XX XX XX'.
- Do NOT use external knowledge.

    MATERIAL TO CLASSIFY:
    "{query}"

    MASTERFORMAT SYSTEM CONTEXT:
    {context_block}

    {feedback}
    """
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    system_instruction='You are a strict technical automation engine. You must output a valid flat raw JSON object.',
                    response_mime_type="application/json",
                    response_schema=CsiOutput,
                ),
            )
        )
        
        raw_text = response.text.strip()
        result = safe_parse_json(raw_text)
        assigned_code = result.get("csi_division", "00 00 00")
    except Exception:
        assigned_code = "00 00 00"

    for ref in refs:
        ref["mat_dict_ref"]["csi_division"] = str(assigned_code).strip()


async def csi_classifier_node_async(state: AgenticState):
    print("\n=== [Agent 3: CSI Classifier] ===")
    
    target_materials = state.get("mapped_materials")
    if not target_materials:
        target_materials = state.get("extracted_materials", [])
        
    material_refs = []  
    
    for page in target_materials:
        for view in page.get("views", []):
            materials = view.get("materials", [])
            
            mat_list = materials if isinstance(materials, list) else materials.values() if isinstance(materials, dict) else []
            for mat_info in mat_list:
                if isinstance(mat_info, dict):
                    search_query = extract_keywords_from_material(mat_info)
                    if search_query:
                        material_refs.append({"search_query": search_query, "mat_dict_ref": mat_info})

    if not material_refs:
        print("No dynamic materials discovered requiring classification.")
        return {"final_specifications": target_materials, "retrieved_context": ""}

    query_to_refs = {}
    for ref in material_refs:
        query_to_refs.setdefault(ref["search_query"], []).append(ref)

    unique_queries = list(query_to_refs.keys())
    query_contexts = {}
    all_retrieved_contexts_log = []

    if unique_queries:
        try:
            dense_vectors = dense_model.encode(unique_queries, normalize_embeddings=True).tolist()
            sparse_vectors = list(sparse_model.embed(unique_queries))
            
            requests = []
            for idx, query in enumerate(unique_queries):
                s_vec = sparse_vectors[idx]
                
                req = models.QueryRequest(
                    prefetch=[
                        models.Prefetch(query=dense_vectors[idx], using="dense", limit=15),
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=s_vec.indices.tolist(),
                                values=s_vec.values.tolist()
                            ),
                            using="sparse",
                            limit=15
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=5
                )
                requests.append(req)
            
            batch_results = qdrant_client.query_batch_points(collection_name=COLLECTION, requests=requests)
            
            for query, lookup_response in zip(unique_queries, batch_results):
                initial_chunks = lookup_response.points
                ranked_chunks = rerank_chunks(query, initial_chunks)
                
                contexts = []
                for chunk in ranked_chunks:
                    if isinstance(chunk, dict) and "content" in chunk:
                        contexts.append(chunk["content"])
                        all_retrieved_contexts_log.append(chunk["content"])
                
                query_contexts[query] = "\n\n".join(set(contexts)) if contexts else "No context available from MasterFormat Database."
                        
        except Exception as e:
            print(f"Batch Vector search pipeline failed: {e}. Falling back safely to empty contexts.")
            for query in unique_queries:
                query_contexts[query] = "No context available from MasterFormat Database."

    feedback = ""
    if state.get("retry_count", 0) > 0 and state.get("error_log"):
        feedback = f"\nCRITICAL CORRECTIONS REQUIRED FROM PREVIOUS ATTEMPT:\n" + "\n".join(state["error_log"])

    tasks = []
    for query, refs in query_to_refs.items():
        context_block = query_contexts.get(query, "No context available from MasterFormat Database.")
        tasks.append(classify_query_task(query, refs, context_block, feedback))
    
    await asyncio.gather(*tasks)
    
    print(f"Successfully evaluated and mapped {len(material_refs)} elements across {len(unique_queries)} parallel async tasks.")

    final_context_log = "\n\n".join(list(set(all_retrieved_contexts_log)))
    return {
        "final_specifications": target_materials, 
        "retrieved_context": final_context_log
    }


def csi_classifier_node(state: AgenticState):
    return asyncio.run(csi_classifier_node_async(state))