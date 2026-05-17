import os
import json
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from state.graph_state import AgenticState
from tools.helpers import hybrid_search

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant_client = QdrantClient(url="http://localhost:6333")

def build_query_from_materials(materials_list):
    query_parts = []
    for page in materials_list:
        for view in page.get("views", []):
            materials = view.get("materials", {})
            for mat_name, mat_info in materials.items():
                query_parts.append(mat_name)
                if isinstance(mat_info, str):
                    query_parts.append(mat_info)
                elif isinstance(mat_info, dict):
                    query_parts.append(str(mat_info.get("notes", "")))
    return " ".join(query_parts).strip()[:1000]


def rerank_chunks(query, docs):
    if not docs: return []
    pairs = [[query, d.get("content", "")] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:5]]


def csi_classifier_node(state: AgenticState):
    print(f"\n=== [Agent 3: CSI Classifier] Computing MasterFormat Classifications ===")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    target_materials = state.get("mapped_materials")
    if not target_materials:
        target_materials = state.get("extracted_materials", [])

    if not state.get("retrieved_context"):
        search_query = build_query_from_materials(target_materials)
        initial_chunks = hybrid_search(qdrant_client, search_query)
        ranked_chunks = rerank_chunks(search_query, initial_chunks)
        context = "\n\n".join([c["content"] for c in ranked_chunks])
    else:
        context = state["retrieved_context"]

    json_string = json.dumps(target_materials, indent=2)
    
    feedback = ""
    if state.get("retry_count", 0) > 0 and state.get("error_log"):
        feedback = f"\nCRITICAL CORRECTIONS REQUIRED FROM PREVIOUS ATTEMPT:\n" + "\n".join(state["error_log"])

    prompt = f"""You are an expert construction cost & CSI classification system.
TASK:
- Take the provided JSON data. For every material item, find the matching 6-digit CSI code from the context. Provide CSI codes for all materials that are detected.
- Update the "csi_division" field in the JSON with that code. Add correct CSI division MasterFormat codes (pattern: XX XX XX or XX XX XX.XX e.g., '03 30 00') in the provided json.
- DO NOT change any other values (dimensions, notes, or names).
- Return the EXACT same JSON structure, fully populated.
{feedback}

JSON DATA:
{json_string}

MASTERFORMAT CONTEXT:
{context}"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    final_result = json.loads(response.choices[0].message.content)
    return {"final_specifications": final_result, "retrieved_context": context}