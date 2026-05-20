import os
import json
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from state.graph_state import AgenticState
from tools.helpers import hybrid_search

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant_client = QdrantClient(url="http://localhost:6333")


def extract_keywords_from_material(mat_info):
    """
    Extracts descriptive keywords for a single material item,
    properly handling Case 1 (Standard) and Case 2 (Mapped from schedules).
    """
    keywords = []
    
    if isinstance(mat_info, str):
        keywords.append(mat_info)
        
    elif isinstance(mat_info, dict):
        # Case 1: Standard Material - Extract the core 'name' field
        if "name" in mat_info and mat_info["name"]:
            keywords.append(str(mat_info["name"]))
        
        # Case 2: Mapped Material - Extract descriptive features from properties
        if "properties" in mat_info and isinstance(mat_info["properties"], dict):
            props = mat_info["properties"]
            for prop_key, prop_val in props.items():
                if prop_val and str(prop_val).strip() not in ["-", "", "none"]:
                    if prop_key.upper() != "MARK":
                        keywords.append(str(prop_val))
        
        notes = mat_info.get("notes", "")
        if notes and str(notes).lower().strip() not in ["none", "mapped from schedule"]:
            keywords.append(str(notes))
            
    return " ".join(keywords).strip()


def rerank_chunks(query, docs):
    if not docs: 
        return []
    pairs = [[query, d.payload.get("content", "")] if hasattr(d, 'payload') else [query, d.get("content", "")] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    cleaned_chunks = []
    for r in ranked[:5]:
        doc = r[0]
        if hasattr(doc, 'payload') and doc.payload:
            cleaned_chunks.append(doc.payload)
        else:
            cleaned_chunks.append(doc)
            
    return cleaned_chunks


def csi_classifier_node(state: AgenticState):
    print("\nMatching MasterFormat Subdivisions...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    target_materials = state.get("mapped_materials") or state.get("extracted_materials", [])
    
    all_retrieved_contexts = []
    
    for page in target_materials:
        for view in page.get("views", []):
            materials = view.get("materials", {})
            for mat_name, mat_info in materials.items():
                
                search_query = extract_keywords_from_material(mat_info)
                
                if not search_query:
                    continue
                
                initial_chunks = hybrid_search(qdrant_client, search_query, top_k=5)
                ranked_chunks = rerank_chunks(search_query, initial_chunks)
                
                for chunk in ranked_chunks:
                    if isinstance(chunk, dict) and "content" in chunk:
                        all_retrieved_contexts.append(chunk["content"])

    unique_contexts = list(set(all_retrieved_contexts))
    context = "\n\n".join(unique_contexts)

    json_string = json.dumps(target_materials, indent=2)
    
    feedback = ""
    if state.get("retry_count", 0) > 0 and state.get("error_log"):
        feedback = f"\nCRITICAL CORRECTIONS REQUIRED FROM PREVIOUS ATTEMPT:\n" + "\n".join(state["error_log"])

    prompt = f"""You are an expert construction cost & CSI classification system.
🚨CRITICAL: Do not change the JSON format. Jsut add "csi_division" at the end of each material. Everything else remains the same.
TASK:
- Take the provided JSON data. For every material item, find the matching 6-digit CSI code from the context. Provide CSI codes for all materials that are detected.
- Update the "csi_division" field in the JSON with that code. Add correct CSI division MasterFormat codes (pattern: XX XX XX or XX XX XX.XX e.g., '09 30 13') in the provided json.
- Return the EXACT same JSON structure with csi codes, fully populated.
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
    
    classified_data = json.loads(response.choices[0].message.content)
    
    return {
        "classified_materials": classified_data.get("extracted_materials", classified_data),
        "retrieved_context": context
    }