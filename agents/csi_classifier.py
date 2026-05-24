import os
import re
import json
from anthropic import Anthropic
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from state.graph_state import AgenticState
from tools.helpers import hybrid_search, safe_parse_json

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
qdrant_client = QdrantClient(url="http://localhost:6333")

def extract_keywords_from_material(mat_info):
    keywords = []
    
    if isinstance(mat_info, str):
        keywords.append(mat_info)
        
    elif isinstance(mat_info, dict):
        if "name" in mat_info and mat_info["name"]:
            keywords.append(str(mat_info["name"]))
        
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
    for r in ranked[:3]:
        doc = r[0]
        if hasattr(doc, 'payload') and doc.payload:
            cleaned_chunks.append(doc.payload)
        else:
            cleaned_chunks.append(doc)
            
    return cleaned_chunks

def csi_classifier_node(state: AgenticState):
    print("\n=== [Agent 3: CSI Classifier]MasterFormat Classifications ===")
    client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    
    target_materials = state.get("mapped_materials")
    if not target_materials:
        target_materials = state.get("extracted_materials", [])
        
    full_retrieved_contexts = []
    
    for page in target_materials:
        for view in page.get("views", []):
            materials = view.get("materials", {})
            for mat_name, mat_info in materials.items():
                
                search_query = extract_keywords_from_material(mat_info)
                if not search_query:
                    continue
                
                initial_chunks = hybrid_search(qdrant_client, search_query, top_k=3)
                ranked_chunks = rerank_chunks(search_query, initial_chunks)
                
                item_chunks = []
                for chunk in ranked_chunks:
                    if isinstance(chunk, dict) and "content" in chunk:
                        item_chunks.append(chunk["content"])
                        full_retrieved_contexts.append(chunk["content"])
                
                if not item_chunks:
                    continue
                
                item_context = "\n\n".join(item_chunks)
                item_json_string = json.dumps({mat_name: mat_info}, indent=2)
                
                feedback = ""
                if state.get("retry_count", 0) > 0 and state.get("error_log"):
                    feedback = f"\nCRITICAL CORRECTIONS REQUIRED FROM PREVIOUS ATTEMPT:\n" + "\n".join(state["error_log"])

                prompt = f"""You are an expert construction specification cost & CSI classification engine.
TASK:
- Analyze the single material detailed below and select its matching 6-digit MasterFormat classification from the context records.
- Do not remove anything, just add the key "csi_division" fro every material with their respective divisions.
- Focus on material specific details: if it is ceramic tile, select the exact specific code (e.g., '09 30 13') rather than general level-3 parent headings (like '09 30 00').
- Return a JSON object containing exactly the mapped code assigned to a "csi_division" property field matching the template pattern 'XX XX XX'.

{feedback}

MATERIAL IDENTIFIER DETAILS:
{item_json_string}

MASTERFORMAT SYSTEM CONTEXT:
{item_context}"""

                try:
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=150,
                        temperature=0.0, 
                        system="You are a strict technical automation engine. You must output valid raw JSON data blocks only. Do not speak or include explanations. Begin directly with your JSON payload.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    raw_text = response.content[0].text.strip()
                    result_json = safe_parse_json(raw_text)
                    
                    if isinstance(mat_info, dict):
                        mat_info["csi_division"] = result_json.get("csi_division", "00 00 00").strip()
                        
                except Exception as e:
                    print(f"[CSI] Unexpected response for '{mat_name}': {e}")
                    if isinstance(mat_info, dict):
                        mat_info["csi_division"] = "00 00 00"

    unique_contexts = list(set(full_retrieved_contexts))
    final_context_log = "\n\n".join(unique_contexts)
    
    return {
        "final_specifications": target_materials, 
        "retrieved_context": final_context_log
    }