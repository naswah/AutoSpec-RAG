from ingestion import run_ingestion
from vector_db import hybrid_search
from rag import rerank, build_prompt, generate_response, mappings
import json
from qdrant_client import QdrantClient
import os
import re
from groq import Groq
from dotenv import load_dotenv

qdrant_client = QdrantClient(url="http://localhost:6333")
load_dotenv(override=True)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_schema_chunk(chunks):
    for c in chunks:
        if c.get("code") == "OUTPUT_SCHEMA":
            return c
    return None


def build_query_from_materials(materials_list):
    query_parts = []

    for page in materials_list:
        for view in page.get("views", []):
            materials = view.get("materials", {})
            for mat_name, mat_info in materials.items():
                query_parts.append(mat_name)
                if isinstance(mat_info, str):
                    query_parts.append(mat_info)
    
    return " ".join(query_parts).strip()[:1000]


def run_pipeline(pdf_path, output_base):

    print(f"\n--- Starting Ingestion for: {os.path.basename(pdf_path)} ---")
    ingestion_results = run_ingestion(pdf_path, output_base)
    
    plan_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = ingestion_results["json_path"]
    
    with open(json_path, 'r', encoding="utf-8") as f:
        materials_list = json.load(f)

    mapped_materials = mappings(materials_list)
    
    json_string = json.dumps(mapped_materials, indent=2)

    search_query = build_query_from_materials(mapped_materials)
    print(f"--- Searching MasterFormat for: {search_query[:100]}... ---")
    
    initial_chunks = hybrid_search(qdrant_client, search_query)
    
    ranked_chunks = rerank(search_query, initial_chunks)

    final_prompt = build_prompt(json_string, ranked_chunks)
    
    final_result = generate_response(final_prompt)

    save_path = os.path.join(os.getcwd(), "Results", f"{plan_name}_Final.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    print(f"✅ Final Specification saved to: {save_path}")
    return final_result

if __name__ == "__main__":
    pdf_path = r"D:\AutoSpec RAG\Example Plans\American Farmhouse 201225 full.pdf"
    output_base = r"D:\AutoSpec RAG\output"
    run_pipeline(pdf_path, output_base)