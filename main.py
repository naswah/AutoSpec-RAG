from ingestion import run_ingestion
from vector_db import hybrid_search
from rag import rerank, build_prompt, generate_response
import json
import pytesseract
from qdrant_client import QdrantClient
import os
import os


def get_schema_chunk(chunks):
    for c in chunks:
        if c.get("code") == "OUTPUT_SCHEMA":
            return c
    return None


def build_query_from_materials(materials_json):
    if not materials_json or not isinstance(materials_json, dict):
        return ""

    # Access the 'materials_found' list directly from the dictionary
    items = materials_json.get("materials_found", [])
    # Also grab 'notes_found' to increase search context
    notes = materials_json.get("notes_found", [])

    query_parts = []

    for m in items:
        if m.get("item"):
            query_parts.append(m["item"])
        if m.get("specs"):
            query_parts.append(m["specs"])
            
    for n in notes:
        if n.get("message"):
            query_parts.append(n["message"])

    return " ".join(query_parts).strip()[:1000]



def load_schema_chunk():
    return {
        "code": "OUTPUT_SCHEMA",
        "content": """
You are a construction specification extraction assistant.

OUTPUT MUST FOLLOW THIS SCHEMA:

{
  "pages": [
    {
      "pg_no": integer,
      "views": [
        {
          "required_info": {
            "csi_division": "string (Format: XX XX XX, e.g., '03 30 00')",
            "description": "string",
            "notes": "string (Include dimensions, sizes, and PSI here)"
          }
        }
      ]
    }
  ]
}
"""
    }


def run_pipeline(pdf_path):

    print("\🚀 Running inference pipeline...")
    
    # 1. INGESTION
    ingestion_output = run_ingestion(
        pdf_path=pdf_path,
        output_base="data", 
        model_path=r"D:\AutoSpec RAG\best.pt"
        model_path=r"D:\AutoSpec RAG\best.pt"
    )

    ocr_data = ingestion_output["ocr_data"]
    materials = ingestion_output["materials"]

    if not materials:
        raise Exception("No materials extracted from Gemini")

    # 2. QUERY BUILDING
    query = build_query_from_materials(materials)

    print("\nQuery:")
    print(query)

    client = QdrantClient(url="http://localhost:6333")
    retrieved = hybrid_search(client, query)

    ranked = rerank(query, retrieved)

    schema_chunk = load_schema_chunk()

    prompt = build_prompt(
        materials,
        ocr_data,
        ranked,
        schema_chunk
    )

    result = generate_response(prompt)
    plan_name = os.path.splitext(os.path.basename(pdf_path))[0]
    results_folder = os.path.join(os.getcwd(), "Results")
                                  
    os.makedirs(results_folder, exist_ok=True)

    save_path = os.path.join(results_folder, f"{plan_name}.json")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Final Spec saved to: {save_path}")
    return result

if __name__ == "__main__":

    print("Starting RAG Inference Pipeline...")
    pdf_path=r"D:\AutoSpec RAG\Example Plans\CR-574_HousePlans.pdf"

    output = run_pipeline(
        pdf_path=r"D:\AutoSpec RAG\Example Plans\GAMEDAY COMPILED FINAL_12302024_SIGNED & SEALED_FLAT.pdf"
    )

    print("\nFINAL RESULT:")
    print(json.dumps(output, indent=2))

    output_dir = "Results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file_path = os.path.join(output_dir, f"{file_base_name}.json")

    with open(output_file_path, "w") as f:
        json.dump(output, f, indent=2)