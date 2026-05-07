from ingestion import run_ingestion
from vector_db import hybrid_search
from rag import rerank, build_prompt, generate_response
import json
import pytesseract
from qdrant_client import QdrantClient


def get_schema_chunk(chunks):
    for c in chunks:
        if c.get("code") == "OUTPUT_SCHEMA":
            return c
    return None


def build_query_from_materials(materials_json):

    if isinstance(materials_json, list):
        if not materials_json:
            return ""
        data = materials_json[0]
    else:
        data= materials_json

    items = data.get("materials_found", [])

    query_parts = []

    for m in items:
        if m.get("item"):
            query_parts.append(m["item"])
        if m.get("specs"):
            query_parts.append(m["specs"])

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

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # 1. INGESTION
    ingestion_output = run_ingestion(
        pdf_path=pdf_path,
        output_base="data", 
        model_path=r"D:\rag-project(cost estimator)\best.pt"
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

    return result

if __name__ == "__main__":

    print("Starting RAG Inference Pipeline...")

    output = run_pipeline(
        pdf_path=r"DD:\rag-project(cost estimator)\Example Plans\CR-574_HousePlans.pdf"
    )

    print("\nFINAL RESULT:")
    print(json.dumps(output, indent=2))