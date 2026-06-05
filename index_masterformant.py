import os
from qdrant_client import QdrantClient
from tools.pdf_helpers import clean_masterformat, chunk_masterformat_hierarchical
from tools.helpers import create_collection, build_vectordb

def run_indexing_setup():
    client = QdrantClient(url="http://localhost:6333")
    print("Seeding Qdrant Storage Collections...")
    
    pdf_source = r"D:\AutoSpec RAG\required_masterformat_2_2018.pdf"
    
    if not os.path.exists(pdf_source):
        print(f"Error: Could not locate baseline file at {pdf_source}")
        return

    text = clean_masterformat(pdf_source)
    chunks = chunk_masterformat_hierarchical(text)

    create_collection(client)
    
    print("Executing Batch Encoding & Hybrid Vector Upsert Pipeline:")
    build_vectordb(client, chunks)
    print("\n Success: MasterFormat has been completely split, contextualized, and indexed into Qdrant!")

if __name__ == "__main__":
    run_indexing_setup()