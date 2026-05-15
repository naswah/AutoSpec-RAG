from qdrant_client import QdrantClient
from processing import clean_masterformat, chunk_masterformat
from vector_db import create_collection, build_vectordb


client = QdrantClient(url="http://localhost:6333")

print("Building MasterFormat index...")

text = clean_masterformat(r"D:\AutoSpec RAG\required_masterformat_2_2018.pdf")
chunks = chunk_masterformat(text)

create_collection(client)
build_vectordb(client, chunks)

print("MasterFormat indexed successfully")