import re
import uuid
import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, SparseVectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_KEY= os.getenv("GEMINI_API_KEY")
TARGET_DIMENSIONS= 768
COLLECTION = 'Master_Format_csv'

dense_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_KEY,
    output_dimensionality=TARGET_DIMENSIONS 
)
sparse_model = SparseTextEmbedding('Qdrant/bm25')

def safe_parse_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(cleaned)

def create_collection(client):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=TARGET_DIMENSIONS, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    print(f"Collection {COLLECTION} verified.")


def build_vectordb(client, chunks):

    texts = [c["content"] for c in chunks]
    print(f"Encoding and indexing {len(texts)} chunks...")
    
    dense_vectors = dense_model.embed_documents(texts)
    sparse_vectors = list(sparse_model.embed(texts))

    points = []

    for idx, (c, d, s) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        payload = {
            "content": c["content"],
            **c["metadata"]
        }
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": d,
                "sparse": {
                    "indices": s.indices.tolist(),
                    "values": s.values.tolist()
                }
            },
            payload=payload
        )
        points.append(point)

    batch_size = 50 
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION, points=batch)
        print(f"Successfully uploaded batch {i // batch_size + 1}")


def hybrid_search(client, query, top_k=5):
    d = dense_model.embed_query(query)
    
    s = list(sparse_model.embed([query]))[0]

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=d, using="dense", limit=top_k),
            models.Prefetch(
                query=models.SparseVector(
                    indices=s.indices.tolist(),
                    values=s.values.tolist()
                ),
                using="sparse",
                limit=top_k
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    )
    return [r.payload for r in results.points]