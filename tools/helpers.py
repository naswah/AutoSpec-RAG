import uuid
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, SparseVectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

dense_model = SentenceTransformer('all-MiniLM-L6-v2')
sparse_model = SparseTextEmbedding('Qdrant/bm25')
COLLECTION = 'MasterFormat'

def create_collection(client):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    print(f"Collection {COLLECTION} verified.")


def build_vectordb(client, chunks):
    texts = [c["content"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    dense_vectors = dense_model.encode(texts, normalize_embeddings=True)
    sparse_vectors = list(sparse_model.embed(texts))
    points = []

    for c, d, s in zip(chunks, dense_vectors, sparse_vectors):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": d.tolist(),
                "sparse": {
                    "indices": s.indices.tolist(),
                    "values": s.values.tolist()
                }
            },
            payload=c
        )
        points.append(point)

    batch_size = 50 
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION, points=batch)


def hybrid_search(client, query, top_k=5):
    d = dense_model.encode(query, normalize_embeddings=True).tolist()
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