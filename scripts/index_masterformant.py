import os
import re
import uuid
import time
import ast
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance, PointStruct
from tools.helpers import dense_model, sparse_model, COLLECTION, TARGET_DIMENSIONS
from config import MASTERFORMAT_CSV

CSV_SRC =MASTERFORMAT_CSV


def _parse_synonym_segments(alt_terms_text):
    """
    Parent-level 'alternate_terms' fields are often a flat list of multiple
    synonym groups for DIFFERENT children, formatted as:
        "Category Name: syn1: syn2. Category Name 2: syn1: syn2."
    e.g. for 07 21 00 (Thermal Insulation):
        "blanket insulation: batt insulation. Board insulation: rigid insulation:
         semi-rigid insulation. Loose fill insulation: fill insulation: granular insulation."
    This splits that into [("blanket insulation", ["batt insulation"]), ...].
    """
    if not isinstance(alt_terms_text, str) or not alt_terms_text.strip():
        return []
    segments = [s.strip() for s in alt_terms_text.split(".") if s.strip()]
    parsed = []
    for seg in segments:
        parts = [p.strip() for p in seg.split(":") if p.strip()]
        if len(parts) >= 2:
            parsed.append((parts[0], parts[1:]))
    return parsed


def _normalize_title(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def redistribute_alternate_terms(df):

    children_by_parent = {}
    for idx, row in df.iterrows():
        children_by_parent.setdefault(row["parent_code"], []).append(idx)

    added_count = 0
    for idx, row in df.iterrows():
        segments = _parse_synonym_segments(row["alternate_terms"])
        if not segments:
            continue

        child_indices = children_by_parent.get(row["code"], [])
        if not child_indices:
            continue

        for category_name, synonyms in segments:
            cat_norm = _normalize_title(category_name)
            for child_idx in child_indices:
                child_title_norm = _normalize_title(df.at[child_idx, "title"])
                if cat_norm == child_title_norm or cat_norm in child_title_norm or child_title_norm in cat_norm:
                    addition = f"{category_name}: {': '.join(synonyms)}"
                    existing = df.at[child_idx, "alternate_terms"]
                    if isinstance(existing, str) and existing.strip():
                        df.at[child_idx, "alternate_terms"] = f"{existing.rstrip('.')}. {addition}."
                    else:
                        df.at[child_idx, "alternate_terms"] = f"{addition}."
                    added_count += 1
                    break  # first matching child wins

    print(f"Redistributed {added_count} parent-level synonym segment(s) down to their matching child codes.")
    return df


def build_chunk_text(r):
    parts = [f"Context: DIVISION {r['division']}: {r['division_name']}\nCode: {r['code']}\nTitle: {r['title']}"]
    if r["includes"]: parts.append(f"Includes: {r['includes']}")
    if r["may_include"]: parts.append(f"May include: {r['may_include']}")
    if r["alternate_terms"]: parts.append(f"Also known as: {r['alternate_terms']}")
    if r["notes"]: parts.append(f"Note: {r['notes']}")
    if r["discussion"]: parts.append(f"Discussion: {r['discussion']}")
    return "\n".join(parts)


def embed_with_retry(model, texts, max_retries=3, initial_delay=3):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return model.embed_documents(texts)
        except Exception as e:
            err_msg = str(e)
            if any(status in err_msg for status in ["502", "503", "429", "UNAVAILABLE", "Bad Gateway"]):
                print(f"Gemini API Gateway error encountered (Attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2 
            else:
                raise e
    return model.embed_documents(texts)


def run_indexing_setup():
    client = QdrantClient(url="http://localhost:6333")
    
    if not os.path.exists(CSV_SRC):
        print(f"Error: Could not locate the CSV data file at {CSV_SRC}")
        return

    print(f"Loading pre-parsed MasterFormat records from: {CSV_SRC}...")
    
    # Explicitly enforce UTF-8 to drop all character corruptions ("aE")
    df = pd.read_csv(CSV_SRC, encoding="utf-8")
    
    text_cols = ["division_name", "title", "includes", "may_include", "alternate_terms", "notes", "discussion", "parent_code"]
    for col in text_cols:
        df[col] = df[col].fillna("")
        
    # Format division strings accurately with zero-padding (e.g. "07")
    df["division"] = df["division"].fillna(0).astype(int).astype(str).str.zfill(2)

    def safe_parse_list(val):
        if isinstance(val, str) and val.strip():
            try:
                if val.startswith("[") and val.endswith("]"):
                    return ast.literal_eval(val)
                return [t.strip() for t in val.split(",") if t.strip()]
            except Exception:
                return []
        return []
    df["related_codes"] = df["related_codes"].apply(safe_parse_list)

    df = redistribute_alternate_terms(df)
    df["content"] = df.apply(build_chunk_text, axis=1)

    print(f"Resetting and preparing Qdrant storage collection: '{COLLECTION}'...")

    if client.collection_exists(collection_name=COLLECTION):
        client.delete_collection(collection_name=COLLECTION)
        
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=TARGET_DIMENSIONS, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )

    BATCH_SIZE = 32 
    NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678") 
    
    print(f"Executing batch indexing pipeline ({BATCH_SIZE} elements per run)...")
    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start : start + BATCH_SIZE]
        texts = chunk["content"].tolist()

        dense_vectors = embed_with_retry(dense_model, texts)
        
        time.sleep(0.1)
        sparse_vectors = list(sparse_model.embed(texts))

        points = []
        for (_, row), d_vec, s_vec in zip(chunk.iterrows(), dense_vectors, sparse_vectors):
            point_id = str(uuid.uuid5(NAMESPACE, str(row["code"])))
            
            payload = {
                "content": row["content"],  
                "code": str(row["code"]),
                "title": row["title"],
                "division": row["division"],
                "division_name": row["division_name"],
                "level": int(row["level"]) if pd.notna(row["level"]) else 1,
                "parent_code": str(row["parent_code"]),
                "related_codes": row["related_codes"],
                "includes": row["includes"],
                "may_include": row["may_include"],
                "alternate_terms": row["alternate_terms"],
                "notes": row["notes"],
                "discussion": row["discussion"],
                "source": "MasterFormat 2018",
            }

            points.append(PointStruct(
                id=point_id,
                vector={
                    "dense": d_vec,
                    "sparse": {
                        "indices": s_vec.indices.tolist(),
                        "values": s_vec.values.tolist()
                    }
                },
                payload=payload
            ))

        client.upsert(collection_name=COLLECTION, points=points)
        print(f"Upserted items tracking update: {start + len(chunk)}/{len(df)}")
        
        time.sleep(0.4)

    print("\nSuccess: MasterFormat 2018 database fully generated from CSV and loaded into Qdrant!")

if __name__ == "__main__":
    run_indexing_setup()