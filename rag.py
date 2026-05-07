from sentence_transformers import CrossEncoder
from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
HF_TOKEN = os.getenv("HF_TOKEN")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs):
    pairs = [[query, d["content"]] for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [r[0] for r in ranked[:5]]


def build_prompt(materials_json, ocr_data, retrieved_chunks, schema_chunk):

    context = "\n\n".join([c["content"] for c in retrieved_chunks])

    return f"""
You are an expert construction cost & CSI classification system.

TASK:
1. Take filtered building materials
2. Map them to correct CSI MasterFormat codes (pattern: XX XX XX or XX XX XX.XX e.g., '03 30 00' do not just output 03)
3. Populate the "notes" field with specific material details (sizes, PSI, thickness) from the "specs" field.
4. Produce structured JSON output strictly following schema

--- FILTERED MATERIALS (GEMINI OUTPUT) ---
{json.dumps(materials_json, indent=2)}

--- RAW OCR CONTEXT (SUPPORT ONLY) ---
{json.dumps(ocr_data, indent=2)}

--- MASTERFORMAT KNOWLEDGE (VECTOR DB CONTEXT) ---
{context}

--- OUTPUT SCHEMA ---
{schema_chunk["content"]}

RULES:
- **CSI Code Format:** Use the full 6-digit code with spaces (XX XX XX). Do NOT provide 2-digit codes.
- **Accuracy:** If the knowledge base mentions "06 10 00 Rough Carpentry" for Lumber, use that full code.
- Return ONLY the JSON object.
"""


def generate_response(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"} 
        )
        
        content = chat_completion.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Groq RAG Error: {e}")
        return {"error": str(e)}