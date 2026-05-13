from sentence_transformers import CrossEncoder
from groq import Groq
import json
import os
from dotenv import load_dotenv
import re

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
HF_TOKEN = os.getenv("HF_TOKEN")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Numerical data le rag search ma affect garna sakcha so we do not send the numerical data to search in RAG, we jus send the material details. We get the context from the vector database. Tyo context ra original filtered json file LLM lai pathaune in order to give nice and correct output


def clean_query_text(text):
    """
    Removes numerical measurements (e.g., 4", 1/2', 3000 PSI) from the query string.
    This ensures semantic search focuses strictly on the material classification.
    """
    if not text:
        return ""
    # Matches numbers (with optional decimals/fractions) and common units or quotes following them
    cleaned = re.sub(r'\b\d+(?:[\.,/]\d+)?\s*(?:"|\'|mm|cm|in|inch|ft|lbs|psi)?\b', '', text, flags=re.IGNORECASE)
    
    cleaned = cleaned.replace('"', '').replace("'", "")
    return re.sub(r'\s+', ' ', cleaned).strip()



def rerank(query, docs):
    cleaned_query = clean_query_text(query)
    
    pairs = [[cleaned_query, d["content"]] for d in docs]
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
4. Check the "Notes" or "Genral Notes" or "Special Notes" section for context. If anny impotant details is provided in the note, try to relate it in oder to provide the CSI division.
5. Produce structured JSON output strictly following schema 

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
- **Accuracy:** If the knowledge base mentions "06 10 00 Rough Carpentry" for Lumber, use that full code. If the item lies in subcategory, then provide the CSI Division for the sub category.
- **Notes Field Preservation:** Ensure ALL numerical measurements and sizes (e.g., 4", 3000 PSI) present in the original "specs" are exactly preserved and placed into the "notes" field along with the necessady detailed information mentioned.
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
            temperature=0.2,
            response_format={"type": "json_object"} 
        )
        
        content = chat_completion.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Groq RAG Error: {e}")
        return {"error": str(e)}