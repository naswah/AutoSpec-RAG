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

def mappings(materials_list):
    code_lookup= {}

    for page in materials_list:
        for view in page.get("views", []):
            materials= view.get("materials", [])
            for mat_key, mat_value in materials.items():
                if mat_value != "Mapping Required" and mat_value != "none":
                    code_lookup[mat_key] = mat_value

    for page in materials_list:
        for view in page.get("views", []):
            materials= view.get("materials", {})
            for mat_name in list(materials.keys()):
                if materials[mat_name] == "Mapping Required":
                    code_to_find = mat_name

                    if code_to_find in code_lookup:
                        materials[mat_name] = code_lookup[code_to_find]
                        print(f"Mapped {code_to_find} to its definition.")
    output_folder = os.path.join(os.getcwd(), "Result 2")
    os.makedirs(output_folder, exist_ok=True)

    save_path = os.path.join(output_folder, "pdf_final.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(materials_list, f, indent=4, ensure_ascii=False)
        
    print(f"Resolved JSON saved to: {save_path}")

    return materials_list

# Numerical data le rag search ma affect garna sakcha so we do not send the numerical data to search in RAG, we jus send the material details. We get the context from the vector database. Tyo context ra original filtered json file LLM lai pathaune in order to give nice and correct output


def rerank(query, docs):
    if not docs: return []
        
    pairs = [[query, d.get("content", "")] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:5]]



def build_prompt(raw_json_string, retrieved_chunks):

    context = "\n\n".join([c["content"] for c in retrieved_chunks])

    return f"""
You are an expert construction cost & CSI classification system.

TASK:
- Take the provided JSON data. For every material item, find the matching 6-digit CSI code from the context. Not just for 1 material. DProvide CSI codes for all the ,materials that are detected.
- Update the "csi_division" field in the JSON with that code.Add correct CSI divisdion MasterFormat codes (pattern: XX XX XX or XX XX XX.XX e.g., '03 30 00' do not just output 03) in the provided json.
- DO NOT change any other values (dimensions, notes, or names).
- Return the EXACT same JSON structure, fully populated.

{raw_json_string}

{context}
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