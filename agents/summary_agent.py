import os
from google import genai
from google.genai import types
from state.graph_state import AgenticState

def summary_agent_node(state: AgenticState):
    print(f"\n=== [Agent 5: Summary Agent] Generating Comprehensive Plan Summary ===")
    client = genai.Client()
    
    valid_pages = state.get("valid_pages", [])
    if not valid_pages:
        print("No valid blueprint images found to summarize.")
        return {"plan_summary": "No blueprint images available for summary."}
    
    page_summaries = []
    
    prompt = """You are an expert architectural plan reviewer and construction estimator. 
    Analyze the provided blueprint drawing image and write an exhaustive technical description of its layout, configurations, and specs.

    Your response MUST follow these structural rules perfectly:
    1. It must consist of EXACTLY TWO (2) paragraphs total. No more, no less.
    2. DO NOT use markdown headings (##), bullet points, numbered lists, or bold keys. Write ONLY two continuous text blocks.
    3. Paragraph 1 Focus: Technical overview of the drawing type, spatial layout, scale, dimensions, wall thicknesses, orientation, and room-by-room traffic circulation.
    4. Paragraph 2 Focus: Detailed breakdown of physical features, structural frameworks, openings (doors/windows), millwork callouts, and any miscellaneous visual notations or legends found on the sheet.

    🚨 CRITICAL CONSTRAINT: DO NOT assign, use, or reference any CSI MasterFormat division codes (e.g., '09 30 13' or 'Division 09'). Focus entirely on a narrative, physical description.

    Write your response in clear, professional plain text prose using exactly two paragraphs."""

    for page in valid_pages:
        page_no = page["page_no"]
        print(f"Extracting all detailed plan data from sheet {page_no}...")
        try:
            image_part = {
                'inline_data': {
                    'mime_type': 'image/png',
                    'data': page['image_b64']
                }
            }

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,  
                    system_instruction="You are a precise architectural description engine. Write comprehensive, fully expanded technical descriptions of construction drawings. Do not summarize or skip small details—write out everything observed. Start directly with your findings.",
                ),
            )
            
            page_text = response.text.strip()
            page_header = f"=================================================================\n" \
                          f" SHEET / PAGE {page_no} DETAILED REPORT\n" \
                          f"=================================================================\n"
            page_summaries.append(f"{page_header}{page_text}")
            
        except Exception as e:
            print(f"[Summary Agent Error] Failed to process page {page_no}: {e}")
            page_summaries.append(f"SHEET / PAGE {page_no}\n[Error compiling data for this blueprint page]")

    final_text_content = "\n\n\n".join(page_summaries)
    
    pdf_name = os.path.splitext(os.path.basename(state["pdf_path"]))[0]
    txt_output_path = os.path.join(state["output_base"], "data", f"{pdf_name}_plan_summary.txt")
    os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
    
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(f"DETAILED CONSTRUCTION PLAN SUMMARY REPORT\n")
        f.write(f"Source Document: {pdf_name}.pdf\n")
        f.write(f"=================================================================\n\n")
        f.write(final_text_content)
        
    print(f"Exhaustive architectural summary stored in text file: {txt_output_path}")
    
    return {"plan_summary": final_text_content}