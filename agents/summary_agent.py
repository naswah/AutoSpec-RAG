import os
from anthropic import Anthropic
from state.graph_state import AgenticState

def summary_agent_node(state: AgenticState):
    print(f"\n=== [Agent 5: Summary Agent] Generating Comprehensive Plan Summary ===")
    client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    
    valid_pages = state.get("valid_pages", [])
    if not valid_pages:
        print("No valid blueprint images found to summarize.")
        return {"plan_summary": "No blueprint images available for summary."}
    
    page_summaries = []
    
    prompt = """You are an expert architectural plan reviewer and construction estimator. 
Analyze this blueprint page/drawing image and write an exhaustive, highly detailed technical description of every single element, note, and layout configuration visible.

Provide a meticulous breakdown covering:
1. Drawing Type & Orientation: (e.g., Floor Plan, Foundation Plan, Exterior Elevation, Framing Details) along with scale/compass indicators if visible.
2. Spatial Layout & Dimensions: Room-by-room walkthrough, structural spans, exact dimensions, thickness of walls, clearance heights, and circulation pathways.
3. Features & Openings: Comprehensive description of doors, windows, millwork, built-ins, partitions, and specialized assemblies.
4. General Notes & Callouts: Incorporate text schedules, legend items, or structural callouts present directly on the sheet.

🚨 CRITICAL CONSTRAINT: DO NOT assign, use, or reference any CSI MasterFormat division codes (e.g., '09 30 13' or 'Division 09'). Focus entirely on a narrative, physical, and architectural transcription of what is shown.

Write your response in clear, professional, un-abbreviated plain text paragraphs with clear headings."""

    for page in valid_pages:
        page_no = page["page_no"]
        print(f"Extracting all detailed plan data from sheet {page_no}...")
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3500, 
                temperature=0.1,  
                system="You are a precise architectural description engine. Write comprehensive, fully expanded technical descriptions of construction drawings. Do not summarize or skip small details—write out everything observed. Start directly with your findings.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": page['image_b64']
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            page_text = response.content[0].text.strip()
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