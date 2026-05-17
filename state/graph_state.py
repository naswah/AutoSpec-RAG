from typing import TypedDict, List, Dict, Any

class AgenticState(TypedDict):
    pdf_path: str
    output_base: str
    valid_pages: List[Dict[str, Any]]         
    extracted_materials: List[Dict[str, Any]] 
    mapped_materials: List[Dict[str, Any]]    
    retrieved_context: str                    
    final_specifications: Dict[str, Any]      
    retry_count: int                          
    error_log: List[str]