# AutoSpec RAG
A sophisticated agentic RAG (Retrieval-Augmented Generation) system designed to extract building materials from architectural house plans (PDFs) and map them to standard CSI MasterFormat (US, 2018) divisions using AI and Computer Vision.

## Overview 🏗️
This project automates the manual process of material estimation by processing architectural drawings through a multi-stage pipeline. It makes the use of LLM and get context from the vector database to provide the result. An agentic workflow is used which is created using Langgraph.

## Key Features 🌟
1. Automated PDF Conversion: Converts multi-page architectural PDFs into high-resolution images for processing.

2. Vision-Aided Extraction: Uses Claude VLM to extract the materials from the image.

3. Mapping: The window/door or any other schedule is passed in every batch for mapping the codes to their respective details.

4. CSI MasterFormat Mapping: Uses a Hybrid Search (Semantic + Keyword) via Qdrant (gemini embeddings) to map extracted materials to official CSI divisions.

## Installation and Setup 🛠️
### Prerequisites 🐍

1. Python 3.12.7

2. Qdrant instance running (default: localhost:6333 )

3. API Keys for Gemini, Claude, HuggingFace token

### Install dependencies 📥
In terminal: pip install -r requirements.txt

### Configuration 🔐
Create an env file (.env) \
CLAUDE_API_KEY=your_claude_api_key_here \
GEMINI_API_KEY=your_gemini_api_key_here \
HF_TOKEN=your_hf_token_here

### Usuage
1. Open Qdrant
2. Run index_masterformat.py 
3. Update path for user plan in config.py
4. Run python main.py

## Pipeline Flow 🔄
1. Ingestion: main.py initializes the pipeline, pulling raw data from inputs using tools/pdf_helpers.py.

2. Parsing: agents/ingestion_agent.py processes and chunks the raw text.

3. State Management: state/graph_state.py maintains the shared memory/state across the execution graph.

4. Agent Processing Loop:

    - CSI Classifier: Categorizes cost items into industry-standard CSI divisions.

    - Validator Agent: Quality-checks calculations and data consistency.

    - Summary Agent: Provides summary of the user plan.

Output: Exports the final results into the local/results directory, scale JSON to local/scale directory.

## Output📊
A structurted JSON with the CSI division, Notes and Descrption of the materials and Category present in the user architectural plan.