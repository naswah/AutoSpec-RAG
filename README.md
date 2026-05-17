# AutoSpec RAG
A sophisticated RAG (Retrieval-Augmented Generation) system designed to extract building materials from architectural house plans (PDFs) and map them to standard CSI MasterFormat (US, 2018) divisions using AI and Computer Vision.

## Overview 🏗️
This project automates the manual process of material estimation by processing architectural drawings through a multi-stage pipeline. It makes the use of LLM and get context from the vector database to provide the result. An agentic workflow is used which is created using Langgraph.

## Key Features 🌟
1. Automated PDF Conversion: Converts multi-page architectural PDFs into high-resolution images for processing.

2. Vision-Aided Extraction: Uses VLMs like Gemini/Groq to extract the materials from the image.

3. Mapping: If codes are present in the document, then the codes are mapped with their respective tables or schedules.

4. CSI MasterFormat Mapping: Uses a Hybrid Search (Vector + Keyword) via Qdrant to map extracted materials to official CSI divisions.

## Installation and Setup 🛠️
### Prerequisites 🐍

1. Python 3.10+

2. Qdrant instance running (default: localhost:6333 )

### Install dependencies 📥
In terminal: pip install -r requirements.txt

### Configuration 🔐
Create an env file (.env) \
GROQ_API_KEY=your_groq_api_key_here \
HF_TOKEN=your_hf_token_here

### Usuage
python main.py

## Pipeline Flow 🔄
1. Ingestion: main.py initializes the pipeline, pulling raw data from inputs using tools/pdf_helpers.py.

2. Parsing: agents/ingestion_agent.py processes and chunks the raw text.

3. State Management: state/graph_state.py maintains the shared memory/state across the execution graph.

4. Agent Processing Loop:

    - Blueprint Mapper: Aligns inputs with construction schemas and master formats.

    - CSI Classifier: Categorizes cost items into industry-standard CSI divisions.

    - Validator Agent: Quality-checks calculations and data consistency.

Output: tools/helpers.py formats the final validated state and exports the results into the output/ and Results/ directories.

## Output📊
A structurted JSON with the CSI division, Notes and Descrption of the materials present in the user architectural plan.


## Note: Please check the dev branch for latest code updates.