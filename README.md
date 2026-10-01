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

1. Python 3.10+

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

Output: Exports the results into the output/ and Results/ directories.

## Output📊
A structurted JSON with the CSI division, Notes and Descrption of the materials and Category present in the user architectural plan.


# Changes in Second Approach
Previously, the workflow was such that the ingestion agent used to generate list of materials after OCR+LLM detected tables follwed by CSI division agent that used to query the vector database for CSI division of the material (using name+notes and category). Validation agent checks the csi format, if not in format then assigns 00 00 00. Then the post processing steps took place like removing duplicates, generating short notes, rename "codes" to "name", etc.
<p align="center">
  <img src="docs\first approach.png" alt="Project Screenshot" width="600">
</p>

In second approach: We add a new agent named dedup agent (for flattenting ccategories, removing duplicates in initial phjase, ect) Now, the workflow is such that the ingestion agent generates the list of materials after OCR and LLM detects the table. Dedup Agent runs for basic flattening of categories and removing exact duplicates. The post processing steps conmes to action (removing duplicates via paraphrasing, generating short notes, renaming code to name, etc). Then the CSI agent queries the database by sending the name + short notes and the validation agent verifys the CSI format. Langgraph orchestration was changed.
<p align="center">
  <img src="docs\second approach.png" alt="Project Screenshot" width="600">
</p>

## Why second approach was tried
The previous method provided "00 00 00" as CSI divisions for many materials thet included "code" instead of name. For example: "code": "F-30" In this case, the estimation_notes were passes as a query for vector datanase (which had noise for eg:  
"notes": "FLOOR TILE, size: ~2\", CERAMIC, HEXAGONAL PATTERN, Manufacturer/Model: -",)\
which is why:
1. We first replaced all the "codes" with the actual name of the material in post-processing step so now, all the materials have "name" key, no "code" key.
2. Post-processing step also gave short notes.
3. We provide the "name" as well as "notes" (short notes) key to the vector db as query.

## Result from second approach

The result was satisfactory. The intended goal was successful (F-30 now provided correct CSI division) but there were some cases where the CSI divisions were provided for materials in first approach but the same material's CSI code was not provided using second approach.