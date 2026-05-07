# AutoSpec RAG
A sophisticated RAG (Retrieval-Augmented Generation) system designed to extract building materials from architectural house plans (PDFs) and map them to standard CSI MasterFormat (US, 2018) divisions using AI and Computer Vision.

## Overview 🏗️
This project automates the manual process of material estimation by processing architectural drawings through a multi-stage pipeline. It combines object detection (YOLO) to find text regions, OCR (Tesseract) to extract text, and LLMs  along with vector database to structure and classify materials against professional construction standards.

## Key Features 🌟
1. Automated PDF Conversion: Converts multi-page architectural PDFs into high-resolution images for processing.

2. Vision-Aided Extraction: Uses a custom-trained YOLO model to identify specific text blocks/ tables and annotations within complex drawings.

3. Smart Material Filtering: Leverages LLM models to filter raw OCR noise and extract only relevant building materials and specifications.

4. CSI MasterFormat Mapping: Uses a Hybrid Search (Vector + Keyword) via Qdrant to map extracted materials to official CSI divisions.

## Installation and Setup 🛠️
### Prerequisites 🐍

1. Python 3.10+
2. Tesseract OCR Engine installed on your system.

3. Qdrant instance running (default: localhost:6333 )

### Install dependencies 📥
In terminal: pip install -r requirements.txt

### Configuration 🔐
Create an env file (.env) \
GROQ_API_KEY=your_groq_api_key_here \
HF_TOKEN=your_hf_token_here

### Usuage
python main.py

## Pipeline Flow 🔄
1. Ingestion: PDF is split into images. YOLO detects text regions.

2. OCR: Tesseract reads text from detected regions.
3. Filtering: Groq filters the text to isolate "item", "specs", and "page".

4. Retrieval: Hybrid search finds matching CSI codes in the vector DB.

5. Generation: Final LLM pass formats the data into the required output schema.

## Output📊
A structurted JSON with the CSI division, Notes and Descrption of the materials present in the user architectural plan.