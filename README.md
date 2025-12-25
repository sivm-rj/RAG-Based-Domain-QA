
# RAG-Based Domain QA with LLaMA-2 7B

Interview-ready Retrieval-Augmented Generation (RAG) system using LLaMA-2, FAISS, and LangChain-style orchestration.

## Features
- FAISS vector retrieval over domain documents
- LLaMA-2 7B generation
- Modular agent-based design
- Benchmark evaluation (QA accuracy)
- API via FastAPI

## Setup
pip install -r requirements.txt

## Run
python embeddings/build_index.py
python scripts/run_pipeline.py

## API
uvicorn api.app:app --reload
