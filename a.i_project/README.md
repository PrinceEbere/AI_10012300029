# Academic City RAG Chatbot
**Student: Prince Ebere Enoch, Index: [Your Index Number]**
**Course: CS4241 - Introduction to Artificial Intelligence - 2026**

## Overview
This is a Retrieval-Augmented Generation (RAG) chatbot for Academic City, using Ghana's election results and 2025 budget data. It implements manual RAG components without pre-built frameworks.

## Features
- Data loading, cleaning, chunking (500 chars, 50 overlap)
- Embedding with Sentence Transformers, FAISS vector storage
- Top-k retrieval with similarity scoring, query expansion, re-ranking
- Prompt engineering with hallucination control
- Full pipeline: Query → Retrieval → Context Selection → Prompt → LLM → Response
- Adversarial testing and evaluation
- Streamlit UI

## Architecture
- Data Flow: Load → Clean → Chunk → Embed → Store (FAISS) → Retrieve (Top-k + Expansion) → Generate (Prompt) → Respond
- Components: Loader, Cleaner, Chunker, Embedder, Retriever, Generator
- Suitability: Modular for domain-specific queries; manual for educational purposes

## Installation
1. Clone the repo: `git clone https://github.com/yourusername/ai_[your_index].git`
2. Install dependencies: `pip install -r requirements.txt`
3. Add `.env` with `GROQ_API_KEY=your_key`
4. Run app: `python app.py` or UI: `streamlit run ui.py`

## Deployment
- GitHub: https://github.com/yourusername/ai_[your_index]
- Cloud: [Streamlit Cloud URL, e.g., https://your-app.streamlit.app]

## Video Walkthrough
[Link to 2-min video explaining design decisions]

## Experiment Logs
See `logs/experiment_logs.txt` for manual logs.

## Documentation
[Link to detailed docs]