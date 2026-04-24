# Student: Prince Ebere Enoch, Index: [Your Index Number]

import streamlit as st
import logging
import faiss
import numpy as np
from src.loader import load_csv, load_pdf
from src.cleaner import clean_text
from src.chunker import chunk_text
from src.embedder import create_embeddings, model as embedder_model
from src.retriever import Retriever
from src.generator import generate_response

# Page configuration and styling
st.set_page_config(
    page_title="Academic City RAG Chatbot",
    page_icon="🎓",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f2f8ff 0%, #ffffff 100%);
        color: #102a43;
    }
    .block-container {
        padding: 2rem 3rem 2rem 3rem;
        max-width: 1200px;
    }
    .css-1aumxhk { padding-top: 0rem; }
    .stButton>button {
        background: linear-gradient(90deg, #4d8cff 0%, #1a5ad1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.9rem 1.6rem !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 16px 30px rgba(26, 90, 209, 0.18) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 20px 28px rgba(26, 90, 209, 0.24) !important;
    }
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 14px !important;
        border: 1px solid #c6d6f4 !important;
        padding: 0.9rem 1rem !important;
        background: #f9fbff !important;
        color: #102a43 !important;
    }
    .stTextInput>div>label,
    .stTextArea>div>label {
        color: #1c3d74 !important;
        font-weight: 600 !important;
    }
    .stMarkdown h1 {
        font-size: 2.8rem;
        color: #082852;
        margin-bottom: 0.2rem;
    }
    .stMarkdown h2, .stMarkdown h3 {
        color: #123b6e;
    }
    .context-card {
        background: rgba(255,255,255,0.92);
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 18px 40px rgba(15, 45, 80, 0.06);
        margin-bottom: 1rem;
        border: 1px solid rgba(77, 142, 255, 0.12);
    }
    .highlight-box {
        background: linear-gradient(135deg, #eef6ff 0%, #ffffff 100%);
        border-left: 4px solid #4d8cff;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        color: #102a43;
        margin-bottom: 1rem;
    }
    .streamlit-expanderHeader {
        color: #0d3a72 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize logging
logging.basicConfig(filename='logs/experiment_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load and prepare data (cached for performance)
@st.cache_data
def load_and_prepare_data():
    csv_data = load_csv("Data/Ghana_Election_Result.csv")
    pdf_text = load_pdf("Data/2025-Budget.pdf")
    csv_text = csv_data.astype(str).to_csv(index=False)
    combined_text = pdf_text + "\n" + csv_text
    clean_text_combined = clean_text(combined_text)
    chunks = chunk_text(clean_text_combined, chunk_size=500, overlap=50)
    chunks = [chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
    embeddings = create_embeddings(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    retriever = Retriever(index, chunks)
    return retriever

retriever = load_and_prepare_data()

# Helper functions from app.py
def expand_query(query):
    synonyms = ["economic policy", "budget statement", "election results", "government spending"]
    return query + " " + " ".join(synonyms)

def build_prompt(query, context_chunks):
    if context_chunks:
        context = "\n\n".join(context_chunks)
        return (
            "You are an academic assistant for Academic City. "
            "Use ONLY the provided context to answer the question. "
            "If the information is not present, say 'I don't know.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    return f"Answer the question: {query}"

def select_context(chunks, scores, max_chars=1200):
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    selected = []
    total_chars = 0
    for chunk, score in ranked:
        if total_chars + len(chunk) > max_chars:
            break
        selected.append(chunk)
        total_chars += len(chunk)
    return selected

def rag_pipeline(query, retriever):
    expanded_query = expand_query(query)
    query_embedding = embedder_model.encode(expanded_query).astype("float32")
    results, scores = retriever.search(query_embedding, k=5)
    if len(scores) > 0 and np.min(scores) < 0.45:
        results, scores = retriever.rerank(results, scores)
    context_chunks = select_context(results, scores, max_chars=1200)
    prompt = build_prompt(query, context_chunks)
    answer = generate_response(query, context_chunks)
    return results, scores, prompt, answer

# Streamlit UI
st.markdown("# Academic City RAG Chatbot")
st.markdown("#### A bright, professional AI assistant for Ghana budget and election insights.")

with st.container():
    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Student: Prince Ebere Enoch")
        st.markdown("**Index:** 10012300029  ")
        st.write("This tool answers domain-specific questions using a manual retrieval-augmented generation pipeline.")
    with right:
        st.markdown("<div class='highlight-box'><strong>Quick Guide</strong><br>Ask about Ghana's economy, budget, or election results and get context-aware answers.</div>", unsafe_allow_html=True)

query = st.text_input("Enter your query about Ghana's economy, budget, or elections:")
submit_button = st.button("Submit Query")

if submit_button:
    if query.strip():
        with st.spinner("Finding relevant context and generating a response..."):
            results, scores, prompt, answer = rag_pipeline(query, retriever)

        st.markdown("### AI Response")
        st.markdown(f"<div class='context-card'>{answer}</div>", unsafe_allow_html=True)

        with st.expander("Retrieved Context and Scores", expanded=False):
            for i, (res, score) in enumerate(zip(results, scores)):
                st.markdown(f"<div class='context-card'><strong>Chunk {i+1} (Score: {score:.2f})</strong><br>{res[:260]}...</div>", unsafe_allow_html=True)

        with st.expander("Final Prompt Sent to LLM", expanded=False):
            st.text_area("Prompt", prompt, height=220)

        logging.info(f"UI Query: {query} | Answer: {answer}")
    else:
        st.error("Please enter a query before submitting.")

st.markdown("---")

with st.expander("Architecture & Pipeline Details", expanded=True):
    st.markdown("- **Data Flow:** Load → Clean → Chunk → Embed → Store (FAISS) → Retrieve (Top-k + Expansion) → Generate (Prompt) → Respond")
    st.markdown("- **Components:** Loader, Cleaner, Chunker, Embedder, Retriever, Generator")
    st.markdown("- **Design:** Professional, modular, and tailored to Academic City use cases")
    st.markdown("- **Visual:** Bright palette, polished cards, animated primary button")
