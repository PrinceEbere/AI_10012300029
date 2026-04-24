# Student: Prince Ebere Enoch, Index: [Your Index Number]
# CS4241 - Introduction to Artificial Intelligence - 2026

import os
import logging
import streamlit as st
import numpy as np
import faiss

from src.loader import load_csv, load_pdf
from src.cleaner import clean_text
from src.chunker import chunk_text
from src.embedder import create_embeddings, get_model
from src.retriever import Retriever
from src.generator import generate_response


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI RAG Chat Assistant",
    page_icon="💬",
    layout="wide"
)


# ----------------------------
# LOGGING SETUP
# ----------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename='logs/experiment_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ----------------------------
# GLOBAL STYLING (CHATGPT STYLE LIGHT IMPROVEMENT)
# ----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f7faff 0%, #ffffff 100%);
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# DATA LOADING (CACHED)
# ----------------------------
@st.cache_data
def load_data():
    csv_data = load_csv("Data/Ghana_Election_Result.csv")
    pdf_text = load_pdf("Data/2025-Budget.pdf")

    logging.info(f"Loaded CSV rows={len(csv_data)}, PDF chars={len(pdf_text)}")

    csv_text = csv_data.astype(str).to_csv(index=False)
    return pdf_text + "\n" + csv_text


# ----------------------------
# VECTOR STORE (CACHED)
# ----------------------------
@st.cache_resource
def build_vector_store():
    combined_text = load_data()

    cleaned = clean_text(combined_text)
    chunks = chunk_text(cleaned, chunk_size=500, overlap=50)

    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]

    embeddings = create_embeddings(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    retriever = Retriever(index, chunks)

    return retriever, chunks


# ----------------------------
# RAG HELPERS
# ----------------------------
def expand_query(query):
    return query + " economic policy budget election government spending"


def select_context(chunks, scores, max_chars=1200):
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    selected = []
    total = 0

    for chunk, score in ranked:
        if total + len(chunk) > max_chars:
            break
        selected.append(chunk)
        total += len(chunk)

    return selected


# ----------------------------
# RAG PIPELINE (FIXED + CLEAN)
# ----------------------------
def rag_pipeline(query, retriever):
    expanded = expand_query(query)

    model = get_model()
    query_embedding = model.encode(expanded).astype("float32")

    results, scores = retriever.search(query_embedding, k=5)

    if len(scores) > 0 and np.max(scores) < 0.45:
        results, scores = retriever.rerank(results, scores)

    context = select_context(results, scores)

    answer = generate_response(query, context)

    return results, scores, context, answer


# ----------------------------
# HERO HEADER (CHATGPT STYLE)
# ----------------------------
st.markdown("""
<div style='text-align:center; padding:1.2rem 0;'>
    <h1 style='color:#1f2937;'>💬 Academic City AI Assistant</h1>
    <p style='color:#6b7280; font-size:1rem;'>
        Chat with your Ghana Budget & Election AI
    </p>
</div>
""", unsafe_allow_html=True)


# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("🎓 AI Assistant")

    st.markdown("### 👤 Student Info")
    st.write("Prince Ebere Enoch")
    st.write("Index: 10012300029")

    st.markdown("### ⚙️ System")
    st.write("RAG + FAISS + LLM")

    st.markdown("### 📊 Features")
    st.write("• Ghana Budget Q&A")
    st.write("• Election Insights")
    st.write("• Document Search")

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []


# ----------------------------
# CHAT MEMORY INIT
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
retriever, chunks = build_vector_store()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------
# INPUT BOX (CHATGPT STYLE)
# ----------------------------
query = st.chat_input("Ask anything about Ghana budget or elections...")

if query:

    # user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            results, scores, context, answer = rag_pipeline(query, retriever)

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffffff, #f3f6ff);
                padding: 1.2rem;
                border-radius: 14px;
                border-left: 5px solid #3b82f6;
                box-shadow: 0 6px 18px rgba(0,0,0,0.06);
                color: #111827;
            ">
            {answer}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📄 Retrieved Context"):
                for i, (res, score) in enumerate(zip(results, scores)):
                    st.markdown(f"**Chunk {i+1} | Score: {score:.2f}**")
                    st.write(res[:250])

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    logging.info(f"Query: {query} | Answer: {answer}")


# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Built with RAG • Academic City • 2026</div>",
    unsafe_allow_html=True
)
