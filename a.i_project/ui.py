# ----------------------------
# HERO HEADER (ChatGPT STYLE)
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
# SIDEBAR (Enhanced but SAFE)
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

    if st.button("🧹 Clear Screen"):
        st.rerun()


# ----------------------------
# CHAT HISTORY INIT
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ----------------------------
# CHAT DISPLAY (ChatGPT STYLE)
# ----------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------
# INPUT (ChatGPT STYLE)
# ----------------------------
query = st.chat_input("Ask anything about Ghana budget or elections...")

if query:

    # Save user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # ----------------------------
    # PROCESS RAG
    # ----------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            results, scores, answer = rag_pipeline(query)

            # Beautiful answer card
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffffff, #f3f6ff);
                padding: 1.2rem;
                border-radius: 14px;
                border-left: 5px solid #3b82f6;
                box-shadow: 0 6px 18px rgba(0,0,0,0.06);
                line-height: 1.6;
                color: #111827;
            ">
            {answer}
            </div>
            """, unsafe_allow_html=True)

            # Context
            with st.expander("📄 Retrieved Context (Sources)"):
                for i, (res, score) in enumerate(zip(results, scores)):
                    st.markdown(f"""
                    <div style="
                        background:#f9fafb;
                        padding:0.8rem;
                        border-radius:10px;
                        margin-bottom:0.5rem;
                        border:1px solid #e5e7eb;
                    ">
                    <strong>Chunk {i+1} | Score: {score:.2f}</strong><br>
                    {res[:250]}...
                    </div>
                    """, unsafe_allow_html=True)

    # Save assistant response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    logging.info(f"Query: {query} | Answer: {answer}")
