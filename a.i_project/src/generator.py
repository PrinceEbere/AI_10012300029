import os
from groq import Groq
from dotenv import load_dotenv
import logging

# ----------------------------
# LOAD ENV VARIABLES
# ----------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# Lazy client initialization (IMPORTANT)
_client = None

def get_client():
    global _client

    if _client is None:
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        _client = Groq(api_key=api_key)

    return _client


# ----------------------------
# RESPONSE GENERATION
# ----------------------------
def generate_response(query, context_chunks):
    """
    Generate response using Groq with controlled memory usage.
    """

    # 🔥 LIMIT CONTEXT SIZE (VERY IMPORTANT)
    context_chunks = context_chunks[:3]  # limit chunks

    if context_chunks:
        context = "\n\n".join(chunk[:300] for chunk in context_chunks)

        prompt = (
            "You are an academic assistant. Use ONLY the provided context to answer the question. "
            "If the answer is not in the context, respond with 'I don't know.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    else:
        prompt = f"Answer the question: {query}"

    try:
        client = get_client()

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,   # 🔥 reduced from 500
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.error(f"Groq error: {str(e)}")

        error_str = str(e).lower()

        if "rate limit" in error_str:
            return "Error: Groq rate limit exceeded. Try again later."

        if "model" in error_str:
            return "Error: Groq model unavailable."

        return "Error generating response."
