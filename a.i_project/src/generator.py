# Student: Prince Ebere Enoch, Index: [Your Index Number]

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ No API key found. Set GROQ_API_KEY in .env file or system environment.")

# Initialize Groq client
client = Groq(api_key=api_key)

# Debug: List available models (remove after testing)
try:
    models = client.models.list()
    print("Available Groq models:", [m.id for m in models.data])
except Exception as e:
    print(f"Error listing models: {e}")

def generate_response(query, context_chunks):
    """
    Generate response using Groq with retrieved context.
    """
    if context_chunks:
        context = "\n\n".join(context_chunks)
        prompt = (
            "You are an academic assistant. Use ONLY the provided context to answer the question. "
            "If the answer is not in the context, respond with 'I don't know.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    else:
        prompt = f"Answer the question: {query}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
        )
        choice = response.choices[0]
        if getattr(choice, "message", None) is not None:
            return choice.message.content.strip()
        return str(getattr(choice, "text", "")).strip()
    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str:
            return "Error: Groq rate limit exceeded. Try again later."
        if "model" in error_str:
            return "Error: Groq model unavailable. Check model name or API plan."
        return f"Error generating response: {str(e)}"