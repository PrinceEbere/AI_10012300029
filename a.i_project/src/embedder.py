from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings



    