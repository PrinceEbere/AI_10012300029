# Student: Prince Ebere Enoch, Index: [Your Index Number]

import numpy as np
import faiss

class Retriever:
    """
    Custom retrieval system using FAISS for vector search.
    Implements top-k retrieval with similarity scoring.
    """
    
    def __init__(self, index, chunks):
        """
        Initialize retriever with FAISS index and text chunks.
        
        Args:
            index: FAISS index (IndexFlatIP)
            chunks: List of text chunks corresponding to embeddings
        """
        self.index = index
        self.chunks = chunks
    
    def search(self, query_embedding, k=5):
        """
        Retrieve top-k similar chunks using FAISS.
        
        Args:
            query_embedding: Query embedding vector (numpy array)
            k: Number of top results to retrieve
        
        Returns:
            results: List of top-k similar chunks
            scores: List of similarity scores (0-1 range)
        """
        # Ensure query embedding is float32
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Flatten results (FAISS returns 2D arrays)
        distances = distances[0]
        indices = indices[0]
        
        # Normalize scores to 0-1 range (cosine similarity from inner product)
        scores = np.clip(distances, 0, 1)
        
        # Retrieve corresponding chunks
        results = [self.chunks[idx] for idx in indices if idx < len(self.chunks)]
        scores = scores[:len(results)]
        
        return results, scores
    
    def rerank(self, results, scores, rerank_model=None):
        """
        Re-rank retrieved results for improved accuracy.
        
        Args:
            results: List of retrieved chunks
            scores: List of similarity scores
            rerank_model: Optional model for re-ranking
        
        Returns:
            reranked_results: Re-ranked chunks
            reranked_scores: Re-ranked scores
        """
        # Simple re-ranking: sort by score descending
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        reranked_results = [res for res, _ in ranked]
        reranked_scores = [score for _, score in ranked]
        
        return reranked_results, reranked_scores