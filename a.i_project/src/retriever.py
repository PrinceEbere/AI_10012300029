import numpy as np


class Retriever:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def search(self, query_embedding, k=5):
        query_embedding = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_embedding, k)

        distances = distances[0]
        indices = indices[0]

        results = []
        scores = []

        for idx, score in zip(indices, distances):
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
                scores.append(float(score))

        scores = np.clip(scores, 0, 1)

        return results, scores

    def rerank(self, results, scores, rerank_model=None):
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

        reranked_results = [r for r, _ in ranked]
        reranked_scores = [s for _, s in ranked]

        return reranked_results, reranked_scores
