from .bm25_retriever import BM25Retriever
from .colbert_retriever import ColBERTRetriever

class HybridRetriever:
    def __init__(self, documents):
        # Initialize both BM25 and Dense retrievers
        self.bm25_retriever = BM25Retriever(documents)
        self.colbert_retriever = ColBERTRetriever(documents)

    def retrieve(self, query, top_k=5):
        # Retrieve using BM25 and Dense Retriever
        bm25_results = self.bm25_retriever.retrieve(query, top_k)
        colbert_results = self.colbert_retriever.retrieve(query, top_k)

        # Combine results, you can also add a re-ranking mechanism here
        combined_results = list(set(bm25_results + colbert_results))

        # Return top-k results (after combining and re-ranking)
        return combined_results[:top_k]