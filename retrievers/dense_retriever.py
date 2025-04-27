from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, corpus: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.corpus = corpus
        self.embeddings = self.model.encode(corpus, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        return [self.corpus[i] for i in indices[0]]