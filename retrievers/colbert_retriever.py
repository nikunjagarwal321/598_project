import numpy as np
from sentence_transformers import SentenceTransformer

class ColBERTRetriever:
    def __init__(self, documents):
        # Load ColBERT Model (Sentence Transformers)
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

        self.documents = documents
        self.document_embeddings = self.encode_documents(documents)

    def encode_documents(self, documents):
        # Token-level embeddings via Sentence-BERT
        embeddings = self.model.encode(documents)
        return np.array(embeddings)

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode([query])

        similarities = np.dot(self.document_embeddings, query_embedding.T)
        ranked_indexes = similarities.argsort(axis=0)[-top_k:][::-1]
        ranked_indexes = ranked_indexes.flatten()
        return [self.documents[i] for i in ranked_indexes]