from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class BM25Retriever:
    def __init__(self, documents):
        # BM25 is based on the term frequency - inverse document frequency (TF-IDF) model
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(documents)

    def retrieve(self, query, top_k=5):
        # Transform query into vector space (TF-IDF)
        query_vector = self.vectorizer.transform([query])
        similarities = np.dot(self.X, query_vector.T).toarray().flatten()
        ranked_indexes = similarities.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in ranked_indexes]
