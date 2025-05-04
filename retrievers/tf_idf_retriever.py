from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(documents)

    def retrieve(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(self.X, query_vector).flatten()
        ranked_indexes = similarities.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in ranked_indexes]
