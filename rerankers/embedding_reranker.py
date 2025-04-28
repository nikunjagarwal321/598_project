from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingReranker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding-based reranker.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use for embeddings
        """
        self.model = SentenceTransformer(model_name)
    
    def rerank(self, query: str, docs: List[str]) -> List[str]:
        """
        Rerank documents based on embedding similarity with the query.
        
        Args:
            query (str): The search query
            docs (List[str]): List of documents to rerank
            
        Returns:
            List[str]: Documents reranked by similarity to the query
        """
        if not docs:
            return []
            
        # Generate embeddings for query and documents
        query_embedding = self.model.encode([query])[0]
        doc_embeddings = self.model.encode(docs)
        
        # Calculate cosine similarities between query and each document
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Create pairs of (similarity, document)
        similarity_doc_pairs = list(zip(similarities, docs))
        
        # Sort by similarity in descending order
        sorted_pairs = sorted(similarity_doc_pairs, key=lambda x: x[0], reverse=True)
        
        # Return only the documents in the new order
        return [doc for _, doc in sorted_pairs]
