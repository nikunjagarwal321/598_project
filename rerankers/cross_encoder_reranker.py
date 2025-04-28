from typing import List
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize the cross-encoder based reranker.
        
        Args:
            model_name (str): Name of the pre-trained cross-encoder model to use
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, docs: List[str]) -> List[str]:
        """
        Rerank documents using a cross-encoder model to jointly score query-document pairs.
        
        Args:
            query (str): The search query
            docs (List[str]): List of documents to rerank
            
        Returns:
            List[str]: Documents reranked by cross-encoder relevance scores
        """
        if not docs:
            return []
            
        # Create query-document pairs for cross-encoder scoring
        query_doc_pairs = [[query, doc] for doc in docs]
        
        # Compute relevance scores
        scores = self.model.predict(query_doc_pairs)
        
        # Create (score, document) pairs
        scored_docs = list(zip(scores, docs))
        
        # Sort by score in descending order
        ranked_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        
        # Return reranked documents
        return [doc for _, doc in ranked_docs]
