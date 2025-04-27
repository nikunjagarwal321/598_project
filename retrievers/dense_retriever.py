from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import numpy as np
import torch


class DPRRetriever:
    def __init__(self, documents):
        # Load pre-trained DPR model for query and passage encoding
        self.query_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            'facebook/dpr-question_encoder-single-nq-base')
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

        self.documents = documents
        self.document_embeddings = self.encode_documents(documents)

    def encode_documents(self, documents):
        embeddings = []
        for doc in documents:
            # Tokenize the document and pass through the encoder
            inputs = self.context_tokenizer(doc, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                embedding = self.context_encoder(**inputs).pooler_output

            embedding = embedding.squeeze(0).numpy()  # squeeze to remove batch dimension
            embeddings.append(embedding)
        return np.array(embeddings)

    def retrieve(self, query, top_k=5):
        # Encode the query
        inputs = self.query_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            query_embedding = self.query_encoder(**inputs).pooler_output

        similarities = np.dot(self.document_embeddings, query_embedding.numpy().T)
        ranked_indexes = similarities.argsort(axis=0)[-top_k:][::-1]
        ranked_indexes = ranked_indexes.flatten()
        return [self.documents[i] for i in ranked_indexes]
