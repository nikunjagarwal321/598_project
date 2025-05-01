from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import numpy as np
import torch


class DPRRetriever:
    def __init__(self, documents, device='cuda'):
        # Load pre-trained DPR model for query and passage encoding
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.query_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            'facebook/dpr-question_encoder-single-nq-base')
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

        self.documents = documents
        self.document_embeddings = self.encode_documents(documents)

    def encode_documents(self, documents):
        embeddings = []

        # Batching: encode documents in chunks
        batch_size = 16  # Adjust this based on memory limits
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start:start + batch_size]

            # Tokenize the batch of documents and move tensors to the correct device
            inputs = self.context_tokenizer(batch_docs, return_tensors='pt', padding=True, truncation=True).to(
                self.device)

            # Disable gradient calculations and get embeddings
            with torch.no_grad():
                outputs = self.context_encoder(**inputs)
                embeddings_batch = outputs.pooler_output  # (batch_size, hidden_size)

            # Convert embeddings to NumPy array (if needed)
            embeddings.append(embeddings_batch.cpu().numpy())  # Move to CPU before converting to numpy

        # Combine all batches
        return np.concatenate(embeddings, axis=0)

    def retrieve(self, query, top_k=5):
        # Encode the query
        inputs = self.query_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            query_embedding = self.query_encoder(**inputs).pooler_output

        similarities = np.dot(self.document_embeddings, query_embedding.numpy().T)
        ranked_indexes = similarities.argsort(axis=0)[-top_k:][::-1]
        ranked_indexes = ranked_indexes.flatten()
        return [self.documents[i] for i in ranked_indexes]
