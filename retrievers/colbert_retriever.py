from __future__ import annotations
from typing import Sequence, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class ColBERTRetriever:
    """
    Simple ColBERT‑style retriever built on SentenceTransformers that runs on
    GPU if one is present.
    """

    def __init__(
        self,
        documents: Sequence[str],
        model_name: str = "bert-base-nli-mean-tokens",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.device     = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size

        # Load the model on the chosen device
        self.model = SentenceTransformer(model_name, device=str(self.device))

        self.documents = list(documents)
        self.document_embeddings = self._encode_documents(self.documents)

    @torch.inference_mode()
    def _encode_documents(self, docs: Sequence[str]) -> np.ndarray:
        """
        Compute embeddings in (mini‑)batches on GPU, then return them as a
        NumPy array on CPU to keep GPU RAM free.
        """
        embs: List[np.ndarray] = []
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i : i + self.batch_size]
            vecs  = self.model.encode(batch,
                                      batch_size=len(batch),
                                      device=self.device,
                                      convert_to_numpy=True,
                                      normalize_embeddings=False)      # shape (B, 768)
            embs.append(vecs)
        return np.vstack(embs).astype(np.float32)                       # (N, 768)

    @torch.inference_mode()
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Return the `top_k` passages most similar to `query`.
        """
        q_vec = self.model.encode(
            query,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=False,  # keep raw dot‑product space
        )                                # shape (768,)

        sims = self.document_embeddings @ q_vec.T                      # (N,)
        top  = np.argpartition(-sims, top_k - 1)[:top_k]
        top  = top[np.argsort(-sims[top])]                             
        return [self.documents[i] for i in top.tolist()]
