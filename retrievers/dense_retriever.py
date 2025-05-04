from __future__ import annotations
from typing import List, Sequence

import numpy as np
import torch
from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
)

class DPRRetriever:
    """
    Dense Passage Retrieval (DPR) helper that encodes the query and passages on
    GPU if available, otherwise on CPU.
    """

    def __init__(
        self,
        documents: Sequence[str],
        model_name_ctx: str = "facebook/dpr-ctx_encoder-single-nq-base",
        model_name_q: str = "facebook/dpr-question_encoder-single-nq-base",
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        self.device   = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_sz = batch_size

        # --- Load models & tokenizers --------------------------------------------------
        self.context_encoder = DPRContextEncoder.from_pretrained(model_name_ctx).to(self.device)
        self.query_encoder   = DPRQuestionEncoder.from_pretrained(model_name_q) .to(self.device)

        self.ctx_tok = DPRContextEncoderTokenizer .from_pretrained(model_name_ctx)
        self.q_tok   = DPRQuestionEncoderTokenizer.from_pretrained(model_name_q)

        self.documents  = list(documents)
        self.document_embeddings = self._encode_documents(self.documents)

    @torch.inference_mode()
    def _encode_documents(self, docs: Sequence[str]) -> np.ndarray:
        """
        Embed passages in (mini‑)batches on the chosen device, then move the
        resulting vectors back to CPU / NumPy so they don’t tie up GPU RAM.
        """
        all_embs: list[np.ndarray] = []
        for i in range(0, len(docs), self.batch_sz):
            batch_texts = docs[i : i + self.batch_sz]
            toks = self.ctx_tok(batch_texts,
                                return_tensors="pt",
                                truncation=True,
                                padding=True).to(self.device)

            embs = self.context_encoder(**toks).pooler_output           # (B, 768)
            all_embs.append(embs.cpu().numpy())                         # free GPU
        return np.vstack(all_embs).astype(np.float32)                   # (N, 768)

    @torch.inference_mode()
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Return the `top_k` passages most similar to `query`.
        """
        toks = self.q_tok(query,
                          return_tensors="pt",
                          truncation=True,
                          padding=True).to(self.device)

        q_vec = self.query_encoder(**toks).pooler_output.squeeze(0)     
        sims  = self.document_embeddings @ q_vec.cpu().numpy()          

        top_ids = np.argpartition(-sims, top_k - 1)[:top_k]             # partial sort
        top_ids = top_ids[np.argsort(-sims[top_ids])]                   # exact order
        return [self.documents[i] for i in top_ids.tolist()]
