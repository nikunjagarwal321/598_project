from __future__ import annotations
from pathlib import Path
from typing import Sequence, List

import numpy as np
import torch
import faiss  # pip install faiss-gpu
from sentence_transformers import SentenceTransformer, util


class ColBERTRetriever:
    """GPU‑friendly ColBERT‑style retriever.

    Highlights
    ----------
    • Works with *any* sentence‑transformers version: we no longer pass unsupported
      kwargs like `dtype` or `device` to the constructor.
    • Automatically selects CUDA when available and casts the entire model +
      embeddings to FP16 on GPU for speed and memory savings.
    • Optional FAISS GPU index for ~10× faster similarity search.
    """

    def __init__(
        self,
        documents: Sequence[str],
        model_name: str = "bert-base-nli-mean-tokens",
        batch_size: int = 128,
        num_workers: int = 4,
        cache_path: str | None = None,
        use_faiss: bool = True,
        device: str | None = None,
    ) -> None:
        # ─── Device & precision ----------------------------------------------------
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = self.device.type == "cuda"

        # ─── Hyper‑parameters ------------------------------------------------------
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dim = 768  # BERT‑base hidden size
        self.cache_path = Path(cache_path) if cache_path else None
        self.use_faiss = bool(use_faiss and self.device.type == "cuda")

        # ─── Load model ------------------------------------------------------------
        # Do *not* pass version‑specific kwargs; move & cast afterwards.
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        if self.fp16:
            self.model.half()

        print(
            f"[ColBERTRetriever] device={self.device} | fp16={self.fp16} | faiss_gpu={self.use_faiss}"
        )

        # ─── Encode / load passage embeddings -------------------------------------
        self.documents = list(documents)
        if self.cache_path and self.cache_path.exists():
            self.document_embeddings = np.load(self.cache_path, mmap_mode="r")
        else:
            self.document_embeddings = self._encode_documents(self.documents)
            if self.cache_path:
                np.save(self.cache_path, self.document_embeddings)

        # ─── Build FAISS GPU index -------------------------------------------------
        if self.use_faiss:
            self.index = self._build_faiss_index(self.document_embeddings)

    # -----------------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_documents(self, docs: Sequence[str]) -> np.ndarray:
        """Tokenise in parallel and encode in batches on `self.device`."""
        emb = util.encode_multi_process(
            sentences=docs,
            model=self.model,
            batch_size=self.batch_size,
            chunk_size=5000,
            num_processes=self.num_workers,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        return emb.astype(np.float16 if self.fp16 else np.float32)

    # -----------------------------------------------------------------------------
    def _build_faiss_index(self, vecs: np.ndarray):
        vecs32 = vecs.astype(np.float32)
        cpu_idx = faiss.IndexFlatIP(self.dim)
        cpu_idx.add(vecs32)
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu_idx)

    # -----------------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_query(self, query: str) -> np.ndarray:
        q = self.model.encode(
            query,
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return q.astype(np.float32)

    # -----------------------------------------------------------------------------
    @torch.inference_mode()
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        q_vec = self._encode_query(query)

        # --- FAISS fast path ------------------------------------------------------
        if self.use_faiss:
            _, idx = self.index.search(q_vec[None, :], top_k)
            return [self.documents[i] for i in idx[0]]

        # --- Fallback mat‑mul -----------------------------------------------------
        doc_emb = torch.from_numpy(self.document_embeddings).to(self.device)
        q_vec_t = torch.from_numpy(q_vec).to(self.device)
        sims = torch.matmul(doc_emb, q_vec_t)  # (N,)
        top = torch.topk(sims, k=top_k).indices.cpu().numpy()
        return [self.documents[i] for i in top.tolist()]