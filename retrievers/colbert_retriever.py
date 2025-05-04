from __future__ import annotations
from pathlib import Path
from typing import Sequence, List

import numpy as np
import torch
import faiss  # pip install faiss-gpu
from sentence_transformers import SentenceTransformer, util


class ColBERTRetriever:
    """ColBERT‑style dense retriever that
    • Encodes passages on the first CUDA device when available (falls back to CPU).
    • Casts the model **and** embeddings to FP16 only when a GPU is present.
    • Optionally builds an in‑VRAM FAISS flat IP index for ultra‑fast top‑k search.
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
        # ─── Resolve device ---------------------------------------------------------
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_fp16 = self.device.type == "cuda"  # cast only when it makes sense

        # ─── Hyper‑parameters -------------------------------------------------------
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dim = 768  # BERT‑base hidden size
        self.use_faiss = use_faiss and torch.cuda.is_available()
        self.cache_path = Path(cache_path) if cache_path else None

        # ─── Load model -------------------------------------------------------------
        self.model = SentenceTransformer(model_name, device=str(self.device))
        if self.use_fp16:
            self.model.half()  # safer than .to(dtype=..)

        print(
            f"[ColBERTRetriever] running on {self.device} | fp16={self.use_fp16} | faiss_gpu={self.use_faiss}"
        )

        # ─── Prepare embeddings -----------------------------------------------------
        self.documents = list(documents)
        if self.cache_path and self.cache_path.exists():
            self.document_embeddings = np.load(self.cache_path, mmap_mode="r")
        else:
            self.document_embeddings = self._encode_documents(self.documents)
            if self.cache_path:
                np.save(self.cache_path, self.document_embeddings)

        # ─── Build FAISS index ------------------------------------------------------
        if self.use_faiss:
            self.index = self._build_faiss_index(self.document_embeddings)

    # ------------------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_documents(self, docs: Sequence[str]) -> np.ndarray:
        """Encode passages in parallel and return an FP16/FP32 NumPy array."""
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
        return emb.astype(np.float16 if self.use_fp16 else np.float32)

    # ------------------------------------------------------------------------------
    def _build_faiss_index(self, vecs: np.ndarray):
        """Build a GPU FAISS IndexFlatIP (inner product) if GPU present."""
        vecs32 = vecs.astype(np.float32)
        cpu_index = faiss.IndexFlatIP(self.dim)
        cpu_index.add(vecs32)
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu_index)

    # ------------------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            query,
            batch_size=1,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return vec.astype(np.float32)

    # ------------------------------------------------------------------------------
    @torch.inference_mode()
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        q_vec = self._encode_query(query)

        # Fast path: FAISS GPU index -------------------------------------------------
        if self.use_faiss:
            _, idx = self.index.search(q_vec[None, :], top_k)
            return [self.documents[i] for i in idx[0]]

        # Fallback: mat‑mul on whichever device the model sits ----------------------
        doc_emb_t = torch.from_numpy(self.document_embeddings).to(self.device)
        q_vec_t = torch.from_numpy(q_vec).to(self.device)
        sims = torch.matmul(doc_emb_t, q_vec_t)
        top_idx = torch.topk(sims, k=top_k).indices.cpu().numpy()
        return [self.documents[i] for i in top_idx.tolist()]
