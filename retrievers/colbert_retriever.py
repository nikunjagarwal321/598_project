from __future__ import annotations
from pathlib import Path
from typing import Sequence, List

import numpy as np
import torch
import faiss                     # pip install faiss-gpu
from sentence_transformers import SentenceTransformer, util


class ColBERTRetriever:
    """
    ColBERT‑style retriever that

    • encodes passages with SentenceTransformers on GPU (FP16)
    • parallelises tokenisation with `encode_multi_process`
    • stores embeddings in either
        - a FAISS GPU index   (fastest retrieval)
        - a torch / NumPy tensor on GPU (fallback)
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
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                              else "cpu"))
        self.batch_size  = batch_size
        self.use_faiss   = use_faiss
        self.dim         = 768                       # BERT‑base hidden size
        self.cache_path  = Path(cache_path) if cache_path else None

        # ─── Load model in FP16 on GPU ──────────────────────────────────────
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model.to(torch.float16)          # convert weights to FP16 in‑place


        # ─── Encode or load embeddings ─────────────────────────────────────
        self.documents = list(documents)
        if self.cache_path and self.cache_path.exists():
            self.document_embeddings = np.load(self.cache_path, mmap_mode="r")
        else:
            self.document_embeddings = self._encode_documents(documents,
                                                              num_workers)
            if self.cache_path:
                np.save(self.cache_path, self.document_embeddings)

        # ─── Build FAISS index (optional) ───────────────────────────────────
        if self.use_faiss:
            self.index = self._build_faiss_index(self.document_embeddings)

    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_documents(self, docs: Sequence[str], n_proc: int) -> np.ndarray:
        """FP16 GPU encoding + multi‑process tokenisation."""
        return util.encode_multi_process(
            sentences         = docs,
            model             = self.model,
            batch_size        = self.batch_size,
            chunk_size        = 5000,
            num_processes     = n_proc,
            normalize_embeddings=False,
            convert_to_numpy  = True,       # already returns FP16 np array
            show_progress_bar = True,
        ).astype(np.float16)

    # -----------------------------------------------------------------------
    def _build_faiss_index(self, vecs: np.ndarray):
        """Create a GPU FAISS index (flat L2) for dot‑product search."""
        vecs = vecs.astype(np.float32)          # FAISS expects FP32
        index_cpu = faiss.IndexFlatIP(self.dim)
        index_cpu.add(vecs)
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0
        return index

    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query,
                                 batch_size=1,
                                 device=self.device,
                                 convert_to_numpy=True,
                                 normalize_embeddings=False).astype(np.float32)

    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Return top‑k passages most similar to `query`."""
        q_vec = self._encode_query(query)

        if self.use_faiss:
            sim, idx = self.index.search(q_vec[None, :], top_k)
            return [self.documents[i] for i in idx[0]]

        # fallback: GPU matmul
        doc_emb = torch.from_numpy(self.document_embeddings).to(self.device)
        q = torch.from_numpy(q_vec).to(self.device)
        sims = torch.matmul(doc_emb, q)              # (N,)
        top  = torch.topk(sims, k=top_k).indices.cpu().numpy()
        return [self.documents[i] for i in top.tolist()]
