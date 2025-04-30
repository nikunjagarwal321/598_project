"""Sliding-window chunker with token overlap.

Useful when generation models need additional left/right context so that
answers do not straddle chunk boundaries.
"""
from typing import Callable, List, Sequence


class OverlappingChunker:
    """
    Args:
        chunk_size (int): Total tokens per window.
        overlap (int):  How many tokens each window *shares* with the previous
            one.  Must satisfy `0 <= overlap < chunk_size`.
        tokenizer / detokenizer: Same contract as in `FixedChunker`.
        drop_last (bool): Whether to keep a tail chunk shorter than
            `chunk_size`.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 128,
        *,
        tokenizer: Callable[[str], Sequence[str]] | None = None,
        detokenizer: Callable[[Sequence[str]], str] | None = None,
        drop_last: bool = False,
    ) -> None:
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 ≤ overlap < chunk_size")
        self.chunk_size, self.overlap = chunk_size, overlap
        self.tokenize = tokenizer or str.split
        self.detokenize = detokenizer or (lambda toks: " ".join(toks))
        self.drop_last = drop_last

    # ------------------------------------------------------------------ #
    def chunk(self, text: str) -> List[str]:
        tokens = list(self.tokenize(text))
        stride = self.chunk_size - self.overlap
        spans: List[str] = []

        for start in range(0, len(tokens), stride):
            span_tokens = tokens[start : start + self.chunk_size]
            if len(span_tokens) < self.chunk_size and self.drop_last:
                break
            spans.append(self.detokenize(span_tokens))

            if len(span_tokens) < self.chunk_size:  # reached the tail
                break

        return spans
