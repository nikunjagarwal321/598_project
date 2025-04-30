"""Fixed-length token chunker.

Splits a document into equally sized segments (`chunk_size` tokens).  If
`drop_last=True`, the final chunk is discarded when it is shorter than
`chunk_size` – useful for strict length budgets (e.g. BERT/BGE encoders).
"""
from typing import Callable, List, Sequence


class FixedChunker:
    """
    Args:
        chunk_size (int): Tokens per chunk.
        tokenizer (Callable[[str], Sequence[str]] | None): Optional
            custom tokenizer.  Must return an *iterable of tokens*.
            Defaults to `str.split`.
        detokenizer (Callable[[Sequence[str]], str] | None): Optional custom
            detokeniser.  Defaults to simple `" ".join`.
        drop_last (bool): Discard a trailing, shorter chunk.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        *,
        tokenizer: Callable[[str], Sequence[str]] | None = None,
        detokenizer: Callable[[Sequence[str]], str] | None = None,
        drop_last: bool = False,
    ) -> None:
        self.chunk_size = chunk_size
        self.tokenize = tokenizer or str.split
        self.detokenize = detokenizer or (lambda toks: " ".join(toks))
        self.drop_last = drop_last

    # --------------------------------------------------------------------- #
    def chunk(self, text: str) -> List[str]:
        tokens = list(self.tokenize(text))
        spans: List[str] = []

        for i in range(0, len(tokens), self.chunk_size):
            span_tokens = tokens[i : i + self.chunk_size]
            if self.drop_last and len(span_tokens) < self.chunk_size:
                break
            spans.append(self.detokenize(span_tokens))

        return spans
