"""Semantic, sentence-aware chunker (offline variant).

Uses a quick regex sentence splitter to keep sentences intact while
packing up to `chunk_char_limit` *characters* per chunk.  This avoids
breaking mid-sentence and is tokeniser-agnostic.

For heavier-weight setups you can swap in spaCy / NLTK by passing
`sentencizer=<callable>` that returns an *iterable of sentences*.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, List


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """
    Args:
        chunk_char_limit (int): Soft limit in characters.
        sentencizer (Callable[[str], Iterable[str]] | None): Optional custom
            sentence splitter.  Defaults to a simple regex.
    """

    def __init__(
        self,
        chunk_char_limit: int = 2000,
        *,
        sentencizer: Callable[[str], Iterable[str]] | None = None,
    ) -> None:
        self.char_limit = chunk_char_limit
        self.split_into_sents = sentencizer or (
            lambda txt: _SENT_SPLIT_RE.split(txt.strip())
        )

    # ------------------------------------------------------------------ #
    def chunk(self, text: str) -> List[str]:
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for sent in self.split_into_sents(text):
            sent = sent.strip()
            if not sent:
                continue

            # Flush if adding the sentence would overflow the limit
            if current_len + len(sent) + 1 > self.char_limit and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_len = [], 0

            current_chunk.append(sent)
            current_len += len(sent) + 1  # +1 for the space

        # Handle remaining fragment
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
