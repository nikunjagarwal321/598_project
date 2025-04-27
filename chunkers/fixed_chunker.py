from typing import List

class FixedChunker:
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size

    def chunk(self, document: str) -> List[str]:
        words = document.split()
        return [" ".join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
