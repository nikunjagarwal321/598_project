from typing import List

class NoReranker:
    def rerank(self, query: str, docs: List[str]) -> List[str]:
        # No re-ranking, just return as-is
        return docs