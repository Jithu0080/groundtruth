"""Hybrid retrieval with Reciprocal Rank Fusion (Cormack et al., 2009)."""

from __future__ import annotations

from groundtruth.chunking import Chunk
from groundtruth.retrievers.base import Hit, Retriever


def rrf(result_lists: list[list[Hit]], k: int, c: int = 60) -> list[Hit]:
    """RRF(d) = sum over lists of 1 / (c + rank(d)). Uses ranks only, so score scales don't matter.

    Ties are broken by chunk_id so the output is deterministic.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, (chunk, _score) in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (c + rank)
            by_id[chunk.chunk_id] = chunk
    best = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
    return [(by_id[cid], scores[cid]) for cid in best]


class HybridRetriever:
    def __init__(self, retrievers: list[Retriever], per_retriever: int = 50, c: int = 60):
        self.retrievers = retrievers
        self.per_retriever = per_retriever
        self.c = c

    def index(self, chunks: list[Chunk]) -> None:
        for r in self.retrievers:
            r.index(chunks)

    def search(self, query: str, k: int) -> list[Hit]:
        n = max(k, self.per_retriever)
        return rrf([r.search(query, n) for r in self.retrievers], k=k, c=self.c)
