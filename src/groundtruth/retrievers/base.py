"""Shared retriever types."""

from __future__ import annotations

from typing import Protocol

from groundtruth.chunking import Chunk

Hit = tuple[Chunk, float]  # (chunk, score), lists of hits are sorted by score descending


class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[Hit]: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...
