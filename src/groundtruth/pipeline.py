"""Build the retriever stack described by a Config."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from groundtruth.chunking import Chunk, chunk_corpus
from groundtruth.config import Config
from groundtruth.corpus import corpus_hash, load_corpus
from groundtruth.retrievers.base import Hit, Reranker, Retriever
from groundtruth.retrievers.bm25 import BM25Retriever
from groundtruth.retrievers.dense import (
    DenseRetriever,
    Encoder,
    LSAEncoder,
    SentenceTransformerEncoder,
)
from groundtruth.retrievers.hybrid import HybridRetriever
from groundtruth.retrievers.rerank import CrossEncoderReranker


@dataclass
class Pipeline:
    config: Config
    docs: dict[str, str]
    chunks: list[Chunk]
    retriever: Retriever
    reranker: Reranker | None

    def search(self, query: str) -> tuple[list[Hit], list[Hit]]:
        """Return (first-stage candidates, final top-k)."""
        first = self.retriever.search(query, self.config.candidates)
        if self.reranker is not None:
            final = self.reranker.rerank(query, first, self.config.k)
        else:
            final = first[: self.config.k]
        return first, final


def _dense(cfg: Config, chash: str) -> DenseRetriever:
    encoder: Encoder
    if cfg.embedding_backend == "lsa":
        encoder = LSAEncoder()
    else:
        encoder = SentenceTransformerEncoder(cfg.embedding_model, cfg.query_prefix)
    model_slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{cfg.embedding_backend}-{cfg.embedding_model}")
    cache = Path(cfg.cache_dir) / "embeddings" / (f"{model_slug}_{cfg.chunk_size}_{cfg.overlap}_{chash}.npy")
    return DenseRetriever(encoder, cache_path=cache)


def build_pipeline(cfg: Config, docs: dict[str, str] | None = None) -> Pipeline:
    """Load the corpus (unless given), chunk it and index it with the configured retriever."""
    docs = docs if docs is not None else load_corpus(cfg.corpus_dir)
    chunks = chunk_corpus(docs, cfg.chunk_size, cfg.overlap)
    chash = corpus_hash(docs)
    retriever: Retriever
    if cfg.retriever == "dense":
        retriever = _dense(cfg, chash)
    elif cfg.retriever == "bm25":
        retriever = BM25Retriever()
    else:
        retriever = HybridRetriever([_dense(cfg, chash), BM25Retriever()], per_retriever=cfg.candidates, c=cfg.rrf_c)
    retriever.index(chunks)
    reranker = CrossEncoderReranker(cfg.reranker) if cfg.reranker else None
    return Pipeline(cfg, docs, chunks, retriever, reranker)
