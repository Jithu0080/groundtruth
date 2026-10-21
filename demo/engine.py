"""Search engine behind the Groundtruth demo: baseline vs production retrieval over the corpus.

Embeddings are precomputed (in Colab, on a GPU) and loaded from .npy files, so the demo
never re-embeds the corpus. Only queries are embedded at request time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from groundtruth.chunking import Chunk, chunk_corpus
from groundtruth.corpus import load_corpus
from groundtruth.golden import GoldenQuery, load_golden
from groundtruth.metrics import match_gold
from groundtruth.retrievers.base import Hit, Reranker
from groundtruth.retrievers.bm25 import BM25Retriever
from groundtruth.retrievers.dense import DenseRetriever, Encoder
from groundtruth.retrievers.hybrid import HybridRetriever

MAX_QUERY_CHARS = 300
K = 10
CANDIDATES = 50


@dataclass
class SystemResult:
    hits: list[Hit]
    latency_ms: float
    gold_rank: int | None  # 1-based rank of the first chunk covering a gold span; None if not in top K


@dataclass
class Comparison:
    query: str
    golden: GoldenQuery | None
    baseline: SystemResult
    production: SystemResult


def _norm(q: str) -> str:
    return " ".join(q.lower().split()).strip(" ?.")


def _dense_from_matrix(encoder: Encoder, chunks: list[Chunk], path: Path) -> DenseRetriever:
    matrix = np.load(path).astype(np.float32)
    if matrix.shape[0] != len(chunks):
        raise ValueError(f"{path.name} has {matrix.shape[0]} rows but the corpus gives {len(chunks)} chunks")
    r = DenseRetriever(encoder)
    r.chunks, r.matrix = chunks, matrix
    return r


class Engine:
    """Baseline = dense, 800-char chunks. Production = hybrid (dense + BM25, RRF) + cross-encoder, 400-char chunks."""

    def __init__(self, data_dir: str | Path, encoder: Encoder, reranker: Reranker):
        d = Path(data_dir)
        self.docs = load_corpus(d / "corpus")
        chunks_400 = chunk_corpus(self.docs, 400, 50)
        chunks_800 = chunk_corpus(self.docs, 800, 100)
        self.baseline_dense = _dense_from_matrix(encoder, chunks_800, d / "emb_800.npy")
        dense_400 = _dense_from_matrix(encoder, chunks_400, d / "emb_400.npy")
        bm25 = BM25Retriever()
        bm25.index(chunks_400)
        self.hybrid = HybridRetriever([dense_400, bm25], per_retriever=CANDIDATES)
        self.reranker = reranker
        golden = load_golden(d / "golden_v1.jsonl", self.docs, "all")
        self.golden = {_norm(g.query): g for g in golden if g.answerable}
        ex_path = d / "examples.json"
        self.examples: list[dict] = json.loads(ex_path.read_text(encoding="utf-8")) if ex_path.is_file() else []
        self.n_chunks = {"baseline": len(chunks_800), "production": len(chunks_400)}

    def _gold_rank(self, hits: list[Hit], golden: GoldenQuery | None) -> int | None:
        if golden is None:
            return None
        for rank, m in enumerate(match_gold([c for c, _ in hits], golden.relevant), start=1):
            if m:
                return rank
        return None

    def compare(self, query: str) -> Comparison:
        q = " ".join(query.split())[:MAX_QUERY_CHARS]
        if not q:
            raise ValueError("type a question")
        golden = self.golden.get(_norm(q))

        t0 = time.perf_counter()
        base_hits = self.baseline_dense.search(q, K)
        base_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        prod_hits = self.reranker.rerank(q, self.hybrid.search(q, CANDIDATES), K)
        prod_ms = (time.perf_counter() - t0) * 1000

        return Comparison(
            query=q,
            golden=golden,
            baseline=SystemResult(base_hits, base_ms, self._gold_rank(base_hits, golden)),
            production=SystemResult(prod_hits, prod_ms, self._gold_rank(prod_hits, golden)),
        )
