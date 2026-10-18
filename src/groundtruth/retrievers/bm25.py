"""BM25 (Okapi) over a sparse term-frequency matrix."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from groundtruth import GroundtruthError
from groundtruth.chunking import Chunk
from groundtruth.retrievers.base import Hit

TOKEN_PATTERN = r"(?u)\b\w+\b"  # noqa: S105 - a regex, not a password. Keeps single characters and numbers such as section "138"


class BM25Retriever:
    """score(q, d) = sum over query terms t of
    IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

    IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5)), which is never negative.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True)
        tf = self.vectorizer.fit_transform([c.text for c in chunks]).tocsc().astype(np.float64)
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = max(doc_len.mean(), 1e-9)
        df = np.diff(tf.indptr)  # documents containing each term (CSC column lengths)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        self.norm = self.k1 * (1 - self.b + self.b * doc_len / avgdl)
        self.tf = tf

    def search(self, query: str, k: int) -> list[Hit]:
        if not self.chunks:
            raise GroundtruthError("BM25Retriever.search called before index()")
        q = self.vectorizer.transform([query])
        scores = np.zeros(len(self.chunks))
        for term, count in zip(q.indices, q.data, strict=True):  # repeated query words count again
            lo, hi = self.tf.indptr[term], self.tf.indptr[term + 1]
            rows, f = self.tf.indices[lo:hi], self.tf.data[lo:hi]
            scores[rows] += count * self.idf[term] * f * (self.k1 + 1) / (f + self.norm[rows])
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunks[i], float(scores[i])) for i in top]
