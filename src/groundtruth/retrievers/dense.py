"""Dense (embedding) retrieval with exact cosine search and an on-disk embedding cache."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from groundtruth import GroundtruthError
from groundtruth.chunking import Chunk
from groundtruth.retrievers.base import Hit

# Query instructions the models were trained with. Check the model card for any model you add.
KNOWN_QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def default_query_prefix(model_name: str) -> str:
    return KNOWN_QUERY_PREFIXES.get(model_name, "")


class Encoder(Protocol):
    def fit(self, texts: list[str]) -> None: ...

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Real embeddings. Vectors are L2-normalised so dot product == cosine similarity."""

    def __init__(self, model_name: str, query_prefix: str | None = None, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on environment
            raise GroundtruthError("sentence-transformers is not installed: pip install -r requirements.txt") from e
        self.model = SentenceTransformer(model_name)
        self.prefix = default_query_prefix(model_name) if query_prefix is None else query_prefix
        self.batch_size = batch_size

    def fit(self, texts: list[str]) -> None:
        """Pretrained model: nothing to fit."""

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        if is_query:
            texts = [self.prefix + t for t in texts]
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=not is_query and len(texts) > 200,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


class LSAEncoder:
    """TF-IDF + truncated SVD. Offline stand-in for tests and smoke runs ONLY.

    It needs no download, but it is not a real embedding model: never report its numbers.
    """

    def __init__(self, n_components: int = 128):
        self.n_components = n_components

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(sublinear_tf=True)
        tfidf = self.vectorizer.fit_transform(texts)
        n = max(2, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n, random_state=0)
        self.svd.fit(tfidf)

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        vecs = self.svd.transform(self.vectorizer.transform(texts))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return np.asarray(vecs / np.maximum(norms, 1e-12), dtype=np.float32)


class DenseRetriever:
    def __init__(self, encoder: Encoder, cache_path: Path | None = None):
        self.encoder = encoder
        self.cache_path = cache_path
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.encoder.fit(texts)
        if self.cache_path is not None and self.cache_path.is_file():
            matrix = np.load(self.cache_path)
            if matrix.shape[0] == len(chunks):
                self.matrix = matrix
                return
        self.matrix = self.encoder.encode(texts, is_query=False)
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, self.matrix)

    def search(self, query: str, k: int) -> list[Hit]:
        if self.matrix is None:
            raise GroundtruthError("DenseRetriever.search called before index()")
        q = self.encoder.encode([query], is_query=True)[0]
        scores = self.matrix @ q  # cosine similarity, because all vectors have length 1
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunks[i], float(scores[i])) for i in top]
