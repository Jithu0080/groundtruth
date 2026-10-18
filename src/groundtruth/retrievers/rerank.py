"""Cross-encoder reranking of first-stage candidates."""

from __future__ import annotations

import numpy as np

from groundtruth import GroundtruthError
from groundtruth.retrievers.base import Hit


class CrossEncoderReranker:
    """Reads (query, chunk) together, so it is accurate but too slow for the whole corpus.

    It can only reorder what the first stage fetched: first-stage recall is its ceiling.
    """

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover - depends on environment
            raise GroundtruthError("sentence-transformers is not installed: pip install -r requirements.txt") from e
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = np.asarray(self.model.predict([(query, chunk.text) for chunk, _ in hits]))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(hits[i][0], float(scores[i])) for i in order]
