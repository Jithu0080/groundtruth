"""Experiment configuration: YAML file -> validated, typed Config."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from groundtruth import GroundtruthError

RETRIEVERS = {"dense", "bm25", "hybrid"}
EMBEDDING_BACKENDS = {"sentence-transformers", "lsa"}


@dataclass(frozen=True)
class Config:
    name: str
    corpus_dir: str = "data/corpus"
    chunk_size: int = 800  # characters
    overlap: int = 100  # characters
    retriever: str = "dense"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # "lsa" exists only so tests run offline; never report its numbers.
    embedding_backend: str = "sentence-transformers"
    # None = use the known prefix for the model (see retrievers.dense.default_query_prefix)
    query_prefix: str | None = None
    reranker: str | None = None
    candidates: int = 50  # first-stage results fetched before fusion/reranking
    k: int = 10
    rrf_c: int = 60
    cache_dir: str = ".cache"

    def validate(self) -> None:
        errors = []
        if self.retriever not in RETRIEVERS:
            errors.append(f"retriever must be one of {sorted(RETRIEVERS)}, got {self.retriever!r}")
        if self.embedding_backend not in EMBEDDING_BACKENDS:
            errors.append(f"embedding_backend must be one of {sorted(EMBEDDING_BACKENDS)}")
        if self.chunk_size < 50:
            errors.append("chunk_size must be at least 50 characters")
        if not 0 <= self.overlap < self.chunk_size:
            errors.append("overlap must be >= 0 and smaller than chunk_size")
        if self.k < 1:
            errors.append("k must be >= 1")
        if self.candidates < self.k:
            errors.append("candidates must be >= k")
        if self.rrf_c < 1:
            errors.append("rrf_c must be >= 1")
        if errors:
            raise GroundtruthError(f"invalid config {self.name!r}: " + "; ".join(errors))


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config. The config name defaults to the file stem."""
    p = Path(path)
    if not p.is_file():
        raise GroundtruthError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise GroundtruthError(f"config {p} must be a YAML mapping")
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise GroundtruthError(f"config {p}: unknown keys {sorted(unknown)}")
    raw.setdefault("name", p.stem)
    cfg = Config(**raw)
    cfg.validate()
    return cfg
