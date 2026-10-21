"""The demo's search engine, on the fictional fixture corpus with an offline encoder and a pass-through reranker."""

import json
import shutil
import sys

import numpy as np
import pytest

from conftest import FIXTURES, ROOT
from groundtruth.chunking import chunk_corpus
from groundtruth.corpus import load_corpus
from groundtruth.retrievers.dense import LSAEncoder

sys.path.insert(0, str(ROOT / "demo"))
from engine import Engine  # noqa: E402


class PassThroughReranker:
    def rerank(self, query, hits, k):
        return hits[:k]


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo_data")
    shutil.copytree(FIXTURES / "mini_corpus", d / "corpus")
    shutil.copy(FIXTURES / "golden_mini.jsonl", d / "golden_v1.jsonl")
    docs = load_corpus(d / "corpus")
    enc = LSAEncoder(n_components=16)
    enc.fit([c.text for c in chunk_corpus(docs, 400, 50)])
    for size, overlap, name in [(400, 50, "emb_400.npy"), (800, 100, "emb_800.npy")]:
        chunks = chunk_corpus(docs, size, overlap)
        np.save(d / name, enc.encode([c.text for c in chunks], is_query=False).astype(np.float16))
    (d / "examples.json").write_text(json.dumps([{"qid": "m01", "query": "x", "tag": "both"}]))
    return Engine(d, encoder=enc, reranker=PassThroughReranker())


def test_golden_question_gets_ranks(engine):
    c = engine.compare("Is serving a demand notice on the drawer required before prosecuting a bounced cheque? ")
    assert c.golden is not None and c.golden.qid == "m01"
    assert len(c.baseline.hits) <= 10 and len(c.production.hits) <= 10
    assert c.production.gold_rank is None or 1 <= c.production.gold_rank <= 10
    assert c.baseline.latency_ms >= 0


def test_free_text_question_has_no_gold(engine):
    c = engine.compare("what about tenancy notice periods")
    assert c.golden is None and c.production.gold_rank is None
    assert c.production.hits


def test_empty_and_long_queries(engine):
    with pytest.raises(ValueError):
        engine.compare("   ")
    assert len(engine.compare("cheque " * 200).query) <= 300


def test_examples_loaded(engine):
    assert engine.examples and engine.examples[0]["qid"] == "m01"


def test_mismatched_embeddings_rejected(tmp_path):
    shutil.copytree(FIXTURES / "mini_corpus", tmp_path / "corpus")
    shutil.copy(FIXTURES / "golden_mini.jsonl", tmp_path / "golden_v1.jsonl")
    np.save(tmp_path / "emb_400.npy", np.zeros((3, 4), dtype=np.float16))
    np.save(tmp_path / "emb_800.npy", np.zeros((3, 4), dtype=np.float16))
    with pytest.raises(ValueError):
        Engine(tmp_path, encoder=LSAEncoder(), reranker=PassThroughReranker())
