"""Generation-eval logic with a fake LLM (no network)."""

import json

import pytest

from conftest import FIXTURES
from groundtruth import GroundtruthError
from groundtruth.config import load_config
from groundtruth.generation import (
    REFUSAL,
    JudgeError,
    answer_relevance,
    faithfulness,
    parse_json,
    run_generation_eval,
)
from groundtruth.golden import load_golden
from groundtruth.llm import LLMClient
from groundtruth.pipeline import build_pipeline


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, max_tokens=800):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_parse_json_handles_fences_and_rejects_garbage():
    assert parse_json('Sure!\n```json\n{"score": 4}\n```') == {"score": 4}
    with pytest.raises(JudgeError):
        parse_json("no json here")


def test_faithfulness_fraction_and_refusal():
    claims = {"claims": [{"claim": "a", "verdict": "SUPPORTED"}, {"claim": "b", "verdict": "UNSUPPORTED"}]}
    assert faithfulness(FakeLLM([json.dumps(claims)]), "answer", []) == 0.5
    assert faithfulness(FakeLLM(['{"claims": []}']), REFUSAL, []) is None
    with pytest.raises(JudgeError):
        faithfulness(FakeLLM(['{"claims": [{"claim": "a", "verdict": "MAYBE"}]}']), "x", [])


def test_relevance_rescaled_and_validated():
    assert answer_relevance(FakeLLM(['{"score": 5}']), "q", "a") == 1.0
    assert answer_relevance(FakeLLM(['{"score": 1}']), "q", "a") == 0.0
    with pytest.raises(JudgeError):
        answer_relevance(FakeLLM(['{"score": 9}']), "q", "a")


def test_run_generation_eval_counts_errors_and_refusals(monkeypatch):
    import groundtruth.generation as g

    pipe = build_pipeline(load_config(FIXTURES / "configs" / "smoke_bm25.yaml"))
    golden = FIXTURES / "golden_mini.jsonl"
    queries = load_golden(golden, pipe.docs, "all")
    probe = next(q for q in queries if not q.answerable)
    # 3 queries: answered + judged; judge returns garbage; unanswerable probe is refused
    monkeypatch.setattr(g, "load_golden", lambda p, d, s: [queries[0], queries[1], probe])
    gen = FakeLLM(["Notice is mandatory [1].", "Fifteen days [1].", REFUSAL])
    judge = FakeLLM(
        [
            '{"claims": [{"claim": "notice mandatory", "verdict": "SUPPORTED"}]}',
            '{"score": 5}',
            "the judge rambled",
        ]
    )
    df, summary = run_generation_eval(pipe, golden, "all", gen, judge)
    assert summary["faithfulness"] == 1.0
    assert summary["answer_relevance"] == 1.0
    assert summary["errors"] == 1
    assert summary["unanswerable_refusal_rate"] == 1.0
    assert "[1]" in gen.prompts[0] and "Question:" in gen.prompts[0]


def test_llm_client_config_errors():
    with pytest.raises(GroundtruthError):
        LLMClient("anthropic", "", "key")
    with pytest.raises(GroundtruthError):
        LLMClient("anthropic", "some-model", None)
    with pytest.raises(GroundtruthError):
        LLMClient("cohere", "m", "k")


def test_llm_client_refuses_plain_http_remote():
    c = LLMClient("openai", "m", None, base_url="http://evil.example.com/v1")
    with pytest.raises(GroundtruthError):
        c.complete("hi")
