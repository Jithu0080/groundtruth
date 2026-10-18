"""Grounded answer generation and generation metrics (faithfulness, answer relevance)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from groundtruth.chunking import Chunk
from groundtruth.golden import load_golden
from groundtruth.llm import LLM
from groundtruth.pipeline import Pipeline

REFUSAL = "Not found in the provided sources."
N_CONTEXT = 5

ANSWER_PROMPT = """Answer the question using only the numbered sources below.
Cite every claim with its source number, like [2].
If the sources do not contain the answer, reply exactly: "{refusal}"

Sources:
{sources}

Question: {question}"""

FAITHFULNESS_PROMPT = """You check whether an answer is supported by source passages.

Sources:
{sources}

Answer:
{answer}

1. List every factual claim in the answer.
2. Mark each claim SUPPORTED if the sources state it, otherwise UNSUPPORTED.
Return JSON only: {{"claims": [{{"claim": "...", "verdict": "SUPPORTED"}}]}}"""

RELEVANCE_PROMPT = """Question: {question}
Answer: {answer}

Score how directly the answer addresses the question:
5 = fully answers it, 3 = partly, 1 = does not address it.
Return JSON only: {{"score": <1-5>, "reason": "..."}}"""


class JudgeError(ValueError):
    """The judge returned something that is not the JSON we asked for."""


def format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))


def parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model reply (models sometimes wrap it in prose/fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise JudgeError("no JSON object in judge reply")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeError(f"invalid JSON in judge reply: {e.msg}") from e
    if not isinstance(obj, dict):
        raise JudgeError("judge reply is not a JSON object")
    return obj


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower().rstrip(".") in answer.lower()


def generate_answer(llm: LLM, question: str, chunks: list[Chunk]) -> str:
    return llm.complete(ANSWER_PROMPT.format(refusal=REFUSAL, sources=format_sources(chunks), question=question))


def faithfulness(judge: LLM, answer: str, chunks: list[Chunk]) -> float | None:
    """Supported claims / all claims. None when the answer makes no claims (e.g. a refusal)."""
    obj = parse_json(judge.complete(FAITHFULNESS_PROMPT.format(sources=format_sources(chunks), answer=answer), 1500))
    claims = obj.get("claims")
    if not isinstance(claims, list):
        raise JudgeError("judge reply has no 'claims' list")
    if not claims:
        return None
    verdicts = [str(c.get("verdict", "")).upper() for c in claims if isinstance(c, dict)]
    if len(verdicts) != len(claims) or any(v not in {"SUPPORTED", "UNSUPPORTED"} for v in verdicts):
        raise JudgeError("judge returned an unknown verdict")
    return sum(v == "SUPPORTED" for v in verdicts) / len(verdicts)


def answer_relevance(judge: LLM, question: str, answer: str) -> float:
    """Judge's 1-5 score rescaled to 0-1."""
    obj = parse_json(judge.complete(RELEVANCE_PROMPT.format(question=question, answer=answer), 300))
    score = obj.get("score")
    if not isinstance(score, (int, float)) or not 1 <= score <= 5:
        raise JudgeError("judge score must be a number from 1 to 5")
    return (float(score) - 1) / 4


def run_generation_eval(
    pipe: Pipeline, golden_path: str | Path, split: str, generator: LLM, judge: LLM, limit: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    queries = load_golden(golden_path, pipe.docs, split)
    if limit is not None:
        queries = queries[:limit]
    rows = []
    for q in queries:
        _, final = pipe.search(q.query)
        ctx = [c for c, _ in final[:N_CONTEXT]]
        row: dict[str, Any] = {"qid": q.qid, "category": q.category, "answerable": q.answerable}
        try:
            answer = generate_answer(generator, q.query, ctx)
        except Exception as e:  # noqa: BLE001 - record the failure, keep the run going
            rows.append({**row, "error": f"generation: {e}"})
            continue
        row.update(answer=answer, refused=is_refusal(answer))
        if not row["refused"]:
            try:
                row["faithfulness"] = faithfulness(judge, answer, ctx)
                row["relevance"] = answer_relevance(judge, q.query, answer)
            except Exception as e:  # noqa: BLE001 - judge failures are counted, not scored
                row["error"] = f"judge: {e}"
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ["faithfulness", "relevance", "refused", "error"]:
        if col not in df:
            df[col] = None
    ans = df[df["answerable"]]
    unans = df[~df["answerable"]]
    summary = {
        "golden": Path(golden_path).as_posix(),  # same string on Windows and Linux CI
        "split": split,
        "n_queries": int(len(df)),
        "faithfulness": _mean(ans["faithfulness"]),
        "answer_relevance": _mean(ans["relevance"]),
        "answered_rate": _mean(~ans["refused"].fillna(True).astype(bool)),
        "unanswerable_refusal_rate": _mean(unans["refused"].fillna(False).astype(bool)) if len(unans) else None,
        "errors": int(df["error"].notna().sum()),
    }
    return df, summary


def _mean(s: pd.Series) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return round(float(s.mean()), 4) if len(s) else None
