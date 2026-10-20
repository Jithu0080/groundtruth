"""Golden-set labelling helpers used by the Colab labelling tool.

Drafts are appended to a JSONL file one line per saved question, so work survives disconnects and
the tool resumes where you stopped. Skipped candidates are remembered in a side file.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from groundtruth import GroundtruthError
from groundtruth.golden import locate_answer, read_jsonl

CATEGORIES = ["holding", "statute", "factual", "procedural", "paraphrase"]
_STOP = set(
    "the a an of to in on for and or is are was were be by with that this which what when how does did do "
    "can could should would whether under any from as at it its their there been has have had not no if".split()
)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def wording_overlap(question: str, passage: str) -> float:
    """Share of the question's content words that also appear in the passage (0..1).

    High overlap makes keyword search look better than it is, so the tool warns above 0.6.
    """
    q = _words(question)
    return len(q & _words(passage)) / len(q) if q else 0.0


class LabelStore:
    def __init__(self, candidates_path: str | Path, drafts_path: str | Path, reviewer: str):
        if not reviewer.strip():
            raise GroundtruthError("set your name as the reviewer")
        self.candidates = read_jsonl(candidates_path)
        self.drafts_path = Path(drafts_path)
        self.skipped_path = self.drafts_path.with_name(self.drafts_path.stem + "_skipped.txt")
        self.reviewer = reviewer.strip()
        self.drafts = read_jsonl(self.drafts_path) if self.drafts_path.is_file() else []
        self.skipped = set(self.skipped_path.read_text().split()) if self.skipped_path.is_file() else set()

    @property
    def done_cids(self) -> set[str]:
        return {str(d.get("cid")) for d in self.drafts} | self.skipped

    def next_candidate(self) -> dict | None:
        return next((c for c in self.candidates if c["cid"] not in self.done_cids), None)

    def save(self, cand: dict, question: str, category: str, answer_text: str = "") -> dict:
        """Validate and append one golden draft; returns the saved record."""
        question = " ".join(question.split())
        if len(question) < 10:
            raise GroundtruthError("write a full question (at least 10 characters)")
        if category not in CATEGORIES:
            raise GroundtruthError(f"category must be one of {CATEGORIES}")
        if any(" ".join(d["query"].lower().split()) == question.lower() for d in self.drafts):
            raise GroundtruthError("you already saved this exact question")
        start, end = locate_answer(cand, answer_text)
        if end - start < 20:
            raise GroundtruthError("the answer text is too short; include the full sentence")
        rec = {
            "qid": f"q{len(self.drafts) + 1:03d}",
            "query": question,
            "category": category,
            "relevant": [{"doc_id": cand["doc_id"], "start": start, "end": end}],
            "reviewed_by": self.reviewer,
            "cid": cand["cid"],
            "notes": "",
        }
        self.drafts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.drafts_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.drafts.append(rec)
        return rec

    def skip(self, cand: dict) -> None:
        self.skipped.add(cand["cid"])
        with open(self.skipped_path, "a", encoding="utf-8") as fh:
            fh.write(cand["cid"] + "\n")

    def progress(self) -> dict[str, int]:
        counts = Counter(d["category"] for d in self.drafts)
        return {c: counts.get(c, 0) for c in CATEGORIES}

    def add_unanswerable(self, questions: list[str]) -> int:
        """Append unanswerable probes (answers not in the corpus) for the generation eval."""
        existing = {" ".join(d["query"].lower().split()) for d in self.drafts}
        added = 0
        with open(self.drafts_path, "a", encoding="utf-8") as fh:
            for q in questions:
                q = " ".join(q.split())
                if not q or q.lower() in existing:
                    continue
                rec = {
                    "qid": f"u{sum(1 for d in self.drafts if not d.get('answerable', True)) + 1:03d}",
                    "query": q,
                    "category": "unanswerable",
                    "relevant": [],
                    "answerable": False,
                    "reviewed_by": self.reviewer,
                    "notes": "answer is not in the corpus",
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.drafts.append(rec)
                existing.add(q.lower())
                added += 1
        return added
