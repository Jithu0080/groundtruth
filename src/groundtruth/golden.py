"""Golden set: loading, validation and candidate sampling."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from groundtruth import GroundtruthError
from groundtruth.metrics import GoldSpan

MIN_PER_CATEGORY = 15
SPLITS = {"dev", "test"}


@dataclass
class GoldenQuery:
    qid: str
    query: str
    category: str
    relevant: list[GoldSpan]
    split: str
    reviewed_by: str
    answerable: bool = True
    notes: str = ""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise GroundtruthError(f"file not found: {p}")
    records = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise GroundtruthError(f"{p}:{lineno}: invalid JSON ({e.msg})") from e
        if not isinstance(obj, dict):
            raise GroundtruthError(f"{p}:{lineno}: each line must be a JSON object")
        records.append(obj)
    return records


def validate_records(records: list[dict], docs: dict[str, str]) -> ValidationReport:
    """Check every record against the corpus. Collects all problems instead of stopping at one."""
    rep = ValidationReport()
    seen_qids: set[str] = set()
    seen_queries: dict[str, str] = {}
    categories: Counter[str] = Counter()
    required = ["qid", "query", "category", "relevant", "split", "reviewed_by"]
    for i, r in enumerate(records, start=1):
        qid = str(r.get("qid", f"<line {i}>"))
        missing = [f for f in required if f not in r]
        if missing:
            rep.errors.append(f"{qid}: missing fields {missing}")
            continue
        if qid in seen_qids:
            rep.errors.append(f"{qid}: duplicate qid")
        seen_qids.add(qid)
        q = str(r["query"]).strip()
        if not q:
            rep.errors.append(f"{qid}: empty query")
        key = " ".join(q.lower().split())
        if key in seen_queries:
            rep.errors.append(f"{qid}: same query text as {seen_queries[key]}")
        seen_queries[key] = qid
        if r["split"] not in SPLITS:
            rep.errors.append(f"{qid}: split must be 'dev' or 'test'")
        if not str(r["reviewed_by"]).strip():
            rep.errors.append(f"{qid}: reviewed_by is empty")
        answerable = r.get("answerable", True)
        rel = r["relevant"]
        if not isinstance(rel, list):
            rep.errors.append(f"{qid}: relevant must be a list")
            continue
        if answerable and not rel:
            rep.errors.append(f"{qid}: answerable query needs at least one relevant span")
        if not answerable and rel:
            rep.errors.append(f"{qid}: unanswerable query must have an empty relevant list")
        for span in rel:
            if not isinstance(span, dict) or not {"doc_id", "start", "end"} <= set(span):
                rep.errors.append(f"{qid}: each relevant item needs doc_id, start, end")
                continue
            doc = docs.get(span["doc_id"])
            if doc is None:
                rep.errors.append(f"{qid}: doc_id {span['doc_id']} not in corpus")
                continue
            s, e = span["start"], span["end"]
            if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(doc)):
                rep.errors.append(f"{qid}: span {s}-{e} outside document {span['doc_id']} (length {len(doc)})")
            elif e - s < 20:
                rep.warnings.append(f"{qid}: span is only {e - s} characters")
        if answerable:
            categories[str(r["category"])] += 1
    for cat, n in sorted(categories.items()):
        if n < MIN_PER_CATEGORY:
            rep.warnings.append(f"category {cat!r} has {n} queries (target >= {MIN_PER_CATEGORY})")
    return rep


def load_golden(path: str | Path, docs: dict[str, str], split: str | None = None) -> list[GoldenQuery]:
    """Load and validate the golden set; raise on any error. `split=None` returns all."""
    records = read_jsonl(path)
    rep = validate_records(records, docs)
    if not rep.ok:
        shown = "\n  ".join(rep.errors[:30])
        more = f"\n  ... and {len(rep.errors) - 30} more" if len(rep.errors) > 30 else ""
        raise GroundtruthError(f"golden set {path} is invalid:\n  {shown}{more}")
    queries = [
        GoldenQuery(
            qid=str(r["qid"]),
            query=str(r["query"]),
            category=str(r["category"]),
            relevant=[GoldSpan(doc_id=s["doc_id"], start=s["start"], end=s["end"]) for s in r["relevant"]],
            split=str(r["split"]),
            reviewed_by=str(r["reviewed_by"]),
            answerable=bool(r.get("answerable", True)),
            notes=str(r.get("notes", "")),
        )
        for r in records
    ]
    if split is not None:
        if split not in SPLITS | {"all"}:
            raise GroundtruthError("split must be dev, test or all")
        if split != "all":
            queries = [q for q in queries if q.split == split]
    if not queries:
        raise GroundtruthError(f"golden set {path} has no queries for split {split!r}")
    return queries


def sample_candidates(
    docs: dict[str, str], n: int = 150, passage_chars: int = 600, max_per_doc: int = 2, seed: int = 0
) -> list[dict]:
    """Randomly sample passages spread across many documents, snapped to sentence boundaries.

    You then write a question for each (in different words from the passage) and review it.
    """
    rng = random.Random(seed)
    doc_ids = [d for d in sorted(docs) if len(docs[d]) > passage_chars * 2]
    if not doc_ids:
        raise GroundtruthError("no documents long enough to sample from")
    out: list[dict] = []
    per_doc: Counter[str] = Counter()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        d = rng.choice(doc_ids)
        if per_doc[d] >= max_per_doc:
            continue
        text = docs[d]
        s = rng.randrange(0, len(text) - passage_chars)
        dot = text.find(". ", s)  # start at the next sentence
        s = dot + 2 if 0 <= dot < s + 200 else s
        e = min(len(text), s + passage_chars)
        dot = text.rfind(". ", s, e)
        e = dot + 1 if dot > s + passage_chars // 2 else e
        if any(c["doc_id"] == d and c["start"] < e and s < c["end"] for c in out):
            continue  # don't sample overlapping passages
        per_doc[d] += 1
        out.append({"cid": f"c{len(out) + 1:03d}", "doc_id": d, "start": s, "end": e, "text": text[s:e]})
    return out


def locate_answer(candidate: dict, answer_text: str) -> tuple[int, int]:
    """Find `answer_text` inside a candidate passage; return its document offsets (start, end).

    Empty `answer_text` means the whole passage is the gold span. Matching ignores differences in
    whitespace, so text copied from a notebook display still matches the stored passage.
    """
    passage, base = candidate["text"], candidate["start"]
    wanted = " ".join(answer_text.split())
    if not wanted:
        return candidate["start"], candidate["end"]
    # Map each non-space-collapsed position back to the original passage offsets.
    norm_chars: list[str] = []
    origin: list[int] = []
    prev_space = False
    for i, ch in enumerate(passage):
        if ch.isspace():
            if prev_space or not norm_chars:
                continue
            norm_chars.append(" ")
            prev_space = True
        else:
            norm_chars.append(ch)
            prev_space = False
        origin.append(i)
    norm = "".join(norm_chars)
    at = norm.find(wanted)
    if at < 0:
        raise GroundtruthError("answer text not found in the passage: copy it exactly from the passage shown")
    start = base + origin[at]
    end = base + origin[at + len(wanted) - 1] + 1
    return start, end


def assign_splits(records: list[dict], n_dev: int, seed: int = 0) -> list[dict]:
    """Return copies of `records` with `split` set: about `n_dev` dev, the rest test.

    Stratified by category so every category appears in both splits in proportion. Deterministic
    for a given seed. Unanswerable probes always go to test.
    """
    rng = random.Random(seed)
    answerable = [r for r in records if r.get("answerable", True)]
    if not 0 <= n_dev <= len(answerable):
        raise GroundtruthError(f"n_dev must be between 0 and {len(answerable)}")
    by_cat: dict[str, list[dict]] = {}
    for r in answerable:
        by_cat.setdefault(str(r["category"]), []).append(r)
    dev_ids: set[str] = set()
    share = n_dev / len(answerable) if answerable else 0.0
    remainders = []
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda r: str(r["qid"]))
        rng.shuffle(items)
        exact = share * len(items)
        take = int(exact)
        dev_ids.update(str(r["qid"]) for r in items[:take])
        remainders.append((exact - take, cat, items[take:]))
    # hand out the leftover dev slots to the categories with the largest remainders
    for _, _, rest in sorted(remainders, key=lambda t: (-t[0], t[1])):
        if len(dev_ids) >= n_dev:
            break
        if rest:
            dev_ids.add(str(rest[0]["qid"]))
    return [{**r, "split": "dev" if str(r["qid"]) in dev_ids else "test"} for r in records]
