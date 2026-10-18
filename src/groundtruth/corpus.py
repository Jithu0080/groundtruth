"""Corpus loading and ingestion (PDF/TXT -> cleaned text files with stable doc_ids)."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from groundtruth import GroundtruthError

DOC_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)


def make_doc_id(name: str) -> str:
    """Turn a file stem into a safe, stable doc_id."""
    doc_id = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    if not doc_id:
        raise GroundtruthError(f"cannot derive a doc_id from file name {name!r}")
    return doc_id


def clean_pages(pages: list[str], min_repeat_ratio: float = 0.5) -> str:
    """Join pages into one text, dropping page numbers and lines repeated on most pages.

    A line in the top or bottom `edge` lines of a page that repeats in that position on at least
    `min_repeat_ratio` of pages (with at least 3 pages) is treated as a running header/footer.
    Only page edges are checked, so repeated sentences in the body are never removed.
    """
    edge = 3
    split = [[ln.strip() for ln in p.splitlines() if ln.strip()] for p in pages]

    def edges(lines: list[str]) -> set[str]:
        return set(lines[:edge]) | set(lines[-edge:])

    repeated: set[str] = set()
    if len(pages) >= 3:
        counts = Counter(ln for lines in split for ln in edges(lines))
        repeated = {ln for ln, c in counts.items() if c / len(pages) >= min_repeat_ratio}
    kept = []
    for lines in split:
        n = len(lines)
        for i, ln in enumerate(lines):
            at_edge = i < edge or i >= n - edge
            if PAGE_NUMBER_RE.match(ln) or (at_edge and ln in repeated):
                continue
            kept.append(ln)
    text = " ".join(kept)
    text = re.sub(r"-\s+(?=[a-z])", "", text)  # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def _pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader  # imported lazily: only ingest needs it

    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def ingest(src: str | Path, out: str | Path) -> int:
    """Clean every .pdf/.txt under `src` into `out/{doc_id}.txt` and write metadata.csv.

    Returns the number of documents written. Fails on doc_id collisions instead of overwriting.
    """
    src_dir, out_dir = Path(src), Path(out)
    files = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in {".pdf", ".txt"})
    if not files:
        raise GroundtruthError(f"no .pdf or .txt files found under {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    seen: dict[str, Path] = {}
    for f in files:
        doc_id = make_doc_id(f.stem)
        if doc_id in seen:
            raise GroundtruthError(f"doc_id collision: {f} and {seen[doc_id]} both map to {doc_id}")
        seen[doc_id] = f
        if f.suffix.lower() == ".pdf":
            pages = _pdf_pages(f)
        else:
            pages = f.read_text(encoding="utf-8", errors="replace").split("\f")
        text = clean_pages(pages)
        if len(text) < 200:
            print(f"warning: {f.name} produced only {len(text)} characters (scanned PDF?); skipped")
            continue
        (out_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        rows.append({"doc_id": doc_id, "title": f.stem, "court": "", "year": "", "source_file": f.name})
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["doc_id", "title", "court", "year", "source_file"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_corpus(corpus_dir: str | Path) -> dict[str, str]:
    """Load {doc_id: text} from a folder of cleaned .txt files."""
    d = Path(corpus_dir)
    if not d.is_dir():
        raise GroundtruthError(f"corpus folder not found: {d} (run `groundtruth ingest` first)")
    docs = {}
    for p in sorted(d.glob("*.txt")):
        if not DOC_ID_RE.match(p.stem):
            raise GroundtruthError(f"invalid doc_id in file name: {p.name}")
        docs[p.stem] = p.read_text(encoding="utf-8")
    if not docs:
        raise GroundtruthError(f"corpus folder {d} contains no .txt files")
    return docs


def corpus_hash(docs: dict[str, str]) -> str:
    """Short content hash; changes whenever any document is added, removed or edited."""
    h = hashlib.sha256()
    for doc_id in sorted(docs):
        h.update(doc_id.encode())
        h.update(b"\0")
        h.update(docs[doc_id].encode())
        h.update(b"\0")
    return h.hexdigest()[:12]
