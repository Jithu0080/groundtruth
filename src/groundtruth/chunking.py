"""Fixed-size character chunking that keeps offsets into the original document."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    start: int  # character offset in the original document (inclusive)
    end: int  # character offset (exclusive)
    text: str


def chunk_document(doc_id: str, text: str, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split text into windows of about `chunk_size` characters overlapping by `overlap`.

    Cuts snap back to the last space in the second half of the window so words stay whole.
    Every chunk satisfies text[chunk.start:chunk.end] == chunk.text.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            space = text.rfind(" ", start, end)
            if space > start + chunk_size // 2:
                end = space
        chunks.append(Chunk(f"{doc_id}:{start}", doc_id, start, end, text[start:end]))
        if end == n:
            break
        start = max(end - overlap, start + 1)  # step back by the overlap, always move forward
    return chunks


def chunk_corpus(docs: dict[str, str], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk every document, in sorted doc_id order so runs are deterministic."""
    out: list[Chunk] = []
    for doc_id in sorted(docs):
        out.extend(chunk_document(doc_id, docs[doc_id], chunk_size, overlap))
    return out
