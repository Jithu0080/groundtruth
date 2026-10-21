"""Groundtruth live demo (Gradio): compare baseline and production retrieval on 700 Supreme Court judgments."""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr
from engine import Comparison, Engine, SystemResult

from groundtruth.retrievers.dense import SentenceTransformerEncoder
from groundtruth.retrievers.rerank import CrossEncoderReranker

HERE = Path(__file__).parent
ENGINE = Engine(
    HERE / "data",
    encoder=SentenceTransformerEncoder("BAAI/bge-small-en-v1.5"),
    reranker=CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
)
RESULTS_MD = (HERE / "data" / "experiments.md").read_text(encoding="utf-8")

INTRO = """
# Groundtruth: Retrieval Evaluation Harness for Legal RAG

Ask a legal question about **700 Supreme Court of India judgments (2023)** and compare what two retrieval
setups return. **Baseline:** dense embeddings over 800-character chunks. **Production:** dense + BM25 hybrid
search with a cross-encoder reranker over 400-character chunks. On the 74-question held-out test set,
production raises recall@10 from **0.513 to 0.676** (95% CI +0.04 to +0.28).

Pick an example from the evaluation set to see where each system ranks the human-labelled correct passage,
or type your own question.
"""

CARD = (
    "<div style='border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:10px 12px;margin:0 0 8px;"
    "{extra}'><div style='font-size:12px;opacity:.75;margin-bottom:4px'>#{rank} · {doc} · chars {start}–{end}"
    "{badge}</div><div style='font-size:14px;line-height:1.45'>{text}</div></div>"
)
GOLD_STYLE = "border:2px solid #2e9e5b;background:rgba(46,158,91,.10);"


def render(result: SystemResult, title: str, note: str) -> str:
    parts = [
        f"<div style='font-weight:600;font-size:16px'>{html.escape(title)}</div>",
        f"<div style='font-size:12px;opacity:.75;margin-bottom:8px'>{html.escape(note)} · "
        f"{result.latency_ms:.0f} ms on this CPU</div>",
    ]
    for rank, (chunk, _score) in enumerate(result.hits, start=1):
        is_gold = result.gold_rank is not None and rank == result.gold_rank
        parts.append(
            CARD.format(
                extra=GOLD_STYLE if is_gold else "",
                rank=rank,
                doc=html.escape(chunk.doc_id),
                start=chunk.start,
                end=chunk.end,
                badge=" · <b style='color:#2e9e5b'>correct passage</b>" if is_gold else "",
                text=html.escape(chunk.text),
            )
        )
    return "".join(parts)


def verdict(c: Comparison) -> str:
    if c.golden is None:
        return (
            "_Your own question: there is no labelled answer, so neither side is scored. "
            "Compare the passages yourself._"
        )
    g = c.golden
    span = g.relevant[0]
    gold_text = ENGINE.docs[span["doc_id"]][span["start"] : span["end"]]

    def where(r: SystemResult) -> str:
        return f"rank **{r.gold_rank}**" if r.gold_rank else "**not in the top 10**"

    return (
        f"**Evaluation question {g.qid}** ({g.category}, {g.split} split). "
        f"Correct passage, baseline: {where(c.baseline)} · production: {where(c.production)}.\n\n"
        f"> **Labelled answer** ({span['doc_id']}): {gold_text}"
    )


def search(query: str):
    try:
        c = ENGINE.compare(query)
    except ValueError as e:
        msg = f"_{e}_"
        return msg, "", ""
    return (
        verdict(c),
        render(c.baseline, "Baseline", "dense · 800-char chunks"),
        render(c.production, "Production", "hybrid + reranker · 400-char chunks"),
    )


EXAMPLE_ROWS = [[e["query"]] for e in ENGINE.examples]

with gr.Blocks(title="Groundtruth: legal retrieval demo") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Search"):
        with gr.Row():
            query = gr.Textbox(
                label="Legal question",
                placeholder="e.g. Can a court make someone pay money as a condition for anticipatory bail?",
                max_lines=2,
                scale=5,
            )
            go = gr.Button("Search", variant="primary", scale=1)
        if EXAMPLE_ROWS:
            gr.Examples(EXAMPLE_ROWS, inputs=[query], label="Examples from the held-out test set")
        summary = gr.Markdown()
        with gr.Row():
            left = gr.HTML()
            right = gr.HTML()
        go.click(search, inputs=[query], outputs=[summary, left, right])
        query.submit(search, inputs=[query], outputs=[summary, left, right])
    with gr.Tab("Results"):
        gr.Markdown(RESULTS_MD)
    with gr.Tab("About"):
        gr.Markdown(
            "Groundtruth measures retrieval for a legal research assistant with a 104-question, human-reviewed "
            "golden set (passages labelled as character spans), recall@k / MRR / nDCG per question type, paired "
            "bootstrap confidence intervals, and a CI gate that blocks any pull request dropping recall@10 by more "
            "than one point.\n\n"
            "Source code: https://github.com/Jithu0080/groundtruth\n\n"
            "Data: [Indian Supreme Court Judgments](https://registry.opendata.aws/indian-supreme-court-judgments/) "
            "(CC-BY-4.0), 700-document sample of 2023 English judgments, text extracted and cleaned.\n\n"
            "Latency shown here is on the free CPU this demo runs on; the evaluation numbers were measured on a GPU."
        )

if __name__ == "__main__":
    demo.queue().launch()
