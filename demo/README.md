---
title: Groundtruth Legal Retrieval Demo
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: Compare legal retrieval setups on 700 SC judgments
---

# Groundtruth: live retrieval demo

Ask a legal question about 700 Supreme Court of India judgments (2023) and compare what two retrieval setups return:

- **Baseline:** dense embeddings (`BAAI/bge-small-en-v1.5`) over 800-character chunks
- **Production:** dense + BM25 hybrid search (reciprocal rank fusion) with a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over 400-character chunks

For questions from the evaluation set, the demo shows where each system ranks the human-reviewed correct passage. On the 74-question held-out test set, production raises recall@10 from 0.513 to 0.676 (95% CI +0.04 to +0.28).

Source, evaluation harness and CI gate: https://github.com/Jithu0080/groundtruth

Data: [Indian Supreme Court Judgments](https://registry.opendata.aws/indian-supreme-court-judgments/), CC-BY-4.0. A 700-document sample of the 2023 English judgments, text extracted and cleaned.
