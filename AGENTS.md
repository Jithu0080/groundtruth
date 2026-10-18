# AGENTS.md — rules for coding agents working on Groundtruth

Source of truth, in order: `docs/PRD.md`, `docs/ARCHITECTURE.md`, `docs/DATABASE.md`, `docs/API.md`, `docs/DESIGN.md`, `docs/IMPLEMENTATION_PLAN.md`. If code and docs disagree, stop and ask; do not silently change requirements.

## Before any change
- Read the files you will touch and their tests first. Reuse existing helpers (`chunking`, `metrics`, `pipeline.build_pipeline`, `gate.check_regression`, `llm.LLMClient`) before writing new ones.
- Keep changes focused on the task. No unrelated refactors, renames or formatting sweeps.

## Code quality and naming
- Python 3.10+ (CI runs 3.11), type hints on every public function, `from __future__ import annotations`.
- snake_case functions/modules, PascalCase classes, UPPER_CASE constants.
- Functions do one thing; prefer pure functions for metrics and chunking.
- Docstrings on public functions state inputs, outputs and units (characters, milliseconds).

## Structure
- Library code lives in `src/groundtruth/`; CLI wiring only in `cli.py`.
- New retrievers implement the `Retriever` protocol in `retrievers/base.py` and register in `pipeline.py`.
- Experiments are YAML files in `configs/`, never code edits.

## Dependencies
- Allowed: numpy, pandas, scikit-learn, scipy, pyyaml, pypdf, sentence-transformers, pytest, ruff, mypy.
- Adding any other dependency requires written justification in the PR and approval. Pin versions in `requirements.txt`.

## Type safety
- `mypy src` must pass. No `Any` in public signatures unless justified.

## Error handling
- Raise `GroundtruthError` with an actionable message for user-facing problems.
- Never swallow exceptions with a bare `except`. Never return a fake 0 for a metric that could not be computed; skip and count it.

## Security and secrets
- Secrets only from environment variables. Never commit `.env`, keys, tokens or real client data.
- Never log request headers or API keys. Treat corpus text and LLM output as untrusted data; parse JSON, never `eval`.

## Data rules
- `data/golden_v*.jsonl` is immutable once frozen. Fixes create a new version file.
- Never renumber `doc_id`s. Gold labels are character spans, never chunk IDs.
- `results/baseline.json` changes only in the PR that earned the change.
- Never report numbers from the LSA test backend or the fixture corpus as results.

## Testing
- Every metric change needs a hand-computed test case.
- `pytest` must pass before a change is done. Unit tests must not need network access.
- Never weaken the regression thresholds or delete tests to make CI green; investigate the cause.

## Documentation
- Update the README results/recommendation only from real runs, with the config and golden version named.
- Update the relevant doc in `docs/` when behaviour changes.

## Git discipline
- One logical change per commit, imperative messages ("Add hybrid retriever").
- Branch per change; PRs must pass `retrieval-eval`.
- Do not commit generated caches (`.cache/`) or large raw corpora over 100 MB.
