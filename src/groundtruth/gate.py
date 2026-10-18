"""The regression gate: compare a fresh summary with results/baseline.json."""

from __future__ import annotations

from typing import Any

OVERALL_TOLERANCE = 0.01  # one point of recall@10
CATEGORY_TOLERANCE = 0.05  # an average can hold steady while one category collapses


def check_regression(
    current: dict[str, Any],
    baseline: dict[str, Any],
    overall_tol: float = OVERALL_TOLERANCE,
    category_tol: float = CATEGORY_TOLERANCE,
) -> list[str]:
    """Return human-readable failures; an empty list means the gate passes."""
    problems = []
    if current.get("golden") != baseline.get("golden") or current.get("split") != baseline.get("split"):
        problems.append(
            f"baseline was measured on {baseline.get('golden')} [{baseline.get('split')}], "
            f"current on {current.get('golden')} [{current.get('split')}]; update the baseline deliberately"
        )
        return problems
    cur, base = current["recall@10"], baseline["recall@10"]
    if cur < base - overall_tol:
        problems.append(f"recall@10 fell from {base:.3f} to {cur:.3f} (allowed drop {overall_tol:.3f})")
    for cat, b in baseline.get("by_category", {}).items():
        c = current.get("by_category", {}).get(cat)
        if c is None:
            problems.append(f"category {cat!r} missing from current run")
        elif c["recall@10"] < b["recall@10"] - category_tol:
            problems.append(f"category {cat!r} recall@10 fell from {b['recall@10']:.3f} to {c['recall@10']:.3f}")
    return problems
