"""Shared test helpers. Tests run offline: fictional fixture corpus + LSA/BM25 backends."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # configs use repo-relative paths
