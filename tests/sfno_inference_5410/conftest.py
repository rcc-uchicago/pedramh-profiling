"""Pytest config — add repo src/ to sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"

for entry in (_SRC,):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)
