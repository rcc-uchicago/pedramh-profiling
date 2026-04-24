"""Pytest config — add repo src/ to sys.path.

The repo has no pyproject.toml / installed package, so tests import
`plasim_makani_packager.*` by prepending {repo}/src to sys.path here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_TESTS = Path(__file__).resolve().parent

for entry in (_SRC, _TESTS):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)
