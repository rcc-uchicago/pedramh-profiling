"""Legacy subprocess launcher — reference path for A/B equivalence tests.

Captured at 2026-05-08 BEFORE the in-process refactor of
``scripts/eval_inference_5410.py``. This helper preserves the
pre-refactor invocation pattern (one ``subprocess.run`` per IC against
upstream ``long_inference.py``) so the A/B tests can compare outputs
between the new in-process orchestrator and the old subprocess path
even after the orchestrator is rewritten.

Codex round-2 minor recommendation: keep an independent old-subprocess
launcher even after ``eval_inference_5410.py`` is rewritten. This file
is the canonical "old path" for the A/B tests.

Intentionally minimal:
  * No preflight (callers do their own).
  * No plan-building (callers pass per-IC entry dicts).
  * No orchestration (one IC per call, synchronous).
  * No ``--async_save`` flag (matches v2.1 default in the new path).
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def launch_legacy_subprocess(
    entry: Mapping[str, Any],
    *,
    K: int,
    upstream_repo: Path,
    output_dir: Path,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run upstream ``long_inference.py`` for ONE IC via subprocess.

    Parameters
    ----------
    entry
        Per-IC dict with keys: ``init_datetime`` (cftime),
        ``ic_nc`` (Path), ``yaml`` (Path), ``save_basename`` (str).
        Same shape as ``build_argv_for_ic`` returns in the new orchestrator.
    K
        Forecast-leads horizon. Must match the per-Y yaml's
        ``ensemble_inference_hours / 6 - 1`` and
        ``prediction_duration_days * 4 - 1``.
    upstream_repo
        Path to ``/work2/.../v2.0`` so ``long_inference.py`` and
        ``utils.YParams`` resolve.
    output_dir
        Where the upstream ``long_inference.py`` writes the per-IC
        NetCDF (``Y{Y}_s{s:04d}_member000_y{Y}.nc``).
    extra_env
        Optional env-var overrides (e.g., ``PYTHONPATH``).

    Returns
    -------
    subprocess.CompletedProcess
        With ``check=True`` raised on non-zero exit.
    """
    upstream_repo = Path(upstream_repo)
    output_dir = Path(output_dir)
    long_inf = upstream_repo / "long_inference.py"
    if not long_inf.is_file():
        raise FileNotFoundError(f"upstream long_inference.py not found: {long_inf}")
    if not Path(entry["ic_nc"]).is_file():
        raise FileNotFoundError(f"IC NC not found: {entry['ic_nc']}")
    if not Path(entry["yaml"]).is_file():
        raise FileNotFoundError(f"per-Y yaml not found: {entry['yaml']}")

    init_dt = entry["init_datetime"]
    final_dt = init_dt + dt.timedelta(hours=(K + 1) * 6)
    init_str = init_dt.strftime("%Y-%m-%d_%H:%M:%S")
    final_str = final_dt.strftime("%Y-%m-%d_%H:%M:%S")

    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        sys.executable, "-u", str(long_inf),
        "--run_num", "5410",
        "--yaml_config", str(entry["yaml"]),
        "--config", "SFNO",
        "--init_datetime", init_str,
        "--final_datetime", final_str,
        "--init_nc_filepaths", str(entry["ic_nc"]),
        "--output_dir", str(output_dir),
        "--save_basename", entry["save_basename"],
    ]

    env = os.environ.copy()
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("RANK", "0")
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("MASTER_ADDR", "localhost")
    env.setdefault("MASTER_PORT", "29500")
    if extra_env:
        env.update(extra_env)

    return subprocess.run(argv, cwd=str(upstream_repo), env=env, check=True)


__all__ = ("launch_legacy_subprocess",)
