#!/usr/bin/env python3
"""Emit per-Y yaml override + per-Y single-file ckpt symlink shim (§B.1, §3 P-2).

Reads the upstream yaml at
``/work2/.../v2.0/config/SFNO_PLASIM_H5_DERECHO_5410.yaml`` and writes
``<config_dir>/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y<Y>.yaml`` (one
per ``Y ∈ {121..128}``) plus the corresponding single-file checkpoint
symlink shim under ``<exp_dir>/<config>/<run_num>/checkpoints/``.

``--K`` is the forecast-leads horizon (required, no default). Each
emitted yaml has ``ensemble_inference_hours = (K+1)*6`` and
``prediction_duration_days = (K+1)*6/24``.

Examples
--------
Single year, K=60 (canonical eval-track horizon)::

    python scripts/build_5410_yaml_override.py \\
        --year 121 --K 60 \\
        --config-dir $REPO_ROOT/config \\
        --exp-dir $RESULTS/sfno_eval_5410/<run_tag>/inference

All 8 test years::

    python scripts/build_5410_yaml_override.py --all-years --K 60 \\
        --config-dir $REPO_ROOT/config \\
        --exp-dir $RESULTS/sfno_eval_5410/<run_tag>/inference
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sfno_inference_5410.stampede3_yaml_override import (  # noqa: E402
    TEST_YEARS,
    build_ckpt_symlink_shim,
    build_per_y_yaml,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit per-Y 5410 yaml override + ckpt symlink shim.",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--year", type=int, help="Single test year (121..128)")
    grp.add_argument(
        "--all-years",
        action="store_true",
        help="Iterate Y ∈ {121..128}",
    )
    p.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Where to write the per-Y yaml files",
    )
    p.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="upstream `exp_dir` — drives ckpt-discovery globstr; the symlink "
        "shim is created at <exp_dir>/<config>/5410/checkpoints/ckpt_epoch_50.tar",
    )
    p.add_argument(
        "--K",
        type=int,
        required=True,
        help="Forecast-leads horizon (positive int). Sets "
        "ensemble_inference_hours=(K+1)*6 and prediction_duration_days=(K+1)*6/24 "
        "in every emitted yaml. Canonical eval-track value: 60.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    K = args.K
    horizon_hours = (K + 1) * 6
    print(f"[build] K={K} raw_steps={K + 1} horizon_hours={horizon_hours} "
          f"prediction_duration_days={horizon_hours / 24.0}")
    years = TEST_YEARS if args.all_years else (args.year,)
    for Y in years:
        yaml_path = build_per_y_yaml(Y, args.config_dir, args.exp_dir, K=K)
        shim_path = build_ckpt_symlink_shim(Y, args.exp_dir)
        print(f"Y={Y}: yaml={yaml_path}")
        print(f"      shim={shim_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
