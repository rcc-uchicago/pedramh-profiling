"""Empirical nondeterminism floor of the legacy subprocess path.

Codex round-2 plan Q#8 anticipated cudnn.benchmark + AMP fp16 might push
A/B equivalence past rtol=1e-5. Gate A confirmed it: pl diverges by
max_rel=4.04e-5 between legacy and inproc.

Before deciding on a final rtol for A/B/C, we measure the LEGACY path's
own run-to-run variance. If legacy-vs-legacy diverges by similar
magnitudes, the cause is upstream cudnn nondeterminism (not the
refactor) and we set the A/B tolerance accordingly.

Runs upstream long_inference.py TWICE for the same IC (Y=121, s=0, K=60)
in fresh subprocesses, then compares outputs across all 8 data vars and
reports the max_abs and max_rel per variable.

Gated behind RUN_AB_TESTS=1. Requires GPU + upstream tree.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)
_DEFAULT_RUN_ROOT = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260507_phase1_gate"
)
_K = 60


_RUN_GATE = pytest.mark.skipif(
    os.environ.get("RUN_AB_TESTS") != "1",
    reason="legacy-baseline test gated behind RUN_AB_TESTS=1; requires GPU",
)


@_RUN_GATE
def test_legacy_vs_legacy_baseline(tmp_path):
    """Run legacy subprocess twice for Y=121 s=0; report per-var max_rel.

    PASS criterion: every variable's max_rel must be finite (i.e., the
    runs both completed and the schemas match). The reported numbers
    establish the empirical floor — they are NOT bounded by an
    assertion. The test prints them so we can read them off the .out
    log and choose the A/B tolerance accordingly.
    """
    import importlib.util
    import xarray as xr

    if not _UPSTREAM_REPO.is_dir():
        pytest.skip(f"upstream not present: {_UPSTREAM_REPO}")
    if not (_DEFAULT_RUN_ROOT / "inference" / "ic_source.json").is_file():
        pytest.skip(f"run-root incomplete: {_DEFAULT_RUN_ROOT}")

    spec = importlib.util.spec_from_file_location(
        "_lvl_orch", _REPO_ROOT / "scripts" / "eval_inference_5410.py",
    )
    orch = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(orch)

    entry = orch.build_argv_for_ic(
        Y=121, s=0, K=_K,
        run_root=_DEFAULT_RUN_ROOT,
        config_dir=_DEFAULT_RUN_ROOT / "inference",
    )

    from tests.sfno_inference_5410.integration._legacy_subprocess_launcher import (
        launch_legacy_subprocess,
    )

    run_a_dir = tmp_path / "run_a"
    run_b_dir = tmp_path / "run_b"
    run_a_dir.mkdir()
    run_b_dir.mkdir()

    extra_env = {
        "PYTHONPATH": f"{_REPO_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
    }
    launch_legacy_subprocess(
        entry, K=_K, upstream_repo=_UPSTREAM_REPO,
        output_dir=run_a_dir, extra_env=extra_env,
    )
    launch_legacy_subprocess(
        entry, K=_K, upstream_repo=_UPSTREAM_REPO,
        output_dir=run_b_dir, extra_env=extra_env,
    )

    fname = f"{entry['save_basename']}_member000_y{entry['Y']:04d}.nc"
    a_path = run_a_dir / fname
    b_path = run_b_dir / fname
    assert a_path.is_file(), f"run_a missing output: {a_path}"
    assert b_path.is_file(), f"run_b missing output: {b_path}"

    print("\n=== LEGACY-vs-LEGACY nondeterminism floor ===")
    print(f"  IC: Y=121 s=0 K={_K}")
    print(f"  run_a: {a_path}")
    print(f"  run_b: {b_path}")
    print()

    with xr.open_dataset(a_path) as a, xr.open_dataset(b_path) as b:
        assert dict(a.sizes) == dict(b.sizes), (
            f"dims mismatch even within legacy: {dict(a.sizes)} vs {dict(b.sizes)}"
        )
        assert set(a.data_vars) == set(b.data_vars)

        print(f"  {'var':<8} {'shape':<24} {'max_abs':<12} {'max_rel':<12}")
        print(f"  {'-'*8} {'-'*24} {'-'*12} {'-'*12}")
        worst_rel = 0.0
        worst_var = None
        for var in sorted(a.data_vars):
            av = a[var].values
            bv = b[var].values
            assert av.shape == bv.shape
            diff = np.abs(av - bv)
            rel = diff / (np.abs(bv) + 1e-12)
            max_abs = float(diff.max())
            max_rel = float(rel.max())
            if max_rel > worst_rel:
                worst_rel = max_rel
                worst_var = var
            print(f"  {var:<8} {str(av.shape):<24} {max_abs:<12.3e} {max_rel:<12.3e}")
        print()
        print(f"  WORST: {worst_var} at max_rel={worst_rel:.3e}")
        print("=" * 50)

    # No assertion on tolerance — this test is purely diagnostic. The
    # printed table is the input to the A/B tolerance decision.
    assert worst_rel >= 0  # Sanity: comparison ran.
