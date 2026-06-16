"""A/B equivalence: legacy subprocess path vs new in-process orchestrator.

Codex round-1 major #1: mocking + count-asserts is not enough. We must
prove the new path produces the SAME NetCDF outputs as the old path on:

  * one IC (gate A — bare equivalence);
  * two same-year ICs (gate B — within-Y reconfigure_for_ic, val_year_changed=False);
  * one cross-year pair (gate C — Y boundary, val_year_changed=True;
    refreshes constant_boundary_data, val_year_start, leap_year, no_leap_year).

For each case we run BOTH paths, save outputs to separate dirs, and
compare on coords + var names + time range + numerical values at
rtol=1e-5.

Gated behind ``RUN_AB_TESTS=1`` because they:
  * require GPU + the upstream PanguWeather/v2.0 tree;
  * require IC NCs + per-Y yamls already built;
  * take 5-15 min wallclock per test (one full Stepper construction +
    rollouts on each side).

Usage::

    RUN_AB_TESTS=1 pytest tests/sfno_inference_5410/integration/test_ab_equivalence.py

Set ``AB_RUN_ROOT`` to point at an existing run-root with all 96 IC NCs
+ all 8 per-Y yamls + ckpt symlink shim. Default points at the
production run-root used by the smoke.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
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

# rtol/atol for numerical equivalence. Cudnn nondeterminism + AMP can
# push us a bit past 1e-7 absolute; rtol=1e-5 is the own-track convention.
_RTOL = 1e-5
_ATOL = 1e-7


_RUN_GATE = pytest.mark.skipif(
    os.environ.get("RUN_AB_TESTS") != "1",
    reason="A/B tests gated behind RUN_AB_TESTS=1; require GPU + upstream tree",
)


def _need_upstream():
    if not _UPSTREAM_REPO.is_dir():
        pytest.skip(f"upstream not present: {_UPSTREAM_REPO}")


def _need_run_root():
    rr = Path(os.environ.get("AB_RUN_ROOT", _DEFAULT_RUN_ROOT))
    if not (rr / "inference" / "ic_source.json").is_file():
        pytest.skip(f"AB_RUN_ROOT missing or incomplete: {rr}")
    return rr


@pytest.fixture(scope="module")
def run_root():
    _need_upstream()
    return _need_run_root()


@pytest.fixture(scope="module")
def orchestrator():
    """Import scripts/eval_inference_5410.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "_ab_orch", _REPO_ROOT / "scripts" / "eval_inference_5410.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _build_plan_for(orchestrator, run_root, ic_specs):
    """Build a per-IC entry list for a small custom (Y, s) list."""
    entries = []
    for Y, s in ic_specs:
        entries.append(orchestrator.build_argv_for_ic(
            Y=Y, s=s, K=_K,
            run_root=run_root,
            config_dir=run_root / "inference",
        ))
    return entries


def _run_legacy_subprocess(entries, *, output_dir):
    """Invoke the legacy reference launcher for each entry."""
    from tests.sfno_inference_5410.integration._legacy_subprocess_launcher import (
        launch_legacy_subprocess,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_extra = {
        "PYTHONPATH": f"{_REPO_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
    }
    for e in entries:
        launch_legacy_subprocess(
            e, K=_K, upstream_repo=_UPSTREAM_REPO,
            output_dir=output_dir, extra_env=env_extra,
        )


def _run_inproc(orchestrator, run_root, *, output_dir,
                 years=None, limit_ics=None, ic_subset=None):
    """Invoke the new in-process orchestrator as a subprocess (clean state).

    Spawning a fresh subprocess for the inproc path ensures we don't
    inherit any imports or torch state from the test process; it's the
    same way SLURM invokes it. Output dir is set up via a fake run-root
    that mirrors AB_RUN_ROOT but redirects upstream_raw to ``output_dir``.

    Pass either (years, limit_ics) for contiguous-IC subsets OR
    ic_subset='Y:s,Y:s,...' for explicit cross-year selection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fake run-root: symlink everything in inference/ except upstream_raw,
    # which points at our test output_dir.
    fake_root = output_dir.parent / "fake_run_root"
    fake_inf = fake_root / "inference"
    if fake_inf.is_symlink() or fake_inf.is_dir():
        if fake_inf.is_dir() and not fake_inf.is_symlink():
            shutil.rmtree(fake_inf)
        else:
            fake_inf.unlink()
    fake_inf.mkdir(parents=True, exist_ok=True)
    src_inf = run_root / "inference"
    for child in src_inf.iterdir():
        if child.name == "upstream_raw":
            continue
        link = fake_inf / child.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(child)
    raw_link = fake_inf / "upstream_raw"
    if raw_link.exists() or raw_link.is_symlink():
        raw_link.unlink()
    raw_link.symlink_to(output_dir)

    cmd = [
        sys.executable, "-u", str(_REPO_ROOT / "scripts" / "eval_inference_5410.py"),
        "--run-root", str(fake_root),
        "--config-dir", str(fake_inf),
        "--K", str(_K),
    ]
    if ic_subset is not None:
        cmd += ["--ic-subset", ic_subset]
    else:
        if years:
            cmd += ["--years", *(str(y) for y in years)]
        if limit_ics is not None:
            cmd += ["--limit-ics", str(limit_ics)]
    cmd += ["--launch"]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("RANK", "0")
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("MASTER_ADDR", "localhost")
    env.setdefault("MASTER_PORT", "29500")
    subprocess.run(cmd, env=env, check=True, cwd=str(_REPO_ROOT))


def _compare_netcdfs(legacy_path: Path, inproc_path: Path) -> None:
    """Strict equivalence: dims, coords, time, var names, values."""
    import xarray as xr
    with xr.open_dataset(legacy_path) as a, xr.open_dataset(inproc_path) as b:
        assert dict(a.sizes) == dict(b.sizes), (
            f"dims mismatch: legacy={dict(a.sizes)} inproc={dict(b.sizes)}"
        )
        assert set(a.data_vars) == set(b.data_vars), (
            f"data_vars mismatch: legacy={sorted(a.data_vars)} "
            f"inproc={sorted(b.data_vars)}"
        )
        # Time coords (cftime objects) must be equal.
        a_t = list(a.time.values)
        b_t = list(b.time.values)
        assert a_t == b_t, f"time mismatch: legacy={a_t[:3]}... inproc={b_t[:3]}..."

        # Numerical equivalence per-variable.
        for var in a.data_vars:
            av = a[var].values
            bv = b[var].values
            assert av.shape == bv.shape, (
                f"{var} shape mismatch: legacy={av.shape} inproc={bv.shape}"
            )
            if not np.allclose(av, bv, rtol=_RTOL, atol=_ATOL):
                # Report the worst absolute + relative diff.
                diff = np.abs(av - bv)
                rel = diff / (np.abs(bv) + _ATOL)
                raise AssertionError(
                    f"{var} numerical mismatch: "
                    f"max_abs={diff.max():.3e} max_rel={rel.max():.3e} "
                    f"(rtol={_RTOL} atol={_ATOL})"
                )


def _legacy_filename(entry) -> str:
    return f"{entry['save_basename']}_member000_y{entry['Y']:04d}.nc"


# === Gate A ======================================================
@_RUN_GATE
def test_one_ic_equivalence(tmp_path, run_root, orchestrator):
    """Y=121 s=0, K=60. Old subprocess vs new in-process."""
    ic_specs = [(121, 0)]
    entries = _build_plan_for(orchestrator, run_root, ic_specs)

    legacy_dir = tmp_path / "legacy"
    inproc_dir = tmp_path / "inproc"
    _run_legacy_subprocess(entries, output_dir=legacy_dir)
    _run_inproc(orchestrator, run_root,
                output_dir=inproc_dir, years=[121], limit_ics=1)

    fname = _legacy_filename(entries[0])
    _compare_netcdfs(legacy_dir / fname, inproc_dir / fname)


# === Gate B ======================================================
@_RUN_GATE
def test_two_same_year_ics_equivalence(tmp_path, run_root, orchestrator):
    """Y=121 s=0 and s=122 (within-Y; val_year_changed=False on the 2nd)."""
    ic_specs = [(121, 0), (121, 122)]
    entries = _build_plan_for(orchestrator, run_root, ic_specs)

    legacy_dir = tmp_path / "legacy"
    inproc_dir = tmp_path / "inproc"
    _run_legacy_subprocess(entries, output_dir=legacy_dir)
    _run_inproc(orchestrator, run_root,
                output_dir=inproc_dir, years=[121], limit_ics=2)

    for e in entries:
        fname = _legacy_filename(e)
        _compare_netcdfs(legacy_dir / fname, inproc_dir / fname)


# === Gate C (Codex's must-have) ====================================
@_RUN_GATE
def test_cross_year_pair_equivalence(tmp_path, run_root, orchestrator):
    """Y=121 s=0 then Y=122 s=0. val_year_changed=True on the 2nd:
    constant_boundary_data, val_year_start, leap_year, no_leap_year all
    refresh. This is the most state-mutating reconfigure_for_ic call —
    Codex called it out as the must-have gate.
    """
    ic_specs = [(121, 0), (122, 0)]
    entries = _build_plan_for(orchestrator, run_root, ic_specs)

    legacy_dir = tmp_path / "legacy"
    inproc_dir = tmp_path / "inproc"
    _run_legacy_subprocess(entries, output_dir=legacy_dir)
    _run_inproc(
        orchestrator, run_root,
        output_dir=inproc_dir,
        ic_subset="121:0,122:0",
    )

    for e in entries:
        fname = _legacy_filename(e)
        _compare_netcdfs(legacy_dir / fname, inproc_dir / fname)
