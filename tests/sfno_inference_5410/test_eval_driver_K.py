"""K-threading regression test for scripts/eval_inference_5410.py.

Asserts the new K-aware contract (per docs/2026-05-08_sfno_5410_explicit_K_horizon_plan.md):
  * ``final_datetime_for(init_dt, K)`` returns ``init_dt + (K+1)*6h``.
  * ``build_argv_for_ic(..., K=K)`` writes the same final_datetime into argv.
  * ``build_run_plan(..., K=K)`` validates IC offsets against K via
    ``nwp_ic_offsets_5410(n_samples, K=K)``.
  * Last IC of each year (s=1342) does not overrun n_samples for K=60.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import cftime
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _load_orchestrator():
    """Import scripts/eval_inference_5410.py without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "_orchestrator", _REPO_ROOT / "scripts" / "eval_inference_5410.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_run_root(tmp_path):
    """Minimal run-root with stub IC NCs + ic_source.json."""
    rr = tmp_path / "run"
    inf = rr / "inference"
    ic_dir = inf / "ic_nc"
    ic_dir.mkdir(parents=True)
    (inf / "ic_source.json").write_text(
        json.dumps({"ic_source": "ic_nc_built_from_h5", "ic_nc_dir": str(ic_dir)})
    )
    for Y in range(121, 129):
        for s in (0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342):
            (ic_dir / f"{Y}_{s:04d}.nc").write_bytes(b"")
    cfg = inf
    for Y in range(121, 129):
        (cfg / f"SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y{Y}.yaml").write_text("# stub\n")
    return rr


def test_final_datetime_for_K60():
    orch = _load_orchestrator()
    init = cftime.DatetimeProlepticGregorian(121, 1, 1, 0, has_year_zero=True)
    final = orch.final_datetime_for(init, 60)
    expected = init + dt.timedelta(hours=61 * 6)  # 366h → 0121-01-16 06:00
    assert final == expected
    assert final.year == 121
    assert final.month == 1
    assert final.day == 16
    assert final.hour == 6


def test_final_datetime_for_K56():
    orch = _load_orchestrator()
    init = cftime.DatetimeProlepticGregorian(121, 1, 1, 0, has_year_zero=True)
    final = orch.final_datetime_for(init, 56)
    assert final == init + dt.timedelta(hours=57 * 6)  # 342h → 0121-01-15 06:00


def test_final_datetime_for_rejects_bool():
    """bool must be rejected even though isinstance(True, int) is True."""
    orch = _load_orchestrator()
    init = cftime.DatetimeProlepticGregorian(121, 1, 1, 0, has_year_zero=True)
    for bad in (True, False, 0, -1, "60"):
        with pytest.raises(ValueError):
            orch.final_datetime_for(init, bad)


def test_build_argv_for_ic_propagates_final_datetime(fake_run_root):
    orch = _load_orchestrator()
    entry = orch.build_argv_for_ic(
        Y=121, s=0, K=60,
        run_root=fake_run_root,
        config_dir=fake_run_root / "inference",
    )
    expected_init = cftime.DatetimeProlepticGregorian(121, 1, 1, 0, has_year_zero=True)
    expected_final = expected_init + dt.timedelta(hours=61 * 6)
    assert entry["init_datetime"] == expected_init
    assert entry["final_datetime"] == expected_final
    # As of v2.1 (in-process orchestrator) build_argv_for_ic no longer
    # returns an argv key — the orchestrator calls Stepper.reconfigure_for_ic
    # in-process. Verify the per-IC dict shape instead.
    assert "argv" not in entry
    assert entry["Y"] == 121
    assert entry["s"] == 0
    assert entry["save_basename"] == "Y121_s0000"
    assert entry["config"] == "SFNO"
    assert entry["yaml"].name == "SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y121.yaml"


def test_build_argv_for_ic_K_required(fake_run_root):
    orch = _load_orchestrator()
    with pytest.raises(TypeError):
        # Intentional: K omitted → TypeError from the required kw-only arg.
        orch.build_argv_for_ic(  # type: ignore[call-arg]
            Y=121, s=0,
            run_root=fake_run_root,
            config_dir=fake_run_root / "inference",
        )


def test_build_run_plan_passes_K_to_offsets(fake_run_root):
    """K=60 with n_samples=1460 must succeed; K such that last_s+K>=n_samples must fail."""
    orch = _load_orchestrator()
    plan = orch.build_run_plan(
        fake_run_root, fake_run_root / "inference", K=60,
    )
    assert len(plan) == 96  # 8 years × 12 ICs

    # For non-leap years (n_samples=1460): last_s=1342, last_s + K + 1 = 1403 < 1460. OK.
    # K=120 would push 1342 + 120 = 1462 >= 1460 → ValueError from nwp_ic_offsets_5410.
    with pytest.raises(ValueError):
        orch.build_run_plan(
            fake_run_root, fake_run_root / "inference", K=120,
        )


def test_last_IC_does_not_overrun_K60(fake_run_root):
    """For Y=125 (non-leap), s=1342, K=60: s + K = 1402 < 1460 ✔."""
    orch = _load_orchestrator()
    entry = orch.build_argv_for_ic(
        Y=125, s=1342, K=60,
        run_root=fake_run_root,
        config_dir=fake_run_root / "inference",
    )
    init = entry["init_datetime"]
    final = entry["final_datetime"]
    # init = Y=125, day-of-year ~ s/4 = 335.5 → ~Dec 1
    # final = init + 61*6h = init + 15.25 days → still in year 125.
    assert final.year == 125, f"final must stay within Y=125; got {final}"
    assert final == init + dt.timedelta(hours=61 * 6)
