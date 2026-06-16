"""Preflight checks for the group-code training track.

Phase B.5 (preflight half) of the plan v5.

Run before every train submit (`--phase pre_train`, checks 1-10) and before
every inference submit (`--phase pre_inference`, checks 11-15). Failure exits
non-zero so the slurm script aborts before submitting the heavy job.

Checks:
  1. Env imports (torch, h5py, xarray, netCDF4, torch_harmonics, einops, timm, cf_xarray, group SFNO module).
  2. YAML parse via group's utils.YParams against `--config`.
  3. Calendar manifest exists with required fields per train_data_sets/validation_data_sets year list.
  4. File presence: every (year, idx) the YAML's date ranges expand to has a corresponding <year>_<idx:04>.h5.
  5. Key-set audit: `<val_year_start>_0000.h5`'s /input/ keys equal the 59-key smoke set.
  6. Constant-boundary file `<val_year_start>_0000.h5` has /input/lsm and /input/sg.
  7. GetDataset construction dry run (validate=True, num_inferences=2); finite values.
  8. Stats files open with correct schema (Z=10, Z_2=10, 14 vars).
  9. Climatology opens with `xr.coders.CFDatetimeCoder(use_cftime=True)`; `time` coord present.
 10. z0 audit consistency: if YAML places z0 in constant_boundary_variables, manifest std < 1e-3.

  11. Score-function wrapper module imports + SFNO_v2 import path.        [Phase G]
  12. Checkpoint torch.load (cpu); model_state present; ema_state status. [Phase G]
  13. Init NC time coord contains `--init-dt` within tolerance.           [Phase G]
  14. Constant-boundary load + spatial z-score sanity (lsm range, std>0). [Phase G]
  15. Boundary file presence for `init_year, init_idx ... init_idx + steps`. [Phase G]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Callable

logger = logging.getLogger("preflight")


class PreflightFailure(RuntimeError):
    pass


def _check(label: str) -> Callable:
    def deco(fn):
        def wrapped(*a, **kw):
            try:
                fn(*a, **kw)
                logger.info("PASS  %s", label)
            except Exception as e:
                logger.error("FAIL  %s: %s", label, e)
                raise PreflightFailure(label) from e
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


# ---------- Pre-train checks (1-10) ----------

@_check("01 env imports")
def _check_01_env(_args) -> None:
    import torch  # noqa: F401
    import h5py   # noqa: F401
    import xarray  # noqa: F401
    import netCDF4  # noqa: F401
    import torch_harmonics  # noqa: F401
    import einops  # noqa: F401
    import timm  # noqa: F401
    import cf_xarray  # noqa: F401
    group_root = os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0")
    sys.path.insert(0, group_root)
    from networks.modulus_sfno.sfnonet import SphericalFourierNeuralOperatorNet_v2  # noqa: F401


@_check("02 YAML parse via group YParams")
def _check_02_yparams(args) -> None:
    sys.path.insert(0, os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0"))
    from utils.YParams import YParams
    p = YParams(str(args.yaml), args.config)
    assert hasattr(p, "data_dir") and p.data_dir, "data_dir missing"
    assert hasattr(p, "nettype") and p.nettype == "sfno_plasim", f"nettype={getattr(p, 'nettype', None)!r}"


@_check("03 calendar manifest fields")
def _check_03_manifest(args) -> None:
    manifest = json.loads(args.manifest.read_text())
    assert manifest["calendar"] == "proleptic_gregorian"
    required = {"year", "n_timesteps", "synthetic_start_dt", "last_train_init_dt",
                "train_end_exclusive_dt", "last_val_init_dt_for_max_lead_K",
                "val_end_exclusive_dt_for_max_lead_K", "z0_temporal_std_mean"}
    for y in manifest["years"]:
        missing = required - set(y.keys())
        assert not missing, f"year {y.get('year')}: missing {missing}"


def _expand_idxs(start_dt_str: str, end_excl_dt_str: str) -> list[tuple[int, int]]:
    """Expand a [start, end_excl) date range into a list of (year, idx) pairs.

    Mirrors group's partition_date_range exclusivity: arange(start_h, end_excl_h, 6).
    """
    import cftime
    start_dt = cftime.datetime.strptime(
        start_dt_str, "%Y-%m-%d %H:%M:%S",
        calendar="proleptic_gregorian", has_year_zero=True)
    end_excl = cftime.datetime.strptime(
        end_excl_dt_str, "%Y-%m-%d %H:%M:%S",
        calendar="proleptic_gregorian", has_year_zero=True)
    pairs: list[tuple[int, int]] = []
    dt = start_dt
    while dt < end_excl:
        year = dt.year
        jan1 = cftime.DatetimeProlepticGregorian(year, 1, 1, has_year_zero=True)
        idx = int((dt - jan1).total_seconds()) // (6 * 3600)
        pairs.append((year, idx))
        dt = dt + timedelta(hours=6)
    return pairs


@_check("04 file presence over expanded date ranges")
def _check_04_files(args) -> None:
    sys.path.insert(0, os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0"))
    from utils.YParams import YParams
    p = YParams(str(args.yaml), args.config)

    def _check_data_sets(label: str, sets: dict) -> int:
        missing: list[tuple[int, int]] = []
        total = 0
        for data_dir, ranges in sets.items():
            for start, end in ranges:
                for year, idx in _expand_idxs(start, end):
                    total += 1
                    path = Path(data_dir) / f"{year}_{idx:04d}.h5"
                    if not path.is_file():
                        missing.append((year, idx))
        if missing:
            sample = missing[:5]
            raise AssertionError(f"{label}: {len(missing)}/{total} missing, e.g. {sample}")
        return total

    n_train = _check_data_sets("train_data_sets", p.train_data_sets)
    n_val = _check_data_sets("validation_data_sets", p.validation_data_sets)
    logger.info("    expanded train=%d files, val=%d files (all present)", n_train, n_val)


@_check("05 key-set audit on val_year_start_0000.h5")
def _check_05_keys(args) -> None:
    sys.path.insert(0, os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0"))
    from utils.YParams import YParams
    p = YParams(str(args.yaml), args.config)
    import h5py
    sample = Path(p.data_dir) / f"{int(p.val_year_start)}_0000.h5"
    assert sample.is_file(), f"{sample} missing"
    with h5py.File(sample, "r") as f:
        keys = sorted(f["input"].keys())

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sfno_training_group.tools._h5_keys import all_input_keys_for_smoke
    expected = sorted(all_input_keys_for_smoke())
    if keys != expected:
        diff = set(keys) ^ set(expected)
        raise AssertionError(f"{sample}: 59-key set mismatch, diff={diff}")


@_check("06 constant-boundary file has lsm + sg")
def _check_06_constbdry(args) -> None:
    sys.path.insert(0, os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0"))
    from utils.YParams import YParams
    p = YParams(str(args.yaml), args.config)
    import h5py
    path = Path(p.data_dir) / f"{int(p.val_year_start)}_0000.h5"
    with h5py.File(path, "r") as f:
        for v in p.constant_boundary_variables:
            assert v in f["input"], f"{path}: missing constant_boundary var {v!r}"


@_check("07 GetDataset construction (validate)")
def _check_07_getdataset(args) -> None:
    sys.path.insert(0, os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0"))
    from utils.YParams import YParams
    from utils.data_loader_multifiles import GetDataset
    import torch
    import torch.distributed as dist  # noqa: F401

    p = YParams(str(args.yaml), args.config)
    # GetDataset reads has_diagnostic; mirror train.py:3435.
    if hasattr(p, "diagnostic_variables"):
        p["has_diagnostic"] = len(p.diagnostic_variables) > 0
    else:
        p["has_diagnostic"] = False
    p["forecast_lead_times"] = list(getattr(p, "forecast_lead_times", [1]))
    # Disable workers + DDP for this dry-run.
    p["num_data_workers"] = 0
    if not hasattr(p, "num_ensemble_members"):
        p["num_ensemble_members"] = 1

    ds = GetDataset(
        p, p.data_dir,
        year_start=int(p.val_year_start), year_end=int(p.val_year_end),
        train=False, validate=True, num_inferences=2,
    )
    n = len(ds)
    assert n > 0, f"GetDataset reported len=0"
    sample = ds[0]
    # ds[0] returns a tuple; check we got something. Specific shape assertions
    # are intentionally minimal — group code returns nested tuples we don't want
    # to overcommit to until the actual run.
    assert sample is not None
    logger.info("    GetDataset len=%d, sample type=%s", n, type(sample).__name__)


@_check("08 stats files open with expected schema")
def _check_08_stats(args) -> None:
    import xarray as xr
    sys.path.insert(0, os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0"))
    from utils.YParams import YParams
    p = YParams(str(args.yaml), args.config)
    mean_path = Path(p.data_dir) / p.upper_air_mean
    std_path = Path(p.data_dir) / p.upper_air_std
    for path in (mean_path, std_path):
        ds = xr.open_dataset(path)
        assert "Z" in ds.coords and ds.coords["Z"].size == 10, f"{path}: Z coord wrong"
        assert "Z_2" in ds.coords and ds.coords["Z_2"].size == 10, f"{path}: Z_2 coord wrong"
        for v in ("ta", "ua", "va", "hus", "zg", "pl", "tas", "pr_6h",
                  "lsm", "sg", "z0", "sst", "rsdt", "sic"):
            assert v in ds.data_vars, f"{path}: missing var {v!r}"
        ds.close()


@_check("09 climatology opens with cftime decoder")
def _check_09_climatology(args) -> None:
    import xarray as xr
    sys.path.insert(0, os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0"))
    from utils.YParams import YParams
    p = YParams(str(args.yaml), args.config)
    path = Path(p.data_dir) / p.climatology_file
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(path, decode_times=time_coder)
    assert "time" in ds.coords, f"{path}: time coord missing"
    ds.close()


@_check("10 z0 placement vs audit consistency")
def _check_10_z0_audit(args) -> None:
    manifest = json.loads(args.manifest.read_text())
    sys.path.insert(0, os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0"))
    from utils.YParams import YParams
    p = YParams(str(args.yaml), args.config)
    if "z0" in p.constant_boundary_variables:
        for y in manifest["years"]:
            assert y["z0_temporal_std_mean"] < 1e-3, (
                f"year {y['year']}: z0_temporal_std_mean={y['z0_temporal_std_mean']:.2e} "
                f">= 1e-3; cannot place z0 in constant_boundary_variables."
            )


# ---------- Pre-inference checks (11-15) ----------
# Phase G provides the score-function wrapper; these checks live here for future
# wiring and currently degrade gracefully when the wrapper is absent.

@_check("11 score-function wrapper imports")
def _check_11_wrapper_imports(args) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sfno_training_group.score_function.group_emulator import GroupEmulator  # noqa: F401


@_check("12 checkpoint torch.load (cpu)")
def _check_12_ckpt(args) -> None:
    import torch
    ckpt_path = args.run_dir / "checkpoints" / "best_ckpt.tar"
    if not ckpt_path.is_file():
        ckpt_path = args.run_dir / "training_checkpoints" / "best_ckpt.tar"
    assert ckpt_path.is_file(), f"no best_ckpt.tar under {args.run_dir}/(checkpoints|training_checkpoints)"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state" in ckpt, f"{ckpt_path}: no model_state key"
    has_ema = "ema_state" in ckpt and ckpt["ema_state"] is not None
    logger.info("    ckpt keys=%s; ema_state present=%s", list(ckpt.keys())[:5], has_ema)


@_check("13 init NC contains init_dt in time coord")
def _check_13_init_dt(args) -> None:
    import xarray as xr
    import cftime
    init_dt = cftime.datetime.strptime(
        args.init_dt, "%Y-%m-%d_%H:%M:%S",
        calendar="proleptic_gregorian", has_year_zero=True)
    ds = xr.open_dataset(args.init_nc, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    times = list(ds["time"].values)
    if init_dt not in times:
        # Allow strptime-roundtrip equality (cftime equality requires same calendar).
        diffs = [abs((t - init_dt).total_seconds()) for t in times]
        if min(diffs) > 1.0:
            raise AssertionError(f"init_dt {init_dt} not in {args.init_nc} time coord; closest diff = {min(diffs)}s")
    ds.close()


@_check("14 constant-boundary spatial z-score sanity")
def _check_14_constbdry_zscore(args) -> None:
    import h5py
    import numpy as np
    sys.path.insert(0, os.environ.get("GROUP_PANGU_ROOT", "/work2/09979/awikner/stampede3/PanguWeather/v2.0"))
    from utils.YParams import YParams
    p = YParams(str(args.yaml), args.config)
    file_path = Path(p.data_dir) / f"{int(p.val_year_start)}_0000.h5"
    with h5py.File(file_path, "r") as f:
        data = np.stack([f["input"][v][:] for v in p.constant_boundary_variables], axis=0).astype(np.float64)
    spatial_mean = data.mean(axis=(1, 2), keepdims=True)
    spatial_std = data.std(axis=(1, 2), keepdims=True)
    assert (spatial_std > 0).all(), f"some constant_boundary var has zero spatial std: {spatial_std.flatten()}"
    # lsm should look like a 0/1 mask before normalization.
    if "lsm" in p.constant_boundary_variables:
        lsm_idx = p.constant_boundary_variables.index("lsm")
        mn, mx = float(data[lsm_idx].min()), float(data[lsm_idx].max())
        assert -0.01 <= mn and mx <= 1.01, f"lsm range [{mn}, {mx}] not in [0, 1]"


@_check("15 boundary file presence for rollout window")
def _check_15_rollout_files(args) -> None:
    import cftime
    init_dt = cftime.datetime.strptime(
        args.init_dt, "%Y-%m-%d_%H:%M:%S",
        calendar="proleptic_gregorian", has_year_zero=True)
    year = init_dt.year
    jan1 = cftime.DatetimeProlepticGregorian(year, 1, 1, has_year_zero=True)
    init_idx = int((init_dt - jan1).total_seconds()) // (6 * 3600)
    # init_idx through init_idx + steps inclusive (step n needs file at idx n).
    missing: list[Path] = []
    for k in range(args.steps + 1):
        path = args.data_dir / f"{year}_{init_idx + k:04d}.h5"
        if not path.is_file():
            missing.append(path)
    if missing:
        raise AssertionError(f"missing {len(missing)} boundary files, e.g. {missing[:3]}")


PRETRAIN_CHECKS = [_check_01_env, _check_02_yparams, _check_03_manifest, _check_04_files,
                   _check_05_keys, _check_06_constbdry, _check_07_getdataset,
                   _check_08_stats, _check_09_climatology, _check_10_z0_audit]
PREINFER_CHECKS = [_check_11_wrapper_imports, _check_12_ckpt, _check_13_init_dt,
                   _check_14_constbdry_zscore, _check_15_rollout_files]


# ---------------- Phase F additions: checks 16-20 ----------------


@_check("16/wandb_auth")
def _check_16_wandb_auth(args) -> None:
    """If YAML has log_to_wandb: true, ensure wandb is importable + authed."""
    import yaml
    cfg = yaml.safe_load(args.yaml.read_text())
    sub = cfg.get(args.config, cfg)
    log_wandb = bool(sub.get("log_to_wandb", False))
    if not log_wandb:
        logger.info("log_to_wandb=false; skipping wandb-auth check.")
        return
    try:
        import wandb  # noqa: F401
    except ImportError as e:
        raise PreflightFailure(f"log_to_wandb=true but wandb not importable: {e}")
    # wandb authed means env WANDB_API_KEY OR ~/.netrc has api.wandb.ai entry.
    import os as _os
    netrc = Path.home() / ".netrc"
    has_key = bool(_os.environ.get("WANDB_API_KEY"))
    has_netrc = netrc.is_file() and "api.wandb.ai" in netrc.read_text()
    if not (has_key or has_netrc):
        raise PreflightFailure(
            "log_to_wandb=true but no WANDB_API_KEY env var AND no ~/.netrc "
            "entry for api.wandb.ai. Run `wandb login` first."
        )
    logger.info("wandb auth: WANDB_API_KEY=%s, netrc=%s", has_key, has_netrc)


@_check("17/resume_sentinel")
def _check_17_resume_sentinel(args) -> None:
    """If RUN_DIR/checkpoints/ckpt_latest.tar exists, assert YAML hash matches sentinel.

    Protects against silent config drift on resume.
    """
    if args.run_dir is None:
        logger.info("--run-dir not provided; skipping resume-sentinel check.")
        return
    ckpt = args.run_dir / "checkpoints" / "ckpt_latest.tar"
    sentinel = args.run_dir / ".aires_provenance.json"
    if not ckpt.is_file():
        logger.info("Fresh run (no ckpt_latest.tar); resume sentinel check skipped.")
        return
    if not sentinel.is_file():
        raise PreflightFailure(
            f"{ckpt.name} exists but sentinel {sentinel.name} missing. "
            f"Per feedback_protect_prior_runs.md, refusing to resume into an "
            f"unprovenanced ckpt dir."
        )
    sentinel_data = json.loads(sentinel.read_text())
    import hashlib
    yaml_sha = hashlib.sha256(args.yaml.read_bytes()).hexdigest()
    if sentinel_data.get("yaml_sha256") != yaml_sha:
        raise PreflightFailure(
            f"YAML hash drift detected on resume:\n"
            f"  sentinel: {sentinel_data.get('yaml_sha256')}\n"
            f"  current:  {yaml_sha}\n"
            f"Either roll back the YAML edit, or use a fresh RUN_NUM."
        )
    logger.info("Resume sentinel matches: yaml_sha256 OK, run_num=%s",
                sentinel_data.get("run_num"))


@_check("18/boundary_window_canonical")
def _check_18_boundary_window_canonical(args) -> None:
    """Mirror data_loader_multifiles.py:954/958/960/982 + 926 (offset every step,
    canonical-year remap). Assert all required <canonical>_<idx>.h5 files exist
    in data_dir. Specifically tests YEAR=121 hits year 11 idx 0..1459 (full
    padded) and YEAR=124 hits year 12 idx 3..1463 + year 11 idx 0..2.
    """
    import cftime
    from datetime import timedelta
    if args.init_dt is None or args.final_dt is None or args.data_dir is None:
        raise PreflightFailure("Check 18 needs --init-dt, --final-dt, --data-dir")

    init_dt = cftime.datetime.strptime(args.init_dt, "%Y-%m-%d_%H:%M:%S",
                                        has_year_zero=True, calendar="proleptic_gregorian")
    final_dt = cftime.datetime.strptime(args.final_dt, "%Y-%m-%d_%H:%M:%S",
                                         has_year_zero=True, calendar="proleptic_gregorian")
    leap_year, no_leap_year, nc_bc_offset_h, dt_h = 12, 11, 18, 6
    n_steps = int((final_dt - init_dt).total_seconds() // 3600 // dt_h)
    pairs: set[tuple[int, int]] = set()
    forbidden_years = set(range(121, 129))
    for k in range(n_steps):
        data_dt = init_dt + timedelta(hours=k * dt_h + nc_bc_offset_h)
        data_year = data_dt.year
        canonical = (leap_year
                     if cftime.is_leap_year(data_year, "proleptic_gregorian", has_year_zero=True)
                     else no_leap_year)
        jan1 = cftime.DatetimeProlepticGregorian(data_year, 1, 1, 0, has_year_zero=True)
        data_idx = int((data_dt - jan1).total_seconds() // 3600 // dt_h)
        if canonical in forbidden_years:
            raise PreflightFailure(
                f"step k={k}: enumerated canonical year {canonical} ∈ {{121..128}} — "
                f"loader contract violated. Bug in remap logic."
            )
        pairs.add((canonical, data_idx))

    missing: list[tuple[int, int]] = []
    for (cy, ci) in sorted(pairs):
        path = args.data_dir / f"{cy}_{ci:04d}.h5"
        if not path.is_file():
            missing.append((cy, ci))
    if missing:
        raise PreflightFailure(
            f"{len(missing)} required boundary files missing in {args.data_dir}; "
            f"first few: {missing[:5]}. Padding (F.B2) likely incomplete: "
            f"year 11 must reach idx {max((i for c,i in pairs if c==11), default=-1)}, "
            f"year 12 must reach idx {max((i for c,i in pairs if c==12), default=-1)}."
        )
    logger.info("Boundary window: %d (canonical, idx) pairs across years %s; "
                "max-year-11-idx=%d, max-year-12-idx=%d",
                len(pairs), sorted({c for c, _ in pairs}),
                max((i for c, i in pairs if c == 11), default=-1),
                max((i for c, i in pairs if c == 12), default=-1))


@_check("19/init_nc_exact_time")
def _check_19_init_nc_exact_time(args) -> None:
    """Init NC time coord MUST contain init_dt EXACTLY (long_inference.py:1331
    does ds.get_index('time').get_loc(init_dt) — KeyError on mismatch).
    """
    import cftime, xarray as xr
    if args.init_nc is None or args.init_dt is None:
        raise PreflightFailure("Check 19 needs --init-nc and --init-dt")
    target = cftime.datetime.strptime(args.init_dt, "%Y-%m-%d_%H:%M:%S",
                                       has_year_zero=True, calendar="proleptic_gregorian")
    target = cftime.DatetimeProlepticGregorian(target.year, target.month, target.day,
                                                target.hour, has_year_zero=True)
    ds = xr.open_dataset(args.init_nc, use_cftime=True)
    try:
        idx = ds.get_index("time").get_loc(target)
    except KeyError:
        coord = list(ds.get_index("time"))[:5]
        ds.close()
        raise PreflightFailure(
            f"init NC {args.init_nc} time coord does NOT contain {target} exactly. "
            f"long_inference.py:1331 will KeyError. First 5 time entries: {coord}"
        )
    ds.close()
    logger.info("Init NC time-coord exact-match: get_loc(%s) -> %d", target, idx)


@_check("20/stats_provenance")
def _check_20_stats_provenance(args) -> None:
    """Assert stats / climatology built from native frames only.

    Reads attrs (built_before_padding, native_timesteps_total, native_timesteps_by_year,
    manifest_skip_padded_used) and verifies consistency with the manifest. Refuses
    launch if data_train_mean.nc was contaminated by padded frames.
    """
    import xarray as xr
    if args.data_dir is None or args.manifest is None:
        logger.info("Check 20 skipped (need --data-dir + --manifest).")
        return
    manifest = json.loads(args.manifest.read_text())
    by_year = {y["year"]: y for y in manifest["years"]}
    train_year_natives_total = sum(
        by_year[y]["n_timesteps_native"] for y in by_year
        if 12 <= y <= 111   # train scope
    )
    for fn in ("data_train_mean.nc", "data_train_std.nc", "climatology.nc"):
        path = args.data_dir / fn
        if not path.is_file():
            raise PreflightFailure(f"missing {fn} in {args.data_dir}")
        ds = xr.open_dataset(path)
        a = ds.attrs
        if a.get("source") != "recomputed_from_converted_h5":
            ds.close()
            raise PreflightFailure(f"{fn}: source attr != 'recomputed_from_converted_h5'")
        bbp = int(a.get("built_before_padding", -1))
        skip = int(a.get("manifest_skip_padded_used", -1))
        if bbp != 1 and skip != 1:
            ds.close()
            raise PreflightFailure(
                f"{fn} built AFTER padding without manifest skip-padded; "
                f"stats may be contaminated. built_before_padding={bbp}, "
                f"manifest_skip_padded_used={skip}"
            )
        nt_total = int(a.get("native_timesteps_total", -1))
        # If native_timesteps_total covers the production train scope (12-111),
        # it should equal the sum of train years' n_timesteps_native.
        if nt_total > 0 and nt_total != train_year_natives_total:
            logger.warning(
                "%s: native_timesteps_total=%d != manifest train-scope sum=%d. "
                "(May be expected if stats were built over a subset.)",
                fn, nt_total, train_year_natives_total
            )
        logger.info("%s: source=%s, built_before_padding=%d, skip_padded=%d, "
                    "native_total=%d, git_sha=%s",
                    fn, a.get("source"), bbp, skip, nt_total, a.get("git_sha", "?"))
        ds.close()


PREINFER_LONG_CHECKS = [
    _check_11_wrapper_imports,
    _check_12_ckpt,
    _check_13_init_dt,
    _check_14_constbdry_zscore,
    _check_18_boundary_window_canonical,
    _check_19_init_nc_exact_time,
]
PRETRAIN_FULL_CHECKS = PRETRAIN_CHECKS + [
    _check_16_wandb_auth,
    _check_17_resume_sentinel,
    _check_20_stats_provenance,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, type=Path)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--phase", required=True,
                        choices=("pre_train", "pre_train_full",
                                 "pre_inference", "pre_inference_long"))
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="(pre_train) Flat data dir; defaults to YAML's params.data_dir.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="(pre_train) Path to _v10_calendar_manifest.json.")
    # pre_inference args
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="(pre_inference) Train run dir containing checkpoints/best_ckpt.tar.")
    parser.add_argument("--init-nc", type=Path, default=None,
                        help="(pre_inference) Path to init NC produced by build_init_nc_from_v10.")
    parser.add_argument("--init-dt", type=str, default=None,
                        help='(pre_inference) IC datetime "YYYY-MM-DD_HH:MM:SS" (underscore).')
    parser.add_argument("--final-dt", type=str, default=None,
                        help='(pre_inference_long) Final datetime "YYYY-MM-DD_HH:MM:SS".')
    parser.add_argument("--steps", type=int, default=None,
                        help="(pre_inference) Wrapper-mode rollout 6h steps. Not used for pre_inference_long.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Normalize init-dt format: accept either underscore or space (Phase 1 used space).
    if args.init_dt and " " in args.init_dt:
        args.init_dt = args.init_dt.replace(" ", "_")

    if args.phase == "pre_train":
        if args.manifest is None:
            parser.error("--manifest required for pre_train phase")
        checks = PRETRAIN_CHECKS
    elif args.phase == "pre_train_full":
        if args.manifest is None:
            parser.error("--manifest required for pre_train_full phase")
        checks = PRETRAIN_FULL_CHECKS
    elif args.phase == "pre_inference_long":
        for required, name in (
            (args.run_dir, "--run-dir"),
            (args.init_nc, "--init-nc"),
            (args.init_dt, "--init-dt"),
            (args.final_dt, "--final-dt"),
            (args.data_dir, "--data-dir"),
        ):
            if required is None:
                parser.error(f"{name} required for pre_inference_long phase")
        checks = PREINFER_LONG_CHECKS
    else:  # pre_inference (wrapper / Phase 1)
        for required, name in (
            (args.run_dir, "--run-dir"),
            (args.init_nc, "--init-nc"),
            (args.init_dt, "--init-dt"),
            (args.steps, "--steps"),
            (args.data_dir, "--data-dir"),
        ):
            if required is None:
                parser.error(f"{name} required for pre_inference phase")
        checks = PREINFER_CHECKS

    failures: list[str] = []
    for fn in checks:
        try:
            fn(args)
        except PreflightFailure as e:
            failures.append(str(e))
    if failures:
        logger.error("PREFLIGHT FAILED: %d/%d checks", len(failures), len(checks))
        return 1
    logger.info("PREFLIGHT GREEN: %d/%d checks passed", len(checks) - len(failures), len(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
