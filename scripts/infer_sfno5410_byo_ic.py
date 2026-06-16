#!/usr/bin/env python3
"""SFNO-5410 inference from a user-supplied (BYO) NetCDF initial condition.

Sister script to ``infer_sfno5410_blocking_h100_packed.py`` (which is the
production sim52-eval driver). This one:

  * reads a single-timestep IC from a user NetCDF (sim52-grid schema; see
    ``src/sfno_inference_5410/byo_ic.py`` for the contract);
  * supports an arbitrary forecast horizon in days (capped at 365 — the
    in-process loop has no year-rollover boundary handling, only
    ``long_inference.py`` does);
  * supports both deterministic single-member and perturbation ensembles;
  * uses sim52 template-year boundary forcing (51 non-leap / 52 leap), the
    only validated boundary mode (BYO boundary deferred per plan v5);
  * writes one NetCDF per ensemble member, with the user's calendar.

Output layout::

    <output_dir>/
      <prefix>_member000.nc        (and member001, member002, ... if ensemble)
      byo_inference_metadata.json

Each NetCDF has time index 0 = IC, indices 1..K = forecast leads at the
user's calendar (init_datetime + 6h*step).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Defaults — match production paths in infer_sfno5410_blocking_h100_packed.py
# ---------------------------------------------------------------------------
DEFAULT_BLOCKING_TREE = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/"
    "source_trees/forecast_modules/PanguPlasim"
)
DEFAULT_PANGU_TREE = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)
DEFAULT_YAML = DEFAULT_BLOCKING_TREE / "yaml_config/SFNO_PLASIM_H5_DERECHO_5410_deterministic.yaml"
DEFAULT_CHECKPOINT = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/"
    "sfno5410_blocking_epoch48_20260509/checkpoints/ckpt_epoch_48.tar"
)
DEFAULT_DATA_DIR = Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data")
DEFAULT_BIAS_DIR = Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/bias")
DEFAULT_CLIMATOLOGY = Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/sigma_data/climatology.nc")

VALID_PERTURBATION_TYPES = (
    "gaussian_noise",
    "gaussian_noise_n_minus_1",
    "perlin_noise",
)
HORIZON_DAYS_CAP = 365  # see module docstring (no year-rollover in in-process loop)
TIMEDELTA_HOURS = 6     # sim52 cadence


def parse_init_datetime(s: str) -> tuple[int, int, int, int]:
    """Parse YYYY-MM-DD_HH:MM:SS → (year, month, day, hour). Minute/second must be 0."""
    try:
        d = _dt.datetime.strptime(s, "%Y-%m-%d_%H:%M:%S")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--init-datetime must be YYYY-MM-DD_HH:MM:SS, got {s!r}: {exc}"
        ) from exc
    if d.minute != 0 or d.second != 0:
        raise argparse.ArgumentTypeError(
            f"--init-datetime sub-hour must be 00:00 (sim52 is 6-hourly), got {s!r}"
        )
    if d.hour % TIMEDELTA_HOURS != 0:
        raise argparse.ArgumentTypeError(
            f"--init-datetime hour must be a multiple of {TIMEDELTA_HOURS} "
            f"(sim52 is 6-hourly: 00, 06, 12, 18), got {s!r}"
        )
    return d.year, d.month, d.day, d.hour


def _is_leap_proleptic(year: int) -> bool:
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


def import_after_paths(blocking_tree: Path, pangu_tree: Path) -> dict[str, Any]:
    sys.path.insert(0, str(blocking_tree))
    sys.path.insert(1, str(pangu_tree))
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.cuda.amp as amp  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415
    from ensemble_inference import Stepper, to_ensemble_batch  # noqa: PLC0415
    from utils.YParams import YParams  # noqa: PLC0415
    from utils.data_loader_multifiles import datetime_class_from_calendar  # noqa: PLC0415

    return {
        "np": np,
        "torch": torch,
        "amp": amp,
        "xr": xr,
        "Stepper": Stepper,
        "to_ensemble_batch": to_ensemble_batch,
        "YParams": YParams,
        "datetime_class_from_calendar": datetime_class_from_calendar,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_member_nc(
    *,
    xr_mod,
    np_mod,
    out_path: Path,
    init_fields: dict[str, Any],
    surface_prediction,
    upper_air_prediction,
    diagnostic_prediction,
    lat,
    lon,
    sigma_levels,
    plevs,
    k_leads: int,
    init_datetime,
    timedelta_hours: int,
) -> None:
    """Write one ensemble member's NetCDF.

    ``surface_prediction``  : (K, N_surf_vars, lat, lon)  — denormalized leads
    ``upper_air_prediction``: (K, N_upper_air_vars, N_levels, lat, lon)
    ``diagnostic_prediction``: (K, N_diag_vars, lat, lon)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Time coord: cftime range so the output carries the user's calendar.
    times = [init_datetime + timedelta(hours=timedelta_hours * i) for i in range(k_leads + 1)]
    data_vars: dict[str, tuple[tuple[str, ...], Any]] = {}

    for i, var in enumerate(("pl", "tas")):
        arr = np_mod.empty((k_leads + 1, len(lat), len(lon)), dtype=np_mod.float32)
        arr[0] = init_fields[var]
        arr[1:] = surface_prediction[:, i]
        data_vars[var] = (("time", "lat", "lon"), arr)

    for i, var in enumerate(("ta", "ua", "va", "hus", "zg")):
        arr = np_mod.empty((k_leads + 1, 10, len(lat), len(lon)), dtype=np_mod.float32)
        arr[0] = init_fields[var]
        arr[1:] = upper_air_prediction[:, i]
        if var == "zg":
            data_vars[var] = (("time", "plev", "lat", "lon"), arr)
        else:
            data_vars[var] = (("time", "lev", "lat", "lon"), arr)

    pr = np_mod.empty((k_leads + 1, len(lat), len(lon)), dtype=np_mod.float32)
    pr[0] = init_fields["pr_6h"]
    pr[1:] = diagnostic_prediction[:, 0]
    data_vars["pr_6h"] = (("time", "lat", "lon"), pr)

    ds = xr_mod.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", times),
            "lev": ("lev", np_mod.asarray(sigma_levels, dtype=np_mod.float32)),
            "plev": ("plev", np_mod.asarray(plevs, dtype=np_mod.int32)),
            "lat": ("lat", np_mod.asarray(lat, dtype=np_mod.float64)),
            "lon": ("lon", np_mod.asarray(lon, dtype=np_mod.float64)),
        },
        attrs={
            "description": "SFNO-5410 BYO-IC forecast; time=0 is the user IC, "
                           "time>0 are forecast leads at +6h intervals",
            "boundary_forcing": "sim52 template year (51 non-leap / 52 leap)",
            "model": "SFNO-5410 (group emulator)",
        },
    )
    ds.to_netcdf(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SFNO-5410 inference from a user NetCDF initial condition.",
    )
    # User inputs
    parser.add_argument("--ic-nc", type=Path, required=True,
                        help="Path to single-timestep IC NetCDF (see byo_ic.py for schema).")
    parser.add_argument("--init-datetime", type=parse_init_datetime, required=True,
                        help='IC datetime as "YYYY-MM-DD_HH:MM:SS" (hour must be 00/06/12/18).')
    parser.add_argument("--horizon-days", type=float, required=True,
                        help=f"Forecast horizon in days (max {HORIZON_DAYS_CAP}).")
    parser.add_argument("--num-members", type=int, default=1,
                        help="Ensemble size. 1 = deterministic. >1 requires --epsilon-factor>0.")
    parser.add_argument("--epsilon-factor", type=float, default=0.0,
                        help="IC perturbation magnitude. 0.0 = no perturbation (deterministic).")
    parser.add_argument("--perturbation-type", type=str, default=None,
                        choices=VALID_PERTURBATION_TYPES,
                        help="Required when --epsilon-factor>0 and --num-members>1.")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Perturber seed for reproducibility (default: random).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for member NetCDFs + metadata.")
    parser.add_argument("--output-prefix", type=str, default="byo",
                        help="Filename prefix: <prefix>_member<NNN>.nc. Default: 'byo'.")
    parser.add_argument("--boundary-template-year", type=int, default=None,
                        help="sim52 template year for boundary forcing. "
                             "Default: 51 if init year non-leap, 52 if leap.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs in --output-dir.")

    # Path overrides (defaults match production)
    parser.add_argument("--blocking-tree", type=Path, default=DEFAULT_BLOCKING_TREE)
    parser.add_argument("--pangu-tree", type=Path, default=DEFAULT_PANGU_TREE)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="sim52 H5 dir for boundary template; not used as IC source.")
    parser.add_argument("--bias-data-dir", type=Path, default=DEFAULT_BIAS_DIR)
    parser.add_argument("--climatology-file", type=Path, default=DEFAULT_CLIMATOLOGY)

    args = parser.parse_args()

    # --- pre-flight CLI validation ---------------------------------------------------
    if args.horizon_days <= 0:
        parser.error(f"--horizon-days must be positive, got {args.horizon_days}")
    if args.horizon_days > HORIZON_DAYS_CAP:
        parser.error(
            f"--horizon-days={args.horizon_days} exceeds cap {HORIZON_DAYS_CAP}; "
            f"the in-process loop has no year-rollover boundary handling. "
            f"For multi-year, use the long_inference.py path (not yet wired)."
        )
    if args.num_members < 1:
        parser.error(f"--num-members must be >=1, got {args.num_members}")
    if args.epsilon_factor < 0:
        parser.error(f"--epsilon-factor must be >=0, got {args.epsilon_factor}")
    if args.num_members > 1 and args.epsilon_factor == 0.0:
        parser.error(
            f"--num-members={args.num_members} but --epsilon-factor=0.0; all members "
            f"would be bit-identical. Set --epsilon-factor>0 (with --perturbation-type) "
            f"or use --num-members=1."
        )
    if args.epsilon_factor > 0 and args.perturbation_type is None:
        parser.error(
            f"--epsilon-factor>0 requires --perturbation-type. "
            f"Valid: {VALID_PERTURBATION_TYPES}"
        )
    if args.num_members == 1 and args.epsilon_factor > 0:
        # Allowed but warn — 1 perturbed member is rarely what you want.
        print(
            f"[byo-warn] --num-members=1 with --epsilon-factor>0: a single "
            f"perturbed member is unusual; proceeding."
        )

    init_year, init_month, init_day, init_hour = args.init_datetime
    K = int(round(args.horizon_days * 24 / TIMEDELTA_HOURS))
    if K < 1:
        parser.error(
            f"--horizon-days={args.horizon_days} rounds to K={K} steps; "
            f"minimum 1 step (0.25 days)."
        )
    horizon_hours = K * TIMEDELTA_HOURS  # may differ slightly from horizon_days*24 due to rounding

    # Boundary template year auto-pick.
    if args.boundary_template_year is None:
        boundary_template_year = 52 if _is_leap_proleptic(init_year) else 51
    else:
        boundary_template_year = args.boundary_template_year

    # --- import upstream after sys.path setup ----------------------------------------
    mods = import_after_paths(args.blocking_tree, args.pangu_tree)
    np = mods["np"]
    torch = mods["torch"]
    amp = mods["amp"]
    xr = mods["xr"]
    Stepper = mods["Stepper"]
    YParams = mods["YParams"]
    to_ensemble_batch = mods["to_ensemble_batch"]
    datetime_class_from_calendar = mods["datetime_class_from_calendar"]

    # Repo module: BYO IC reader. Adding the AI-RES src dir to sys.path is intentional
    # — it lives outside the upstream trees we already inserted above.
    here = Path(__file__).resolve().parent
    src_dir = here.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from sfno_inference_5410.byo_ic import (  # noqa: PLC0415
        validate_byo_ic, stack_for_model,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SFNO-5410 inference (H100/packed-Derecho path).")
    torch.cuda.set_device(0)

    # --- output dir freshness gate ----------------------------------------------------
    out_dir = args.output_dir
    if out_dir.exists() and any(out_dir.glob(f"{args.output_prefix}_member*.nc")):
        if not args.force:
            raise RuntimeError(
                f"{out_dir} already contains {args.output_prefix}_member*.nc files; "
                f"pass --force to overwrite or pick a fresh --output-dir."
            )
        for nc in out_dir.glob(f"{args.output_prefix}_member*.nc"):
            nc.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- read + validate user IC FIRST (cheap, fail fast) ----------------------------
    raw_ic = validate_byo_ic(args.ic_nc)
    print(f"[byo] IC NetCDF validated: {args.ic_nc} ({len(raw_ic)} variables)")

    # --- build params -----------------------------------------------------------------
    # NB: init_datetime must be a cftime obj; the data_loader uses it to compute the
    # boundary date range, which is mapped to template years via leap_year/no_leap_year
    # below.
    params = YParams(os.path.abspath(args.yaml), "SFNO")
    params["run_num"] = "5410"
    params["world_size"] = 1
    params["local_rank"] = 0
    params["enable_amp"] = True
    params["has_diagnostic"] = bool(getattr(params, "diagnostic_variables", []))
    params["batch_size"] = 1
    params["num_ensemble_members"] = args.num_members
    params["ensemble_members_per_pred"] = args.num_members
    params["epsilon_factor"] = float(args.epsilon_factor)
    if args.perturbation_type is not None:
        params["perturbation_type"] = args.perturbation_type
    params["ensemble_inference_hours"] = horizon_hours
    params["best_checkpoint_path"] = str(args.checkpoint)
    params["save_forecasts"] = False
    params["data_dir"] = str(args.data_dir)
    params["bias_data_dir"] = str(args.bias_data_dir)
    params["climatology_file"] = str(args.climatology_file)
    # Pin both leap_year and no_leap_year so the data_loader picks the same boundary
    # template regardless of init year leap-ness. (Validation: the user's horizon must
    # not exceed the template year's H5 frame count, see HORIZON_DAYS_CAP.)
    params["leap_year"] = boundary_template_year
    params["no_leap_year"] = boundary_template_year

    datetime_class = datetime_class_from_calendar(params.calendar)
    init_dt = datetime_class(init_year, init_month, init_day, init_hour, has_year_zero=True)
    params["init_datetimes"] = [init_dt]
    save_basename = f"{args.output_prefix}_internal"
    params["save_basenames"] = [save_basename]
    params["output_dirs"] = [str(out_dir)]

    # --- construct Stepper (loads model, dataset, perturber if epsilon>0) ------------
    t0 = time.time()
    print(f"[byo] Constructing Stepper (model + dataset + boundary loader)...")
    stepper = Stepper([params], world_rank=0, use_6h_24h_model=False)
    stepper.model.eval()
    if stepper.model.training:
        raise RuntimeError("model.training stayed True after eval()")
    if args.random_seed is not None and args.epsilon_factor > 0:
        # Reseed the perturber for reproducibility.
        stepper.perturber.generator.manual_seed(args.random_seed)
        print(f"[byo] perturber reseeded with {args.random_seed}")

    # --- normalize the user IC using the dataset transforms --------------------------
    # surface_transform expects (N_surf_vars, lat, lon); upper_air expects
    # (N_upper_air_vars, N_levels, lat, lon). Returns the same shape, normalized.
    surf_vars = tuple(stepper.dataset.surface_variables)
    ua_vars = tuple(stepper.dataset.upper_air_variables)
    surf_raw, ua_raw = stack_for_model(raw_ic, surf_vars, ua_vars)
    surf_norm = stepper.dataset.surface_transform(torch.from_numpy(surf_raw))
    ua_norm = stepper.dataset.upper_air_transform(torch.from_numpy(ua_raw))

    # Tile by num_members along a new leading batch dim: (M, N_vars, ...).
    # to_ensemble_batch expects an existing batch dim, so unsqueeze(0) gives (1, ...);
    # tiling by M then yields (M, ...).
    input_surface = to_ensemble_batch(surf_norm.unsqueeze(0), args.num_members).to(
        stepper.device, dtype=torch.float32
    )
    input_upper_air = to_ensemble_batch(ua_norm.unsqueeze(0), args.num_members).to(
        stepper.device, dtype=torch.float32
    )

    # Apply perturber (only fires for ensemble runs with epsilon>0; gated above).
    if args.epsilon_factor > 0:
        print(f"[byo] applying {args.perturbation_type} perturbation, epsilon={args.epsilon_factor}")
        input_surface, input_upper_air = stepper.perturber(input_surface, input_upper_air)

    # --- pull boundary forcing from data_loader (we discard its IC) ------------------
    # Single-IC mode: data_loader yields exactly one batch.
    print("[byo] fetching boundary forcing from sim52 template...")
    batch_iter = iter(stepper.data_loader)
    batch = next(batch_iter)
    # batch[0]=input_surface (DISCARD), batch[1]=input_upper_air (DISCARD),
    # batch[2]=varying_boundary_data, batch[-1]=particle_idxs
    varying_boundary_data = batch[2].to(stepper.device, dtype=torch.float32)
    # Tile boundary by num_members. data_loader returned shape (1, K, N_b, lat, lon)
    # for a single IC; flatten gives (M, K, N_b, lat, lon).
    varying_boundary_data = to_ensemble_batch(varying_boundary_data, args.num_members)

    # constant_boundary_data is preloaded with batch_size=1 copies; tile by num_members.
    constant_boundary_data = to_ensemble_batch(stepper.constant_boundary_data, args.num_members)

    # --- K-step rollout --------------------------------------------------------------
    M = args.num_members
    N_surf = len(surf_vars)
    N_ua = len(ua_vars)
    N_lev = len(stepper.dataset.levels)
    H, W = surf_raw.shape[-2:]
    surface_prediction = np.empty((M, K, N_surf, H, W), dtype=np.float32)
    upper_prediction = np.empty((M, K, N_ua, N_lev, H, W), dtype=np.float32)
    diagnostic_prediction = np.empty((M, K, 1, H, W), dtype=np.float32)

    print(f"[byo] rolling out K={K} steps for {M} member(s)...")
    with torch.inference_mode(), amp.autocast(enabled=params.enable_amp):
        for step in range(1, K + 1):
            out_surface, out_upper_air, out_diagnostic, *_ = stepper.model(
                input_surface,
                constant_boundary_data,
                varying_boundary_data[:, step - 1],
                input_upper_air,
            )
            input_surface, input_upper_air = out_surface, out_upper_air
            surface_prediction[:, step - 1] = (
                stepper.dataset.surface_inv_transform(input_surface.detach().cpu()).numpy()
            )
            upper_prediction[:, step - 1] = (
                stepper.dataset.upper_air_inv_transform(input_upper_air.detach().cpu()).numpy()
            )
            diagnostic_prediction[:, step - 1] = (
                stepper.dataset.diagnostic_transform(out_diagnostic.detach().cpu()).numpy()
            )
            if step == 1 or step % 20 == 0 or step == K:
                print(f"[byo]   step {step}/{K} done")

    # --- write per-member NetCDFs ----------------------------------------------------
    lat = np.asarray(params.lat, dtype=np.float64)
    lon = np.asarray(params.lon, dtype=np.float64)
    written: list[dict[str, Any]] = []
    for m in range(M):
        out_nc = out_dir / f"{args.output_prefix}_member{m:03d}.nc"
        write_member_nc(
            xr_mod=xr,
            np_mod=np,
            out_path=out_nc,
            init_fields=raw_ic,
            surface_prediction=surface_prediction[m],
            upper_air_prediction=upper_prediction[m],
            diagnostic_prediction=diagnostic_prediction[m],
            lat=lat,
            lon=lon,
            sigma_levels=stepper.dataset.sigma_levels,
            plevs=stepper.dataset.levels,
            k_leads=K,
            init_datetime=init_dt,
            timedelta_hours=int(params.timedelta_hours),
        )
        written.append({
            "member": m,
            "path": str(out_nc),
            "size_bytes": out_nc.stat().st_size,
            "sha256": sha256_file(out_nc),
        })
        print(f"[byo] wrote {out_nc.name} ({out_nc.stat().st_size/1e6:.1f} MB)")

    # --- metadata --------------------------------------------------------------------
    metadata = {
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_device_name": torch.cuda.get_device_name(0),
        "ic_nc": str(args.ic_nc.resolve()),
        "ic_nc_sha256": sha256_file(args.ic_nc),
        "init_datetime": str(init_dt),
        "horizon_days": args.horizon_days,
        "K_steps": K,
        "horizon_hours": horizon_hours,
        "boundary_template_year": boundary_template_year,
        "num_members": args.num_members,
        "epsilon_factor": args.epsilon_factor,
        "perturbation_type": args.perturbation_type,
        "random_seed": args.random_seed,
        "output_dir": str(out_dir.resolve()),
        "output_prefix": args.output_prefix,
        "blocking_tree": str(args.blocking_tree),
        "pangu_tree": str(args.pangu_tree),
        "yaml": str(args.yaml),
        "yaml_sha256": sha256_file(args.yaml),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_dir": str(args.data_dir),
        "bias_data_dir": str(args.bias_data_dir),
        "climatology_file": str(args.climatology_file),
        "elapsed_seconds": time.time() - t0,
        "members_written": written,
    }
    metadata_path = out_dir / "byo_inference_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[byo] DONE. Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
