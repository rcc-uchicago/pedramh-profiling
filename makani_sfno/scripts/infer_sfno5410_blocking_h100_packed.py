#!/usr/bin/env python3
"""Full SFNO-5410 blocking-runtime inference using the packed Derecho env.

This is the valid Stampede3 production path for the blocking emulator:
exact blocking source tree + epoch-48 checkpoint + H100 CUDA execution.
It writes the raw NetCDF schema expected by ``scripts/score_5410.py``:
``inference/upstream_raw/Y{Y}_s{s:04d}_member000_y{Y:04d}.nc`` with
time index 0 as the IC state and indices 1..K as forecast leads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any


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
DEFAULT_RUN_ROOT = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/"
    "20260509_blocking_96ic_h100_packed_derecho_env_valid"
)
TEST_YEARS = tuple(range(121, 129))
IC_OFFSETS = (0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342)
SIGMA_LEVELS = (
    "0.03830000013113022",
    "0.11910000443458557",
    "0.21085000783205032",
    "0.3168500065803528",
    "0.4368000030517578",
    "0.5668000280857086",
    "0.6993500888347626",
    "0.8233500719070435",
    "0.9240999817848206",
    "0.983299970626831",
)
PLEVS_PA = (20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000)


def import_after_paths(blocking_tree: Path, pangu_tree: Path):
    sys.path.insert(0, str(blocking_tree))
    sys.path.insert(1, str(pangu_tree))
    import h5py  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.cuda.amp as amp  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415
    from ensemble_inference import Stepper, to_ensemble_batch  # noqa: PLC0415
    from utils.YParams import YParams  # noqa: PLC0415
    from utils.data_loader_multifiles import datetime_class_from_calendar  # noqa: PLC0415

    return {
        "h5py": h5py,
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


def sha256_array(np_mod, arr) -> str:
    return hashlib.sha256(np_mod.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_tensor(np_mod, tensor) -> str:
    return sha256_array(np_mod, tensor.detach().cpu().contiguous().numpy())


def cftime_from_index(datetime_class, year: int, idx: int):
    return datetime_class(year, 1, 1, has_year_zero=True) + timedelta(hours=6 * idx)


def plan(limit_ics: int | None = None) -> list[dict[str, int]]:
    entries = [{"year": year, "ic_index": idx} for year in TEST_YEARS for idx in IC_OFFSETS]
    if limit_ics is not None:
        if limit_ics < 1:
            raise ValueError("--limit-ics must be >= 1")
        entries = entries[:limit_ics]
    return entries


def h5_key(var: str, level_i: int | None = None) -> str:
    if var in ("pl", "tas", "pr_6h"):
        return var
    if var in ("ta", "ua", "va", "hus"):
        if level_i is None:
            raise ValueError(f"{var} requires level_i")
        return f"{var}_{SIGMA_LEVELS[level_i]}"
    if var == "zg":
        if level_i is None:
            raise ValueError("zg requires level_i")
        return f"zg_{PLEVS_PA[level_i]}.0"
    raise ValueError(f"unknown h5 variable {var!r}")


def read_ic_raw(h5py_mod, np_mod, data_dir: Path, year: int, idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with h5py_mod.File(data_dir / f"{year}_{idx:04d}.h5", "r") as f:
        inp = f["input"]
        for var in ("pl", "tas", "pr_6h"):
            out[var] = np_mod.asarray(inp[h5_key(var)], dtype=np_mod.float32)
        for var in ("ta", "ua", "va", "hus", "zg"):
            out[var] = np_mod.stack(
                [np_mod.asarray(inp[h5_key(var, i)], dtype=np_mod.float32) for i in range(10)],
                axis=0,
            )
    return out


def lat_weighted_rmse(np_mod, pred, truth, lat) -> float:
    weights = np_mod.cos(np_mod.deg2rad(lat)).astype(np_mod.float64)
    err2 = (pred.astype(np_mod.float64) - truth.astype(np_mod.float64)) ** 2
    return float(np_mod.sqrt(np_mod.sum(err2 * weights[:, None]) / (np_mod.sum(weights) * pred.shape[1])))


def write_raw_nc(
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
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_coord = np_mod.arange(k_leads + 1, dtype=np_mod.int32) * 6
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
            "time": ("time", time_coord),
            "lev": ("lev", np_mod.asarray(sigma_levels, dtype=np_mod.float32)),
            "plev": ("plev", np_mod.asarray(plevs, dtype=np_mod.int32)),
            "lat": ("lat", np_mod.asarray(lat, dtype=np_mod.float64)),
            "lon": ("lon", np_mod.asarray(lon, dtype=np_mod.float64)),
        },
        attrs={"description": "SFNO-5410 blocking runtime raw forecast; time=0 is IC, time>0 are leads"},
    )
    ds["time"].attrs["units"] = "hours"
    ds.to_netcdf(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--blocking-tree", type=Path, default=DEFAULT_BLOCKING_TREE)
    parser.add_argument("--pangu-tree", type=Path, default=DEFAULT_PANGU_TREE)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bias-data-dir", type=Path, default=DEFAULT_BIAS_DIR)
    parser.add_argument("--climatology-file", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--K", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-ics", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mods = import_after_paths(args.blocking_tree, args.pangu_tree)
    h5py_mod = mods["h5py"]
    np_mod = mods["np"]
    torch_mod = mods["torch"]
    amp = mods["amp"]
    xr_mod = mods["xr"]
    Stepper = mods["Stepper"]
    YParams = mods["YParams"]
    to_ensemble_batch = mods["to_ensemble_batch"]
    datetime_class_from_calendar = mods["datetime_class_from_calendar"]

    if not torch_mod.cuda.is_available():
        raise RuntimeError("CUDA is required for the valid SFNO-5410 blocking path")
    torch_mod.cuda.set_device(0)

    run_plan = plan(args.limit_ics)
    raw_dir = args.run_root / "inference" / "upstream_raw"
    if raw_dir.exists() and any(raw_dir.glob("*.nc")):
        if not args.force:
            raise RuntimeError(f"{raw_dir} already contains NetCDFs; pass --force to replace")
        for path in raw_dir.glob("*.nc"):
            path.unlink()
    raw_dir.mkdir(parents=True, exist_ok=True)

    params = YParams(os.path.abspath(args.yaml), "SFNO")
    params["run_num"] = "5410"
    params["world_size"] = 1
    params["local_rank"] = 0
    params["enable_amp"] = True
    params["has_diagnostic"] = bool(getattr(params, "diagnostic_variables", []))
    params["num_ensemble_members"] = 1
    params["ensemble_members_per_pred"] = 1
    params["ensemble_inference_hours"] = args.K * int(params.timedelta_hours)
    params["batch_size"] = args.batch_size
    params["best_checkpoint_path"] = str(args.checkpoint)
    params["save_forecasts"] = False
    params["data_dir"] = str(args.data_dir)
    params["bias_data_dir"] = str(args.bias_data_dir)
    params["climatology_file"] = str(args.climatology_file)

    datetime_class = datetime_class_from_calendar(params.calendar)
    params["init_datetimes"] = [
        cftime_from_index(datetime_class, item["year"], item["ic_index"]) for item in run_plan
    ]
    params["save_basenames"] = [
        str(raw_dir / f"Y{item['year']}_s{item['ic_index']:04d}") for item in run_plan
    ]
    params["output_dirs"] = [str(raw_dir) for _ in run_plan]

    metadata = {
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "torch": torch_mod.__version__,
        "torch_cuda": torch_mod.version.cuda,
        "cuda_available": torch_mod.cuda.is_available(),
        "torch_device_name": torch_mod.cuda.get_device_name(0),
        "numpy": np_mod.__version__,
        "run_root": str(args.run_root),
        "raw_dir": str(raw_dir),
        "blocking_tree": str(args.blocking_tree),
        "pangu_tree": str(args.pangu_tree),
        "yaml": str(args.yaml),
        "yaml_sha256": sha256_file(args.yaml),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_dir": str(args.data_dir),
        "K": args.K,
        "batch_size": args.batch_size,
        "plan_size": len(run_plan),
    }
    try:
        import torch_harmonics

        metadata["torch_harmonics"] = getattr(torch_harmonics, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        metadata["torch_harmonics"] = f"ERROR: {exc!r}"

    t0 = time.time()
    stepper = Stepper([params], world_rank=0, use_6h_24h_model=False)
    stepper.model.eval()
    if stepper.model.training:
        raise RuntimeError("model.training stayed True")

    lat = np_mod.asarray(params.lat, dtype=np_mod.float64)
    lon = np_mod.asarray(params.lon, dtype=np_mod.float64)
    rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    z_var_index = list(stepper.dataset.upper_air_variables).index("zg")
    z_level_index = list(stepper.dataset.levels).index(50000)

    with torch_mod.inference_mode(), amp.autocast(enabled=params.enable_amp):
        for batch_i, data in enumerate(stepper.data_loader):
            input_surface, input_upper_air, varying_boundary_data = [
                x.to(stepper.device, dtype=torch_mod.float32) for x in data[:-1]
            ]
            particle_idxs = data[-1].detach().cpu().numpy().astype(int)
            constant_boundary_data = to_ensemble_batch(stepper.constant_boundary_data, 1)

            if batch_i == 0:
                fingerprints["normalized_input_surface_sha256"] = sha256_tensor(np_mod, input_surface)
                fingerprints["normalized_input_upper_air_sha256"] = sha256_tensor(np_mod, input_upper_air)
                fingerprints["normalized_varying_boundary_sha256"] = sha256_tensor(np_mod, varying_boundary_data)
                fingerprints["normalized_constant_boundary_sha256"] = sha256_tensor(np_mod, constant_boundary_data)

            bsz = int(input_surface.shape[0])
            surface_prediction = np_mod.empty((bsz, args.K, 2, 64, 128), dtype=np_mod.float32)
            upper_prediction = np_mod.empty((bsz, args.K, 5, 10, 64, 128), dtype=np_mod.float32)
            diagnostic_prediction = np_mod.empty((bsz, args.K, 1, 64, 128), dtype=np_mod.float32)

            for step in range(1, args.K + 1):
                out_surface, out_upper_air, out_diagnostic, *_ = stepper.model(
                    input_surface,
                    constant_boundary_data,
                    varying_boundary_data[:, step - 1],
                    input_upper_air,
                )
                if batch_i == 0 and step == 1:
                    fingerprints["first_forward_surface_sha256"] = sha256_tensor(np_mod, out_surface)
                    fingerprints["first_forward_upper_air_sha256"] = sha256_tensor(np_mod, out_upper_air)
                    fingerprints["first_forward_diagnostic_sha256"] = sha256_tensor(np_mod, out_diagnostic)

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

            for local_i, particle_idx in enumerate(particle_idxs):
                item = run_plan[int(particle_idx)]
                year = int(item["year"])
                ic_index = int(item["ic_index"])
                init_fields = read_ic_raw(h5py_mod, np_mod, args.data_dir, year, ic_index)
                out_nc = raw_dir / f"Y{year}_s{ic_index:04d}_member000_y{year:04d}.nc"
                write_raw_nc(
                    xr_mod=xr_mod,
                    np_mod=np_mod,
                    out_path=out_nc,
                    init_fields=init_fields,
                    surface_prediction=surface_prediction[local_i],
                    upper_air_prediction=upper_prediction[local_i],
                    diagnostic_prediction=diagnostic_prediction[local_i],
                    lat=lat,
                    lon=lon,
                    sigma_levels=stepper.dataset.sigma_levels,
                    plevs=stepper.dataset.levels,
                    k_leads=args.K,
                )
                rows.append(
                    {
                        "year": year,
                        "ic_index": ic_index,
                        "raw_nc": str(out_nc),
                        "size": out_nc.stat().st_size,
                        "sha256": sha256_file(out_nc),
                    }
                )

                if year == 121 and ic_index == 0:
                    for lead in (6, 24, 120, 336):
                        if lead > args.K * int(params.timedelta_hours):
                            continue
                        step = lead // int(params.timedelta_hours)
                        pred_z500 = upper_prediction[local_i, step - 1, z_var_index, z_level_index]
                        truth = read_ic_raw(h5py_mod, np_mod, args.data_dir, year, ic_index + step)["zg"][z_level_index]
                        sanity_rows.append(
                            {
                                "lead_hours": lead,
                                "rmse_z500": lat_weighted_rmse(np_mod, pred_z500, truth, lat),
                            }
                        )
            print(f"[infer] completed batch {batch_i + 1}; total files={len(rows)}", flush=True)

    sanity_pass = True
    thresholds = {6: 5.0, 24: 10.0, 120: 30.0, 336: 90.0}
    for row in sanity_rows:
        if row["rmse_z500"] > thresholds[int(row["lead_hours"])]:
            sanity_pass = False
    expected_files = len(run_plan)
    if len(rows) != expected_files:
        sanity_pass = False
    if len(list(raw_dir.glob("*.nc"))) != expected_files:
        sanity_pass = False

    metadata["elapsed_seconds"] = time.time() - t0
    metadata["completed_netcdf_forecasts"] = len(rows)
    metadata["expected_netcdf_forecasts"] = expected_files
    metadata["fingerprints"] = fingerprints
    metadata["sanity_gate"] = {
        "passed": sanity_pass,
        "name": "Y121_s0000_z500_rmse_thresholds",
        "thresholds": thresholds,
        "rows": sanity_rows,
    }

    manifest_csv = args.run_root / "inference" / "raw_manifest.csv"
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata["manifest_csv"] = str(manifest_csv)

    metadata_path = args.run_root / "inference" / "inference_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(json.dumps(metadata, indent=2, default=str))

    if not sanity_pass:
        raise RuntimeError(f"sanity gate failed: {metadata['sanity_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
