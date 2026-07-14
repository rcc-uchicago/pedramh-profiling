#!/usr/bin/env python3
"""Small SFNO-5410 blocking-runtime z500 RMSE diagnostic with local paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import timedelta
from pathlib import Path


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
DEFAULT_OUT_DIR = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/"
    "20260509_blocking_h5_rmse_local_debug"
)


def import_after_paths(blocking_tree: Path, pangu_tree: Path):
    sys.path.insert(0, str(blocking_tree))
    sys.path.insert(1, str(pangu_tree))
    import h5py  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.cuda.amp as amp  # noqa: PLC0415
    from ensemble_inference import Stepper, to_ensemble_batch  # noqa: PLC0415
    from utils.YParams import YParams  # noqa: PLC0415
    from utils.data_loader_multifiles import datetime_class_from_calendar  # noqa: PLC0415

    return {
        "h5py": h5py,
        "np": np,
        "torch": torch,
        "amp": amp,
        "Stepper": Stepper,
        "to_ensemble_batch": to_ensemble_batch,
        "YParams": YParams,
        "datetime_class_from_calendar": datetime_class_from_calendar,
    }


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tensor(tensor, np_mod) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np_mod.ascontiguousarray(arr).tobytes()).hexdigest()


def h5_z500(h5py_mod, np_mod, data_dir: Path, year: int, idx: int):
    with h5py_mod.File(data_dir / f"{year}_{idx:04d}.h5", "r") as f:
        return np_mod.asarray(f["input"]["zg_50000.0"], dtype=np_mod.float32)


def lat_weighted_rmse(np_mod, pred, truth, lat) -> float:
    weights = np_mod.cos(np_mod.deg2rad(lat)).astype(np_mod.float64)
    err2 = (pred.astype(np_mod.float64) - truth.astype(np_mod.float64)) ** 2
    return float(np_mod.sqrt(np_mod.sum(err2 * weights[:, None]) / (np_mod.sum(weights) * pred.shape[1])))


def cftime_from_index(datetime_class, year: int, idx: int):
    return datetime_class(year, 1, 1, has_year_zero=True) + timedelta(hours=6 * idx)


def write_summary(np_mod, rows: list[dict[str, object]], out_path: Path) -> None:
    by_lead: dict[int, list[float]] = {}
    for row in rows:
        by_lead.setdefault(int(row["lead_hours"]), []).append(float(row["rmse_z500"]))
    summary = []
    for lead in sorted(by_lead):
        vals = np_mod.asarray(by_lead[lead], dtype=np_mod.float64)
        summary.append(
            {
                "lead_hours": lead,
                "n": int(vals.size),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)


def force_cpu_restore(stepper_cls, torch_mod) -> None:
    def restore_checkpoint_cpu(self, model, checkpoint_path):
        checkpoint = torch_mod.load(checkpoint_path, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(checkpoint["model_state"])
        except Exception:
            model.load_state_dict({key[7:]: val for key, val in checkpoint["model_state"].items()})
        self.iters = checkpoint["iters"]
        self.startEpoch = checkpoint["epoch"]
        print("START EPOCH:", self.startEpoch)

    stepper_cls.restore_checkpoint = restore_checkpoint_cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocking-tree", type=Path, default=DEFAULT_BLOCKING_TREE)
    parser.add_argument("--pangu-tree", type=Path, default=DEFAULT_PANGU_TREE)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bias-data-dir", type=Path, default=DEFAULT_BIAS_DIR)
    parser.add_argument("--climatology-file", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--years", default="121")
    parser.add_argument("--indices", default="0")
    parser.add_argument("--leads", default="6,24,72,120,240,336")
    parser.add_argument("--horizon-hours", type=int, default=336)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mods = import_after_paths(args.blocking_tree, args.pangu_tree)
    h5py_mod = mods["h5py"]
    np_mod = mods["np"]
    torch_mod = mods["torch"]
    amp = mods["amp"]
    Stepper = mods["Stepper"]
    YParams = mods["YParams"]
    to_ensemble_batch = mods["to_ensemble_batch"]
    datetime_class_from_calendar = mods["datetime_class_from_calendar"]

    if args.device == "cuda" and not torch_mod.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    use_cpu = args.device == "cpu" or (args.device == "auto" and not torch_mod.cuda.is_available())
    if use_cpu:
        force_cpu_restore(Stepper, torch_mod)
    elif torch_mod.cuda.is_available():
        torch_mod.cuda.set_device(0)

    years = parse_int_list(args.years)
    indices = parse_int_list(args.indices)
    leads = parse_int_list(args.leads)

    params = YParams(os.path.abspath(args.yaml), "SFNO")
    params["run_num"] = "5410"
    params["world_size"] = 1
    params["local_rank"] = 0
    params["enable_amp"] = True
    params["has_diagnostic"] = bool(getattr(params, "diagnostic_variables", []))
    params["num_ensemble_members"] = 1
    params["ensemble_members_per_pred"] = 1
    params["ensemble_inference_hours"] = args.horizon_hours
    params["batch_size"] = len(years) * len(indices)
    params["best_checkpoint_path"] = str(args.checkpoint)
    params["save_forecasts"] = False
    params["data_dir"] = str(args.data_dir)
    if hasattr(params, "bias_data_dir"):
        params["bias_data_dir"] = str(args.bias_data_dir)
    if hasattr(params, "climatology_file"):
        params["climatology_file"] = str(args.climatology_file)

    datetime_class = datetime_class_from_calendar(params.calendar)
    init_datetimes = []
    schedule = []
    for year in years:
        for idx in indices:
            init_datetimes.append(cftime_from_index(datetime_class, year, idx))
            schedule.append({"year": year, "ic_index": idx})
    params["init_datetimes"] = init_datetimes
    params["save_basenames"] = [str(args.out_dir / f"diagnostic_y{x['year']}_i{x['ic_index']}") for x in schedule]
    params["output_dirs"] = [str(args.out_dir) for _ in schedule]

    stepper = Stepper([params], world_rank=0, use_6h_24h_model=False)
    stepper.model.eval()
    if stepper.model.training:
        raise RuntimeError("model.training stayed True")

    z_var_index = list(stepper.dataset.upper_air_variables).index("zg")
    z_level_index = list(stepper.dataset.levels).index(50000)
    lat = np_mod.asarray(params.lat, dtype=np_mod.float64)

    rows: list[dict[str, object]] = []
    fingerprints: dict[str, object] = {}
    with torch_mod.inference_mode(), amp.autocast(enabled=params.enable_amp):
        for batch_i, data in enumerate(stepper.data_loader):
            input_surface, input_upper_air, varying_boundary_data = [
                x.to(stepper.device, dtype=torch_mod.float32) for x in data[:-1]
            ]
            particle_idxs = data[-1].detach().cpu().numpy().astype(int)
            constant_boundary_data = to_ensemble_batch(stepper.constant_boundary_data, 1)

            if batch_i == 0:
                fingerprints["normalized_input_surface_sha256"] = sha256_tensor(input_surface, np_mod)
                fingerprints["normalized_input_upper_air_sha256"] = sha256_tensor(input_upper_air, np_mod)
                fingerprints["normalized_varying_boundary_sha256"] = sha256_tensor(varying_boundary_data, np_mod)
                fingerprints["normalized_constant_boundary_sha256"] = sha256_tensor(constant_boundary_data, np_mod)

            for step in range(1, args.horizon_hours // params.timedelta_hours + 1):
                out_surface, out_upper_air, out_diagnostic, *_ = stepper.model(
                    input_surface,
                    constant_boundary_data,
                    varying_boundary_data[:, step - 1],
                    input_upper_air,
                )
                if batch_i == 0 and step == 1:
                    fingerprints["first_forward_surface_sha256"] = sha256_tensor(out_surface, np_mod)
                    fingerprints["first_forward_upper_air_sha256"] = sha256_tensor(out_upper_air, np_mod)
                    fingerprints["first_forward_diagnostic_sha256"] = sha256_tensor(out_diagnostic, np_mod)
                input_surface, input_upper_air = out_surface, out_upper_air
                lead_hours = step * params.timedelta_hours
                if lead_hours not in leads:
                    continue
                raw_upper = stepper.dataset.upper_air_inv_transform(input_upper_air.detach().cpu()).numpy()
                pred_z500 = raw_upper[:, z_var_index, z_level_index]
                for local_i, particle_idx in enumerate(particle_idxs):
                    sched = schedule[int(particle_idx)]
                    truth_idx = int(sched["ic_index"]) + step
                    truth_z500 = h5_z500(h5py_mod, np_mod, args.data_dir, int(sched["year"]), truth_idx)
                    rmse = lat_weighted_rmse(np_mod, pred_z500[local_i], truth_z500, lat)
                    rows.append(
                        {
                            "year": int(sched["year"]),
                            "ic_index": int(sched["ic_index"]),
                            "truth_index": truth_idx,
                            "lead_hours": lead_hours,
                            "rmse_z500": rmse,
                            "pred_mean": float(np_mod.mean(pred_z500[local_i])),
                            "truth_mean": float(np_mod.mean(truth_z500)),
                        }
                    )

    csv_path = args.out_dir / "z500_rmse_by_ic_lead.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.out_dir / "z500_rmse_summary_by_lead.json"
    write_summary(np_mod, rows, summary_path)

    metadata = {
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "torch": torch_mod.__version__,
        "torch_cuda": torch_mod.version.cuda,
        "cuda_available": torch_mod.cuda.is_available(),
        "run_device": str(stepper.device),
        "numpy": np_mod.__version__,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "yaml_sha256": sha256_file(args.yaml),
        "rows": rows,
        "fingerprints": fingerprints,
        "csv": str(csv_path),
        "summary": str(summary_path),
    }
    try:
        import torch_harmonics

        metadata["torch_harmonics"] = getattr(torch_harmonics, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        metadata["torch_harmonics"] = f"ERROR: {exc!r}"
    with (args.out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
