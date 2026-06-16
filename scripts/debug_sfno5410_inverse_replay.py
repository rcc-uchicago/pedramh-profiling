#!/usr/bin/env python3
"""Replay Derecho SFNO-5410 block0 spectra through the local inverse transform."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
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
DEFAULT_REF_NPZ = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_fingerprints/"
    "sfno5410_y121_s0_block0_deep_hooks_20260509/block0_deep_hooks_cpu_fp32_no_autocast.npz"
)
DEFAULT_OUT_DIR = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/"
    "20260509_inverse_replay_y121s0"
)
DEFAULT_DATA_DIR = Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data")
DEFAULT_BIAS_DIR = Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/bias")
DEFAULT_CLIMATOLOGY = Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/sigma_data/climatology.nc")


def import_after_paths(blocking_tree: Path, pangu_tree: Path):
    sys.path.insert(0, str(blocking_tree))
    sys.path.insert(1, str(pangu_tree))

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch.cuda.amp as amp  # noqa: PLC0415
    from torch.amp import autocast  # noqa: PLC0415
    from ensemble_inference import Stepper, to_ensemble_batch  # noqa: PLC0415
    from utils.YParams import YParams  # noqa: PLC0415
    from utils.data_loader_multifiles import datetime_class_from_calendar  # noqa: PLC0415

    return {
        "np": np,
        "torch": torch,
        "amp": amp,
        "autocast": autocast,
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


def stats(np_mod, arr) -> dict[str, Any]:
    arr = np_mod.ascontiguousarray(arr)
    if np_mod.iscomplexobj(arr):
        stat_arr = np_mod.stack([arr.real, arr.imag], axis=-1).astype(np_mod.float64, copy=False)
        note = "complex stats over real/imag view"
    else:
        stat_arr = arr.astype(np_mod.float64, copy=False)
        note = ""
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np_mod.nanmin(stat_arr)),
        "max": float(np_mod.nanmax(stat_arr)),
        "mean": float(np_mod.nanmean(stat_arr)),
        "std": float(np_mod.nanstd(stat_arr)),
        "sha256": sha256_array(np_mod, arr),
        "note": note,
    }


def diff_stats(np_mod, got, ref) -> dict[str, Any]:
    got = np_mod.ascontiguousarray(got)
    ref = np_mod.ascontiguousarray(ref)
    diff = got.astype(np_mod.float64) - ref.astype(np_mod.float64)
    return {
        "sha256_equal": sha256_array(np_mod, got) == sha256_array(np_mod, ref),
        "rmse": float(np_mod.sqrt(np_mod.mean(diff * diff))),
        "mean_abs": float(np_mod.mean(np_mod.abs(diff))),
        "max_abs": float(np_mod.max(np_mod.abs(diff))),
        "mean_diff": float(np_mod.mean(diff)),
    }


def cftime_from_index(datetime_class, year: int, index: int):
    return datetime_class(year, 1, 1, has_year_zero=True) + timedelta(hours=6 * index)


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


def build_stepper(mods, args: argparse.Namespace, amp_enabled: bool):
    torch_mod = mods["torch"]
    Stepper = mods["Stepper"]
    YParams = mods["YParams"]
    datetime_class_from_calendar = mods["datetime_class_from_calendar"]

    params = YParams(os.path.abspath(args.yaml), "SFNO")
    params["run_num"] = "5410"
    params["world_size"] = 1
    params["local_rank"] = 0
    params["enable_amp"] = amp_enabled
    params["has_diagnostic"] = bool(getattr(params, "diagnostic_variables", []))
    params["num_ensemble_members"] = 1
    params["ensemble_members_per_pred"] = 1
    params["ensemble_inference_hours"] = 6
    params["batch_size"] = 1
    params["best_checkpoint_path"] = str(args.checkpoint)
    params["save_forecasts"] = False
    if hasattr(params, "data_dir"):
        params["data_dir"] = str(args.data_dir)
    if hasattr(params, "bias_data_dir"):
        params["bias_data_dir"] = str(args.bias_data_dir)
    if hasattr(params, "climatology_file"):
        params["climatology_file"] = str(args.climatology_file)

    datetime_class = datetime_class_from_calendar(params.calendar)
    params["init_datetimes"] = [cftime_from_index(datetime_class, args.year, args.index)]
    params["save_basenames"] = [str(Path(args.out_dir) / f"replay_y{args.year}_i{args.index}")]
    params["output_dirs"] = [str(Path(args.out_dir))]

    force_cpu_restore(Stepper, torch_mod)
    stepper = Stepper([params], world_rank=0, use_6h_24h_model=False)
    stepper.model.eval()
    if stepper.model.training:
        raise RuntimeError("model.training stayed True")
    return stepper


def base_model(model):
    return model.module if hasattr(model, "module") else model


def to_numpy(torch_mod, tensor):
    return tensor.detach().cpu().contiguous().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocking-tree", type=Path, default=DEFAULT_BLOCKING_TREE)
    parser.add_argument("--pangu-tree", type=Path, default=DEFAULT_PANGU_TREE)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ref-npz", type=Path, default=DEFAULT_REF_NPZ)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bias-data-dir", type=Path, default=DEFAULT_BIAS_DIR)
    parser.add_argument("--climatology-file", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--year", type=int, default=121)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mods = import_after_paths(args.blocking_tree, args.pangu_tree)
    np_mod = mods["np"]
    torch_mod = mods["torch"]
    autocast = mods["autocast"]

    stepper = build_stepper(mods, args, amp_enabled=args.amp)
    model = base_model(stepper.model)
    spectral = model.blocks[0].filter.filter
    spectral_device = next(spectral.inverse_transform.buffers()).device
    ref = np_mod.load(args.ref_npz)

    rows: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}

    with torch_mod.inference_mode(), autocast(enabled=False, device_type="cuda"):
        ref_contracted = torch_mod.from_numpy(ref["filter_after_contraction_full_spectrum_complex"]).to(spectral_device)
        replay_pre_bias = spectral.inverse_transform(ref_contracted.contiguous())
        outputs["inverse_from_derecho_contracted_pre_bias"] = to_numpy(torch_mod, replay_pre_bias)
        rows.append(
            {
                "name": "inverse_from_derecho_contracted_pre_bias",
                **stats(np_mod, outputs["inverse_from_derecho_contracted_pre_bias"]),
                **{
                    f"vs_derecho_{k}": v
                    for k, v in diff_stats(
                        np_mod,
                        outputs["inverse_from_derecho_contracted_pre_bias"],
                        ref["filter_after_inverse_transform_pre_bias"],
                    ).items()
                },
            }
        )

        ref_forward = torch_mod.from_numpy(ref["filter_after_forward_transform_complex"]).to(spectral_device)
        replay_residual = spectral.inverse_transform(ref_forward.contiguous())
        outputs["inverse_from_derecho_forward_spectrum"] = to_numpy(torch_mod, replay_residual)
        rows.append(
            {
                "name": "inverse_from_derecho_forward_spectrum",
                **stats(np_mod, outputs["inverse_from_derecho_forward_spectrum"]),
                **{
                    f"vs_derecho_{k}": v
                    for k, v in diff_stats(
                        np_mod,
                        outputs["inverse_from_derecho_forward_spectrum"],
                        ref["input_to_block0_filter"],
                    ).items()
                },
            }
        )

    npz_path = args.out_dir / ("inverse_replay_amp.npz" if args.amp else "inverse_replay_fp32.npz")
    np_mod.savez_compressed(npz_path, **outputs)
    csv_path = args.out_dir / ("inverse_replay_amp.csv" if args.amp else "inverse_replay_fp32.csv")
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "torch": torch_mod.__version__,
        "torch_cuda": torch_mod.version.cuda,
        "cuda_available": torch_mod.cuda.is_available(),
        "cpu_capability": torch_mod.backends.cpu.get_cpu_capability(),
        "spectral_device": str(spectral_device),
        "numpy": np_mod.__version__,
        "blocking_tree": str(args.blocking_tree),
        "pangu_tree": str(args.pangu_tree),
        "yaml": str(args.yaml),
        "yaml_sha256": sha256_file(args.yaml),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_dir": str(args.data_dir),
        "bias_data_dir": str(args.bias_data_dir),
        "climatology_file": str(args.climatology_file),
        "ref_npz": str(args.ref_npz),
        "ref_npz_sha256": sha256_file(args.ref_npz),
        "spectral_repr": repr(spectral),
        "inverse_transform_repr": repr(spectral.inverse_transform),
        "npz": str(npz_path),
        "csv": str(csv_path),
        "rows": rows,
    }
    report_path = args.out_dir / ("inverse_replay_amp_report.json" if args.amp else "inverse_replay_fp32_report.json")
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({"report": str(report_path), "csv": str(csv_path), "npz": str(npz_path)}, indent=2))


if __name__ == "__main__":
    main()
