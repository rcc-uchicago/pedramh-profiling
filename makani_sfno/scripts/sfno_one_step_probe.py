#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_bytes_and_summary(torch: Any, tensor: Any) -> tuple[bytes, dict[str, Any]]:
    t = tensor.detach().contiguous().cpu()
    arr = t.numpy()
    raw = arr.tobytes(order="C")
    finite = torch.isfinite(t) if not torch.is_complex(t) else torch.isfinite(t.real) & torch.isfinite(t.imag)
    if torch.is_complex(t):
        values = t.abs().float()
    else:
        values = t.float()
    summary = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device_before_cpu": str(tensor.device),
        "numel": int(t.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite_all": bool(finite.all().item()) if t.numel() else True,
        "l2": float(torch.linalg.vector_norm(values).item()) if t.numel() else 0.0,
        "mean": float(values.mean().item()) if t.numel() else 0.0,
        "std": float(values.std(unbiased=False).item()) if t.numel() else 0.0,
        "min": float(values.min().item()) if t.numel() else 0.0,
        "max": float(values.max().item()) if t.numel() else 0.0,
    }
    return raw, summary


def summarize_tensor(torch: Any, tensor: Any) -> dict[str, Any]:
    _, summary = tensor_bytes_and_summary(torch, tensor)
    return summary


def summarize_output(torch: Any, obj: Any) -> Any:
    if torch.is_tensor(obj):
        return summarize_tensor(torch, obj)
    if isinstance(obj, (list, tuple)):
        return [summarize_output(torch, item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): summarize_output(torch, v) for k, v in obj.items()}
    return {"type": type(obj).__name__, "repr": repr(obj)[:500]}


def aggregate_hash(torch: Any, named_tensors: list[tuple[str, Any]]) -> str:
    h = hashlib.sha256()
    for name, tensor in named_tensors:
        raw, summary = tensor_bytes_and_summary(torch, tensor)
        h.update(name.encode("utf-8"))
        h.update(json.dumps(
            {"shape": summary["shape"], "dtype": summary["dtype"]},
            sort_keys=True,
        ).encode("utf-8"))
        h.update(raw)
    return h.hexdigest()


def import_record(module_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {"module": module_name}
    try:
        module = importlib.import_module(module_name)
        record["version"] = getattr(module, "__version__", None)
        try:
            file_path = Path(inspect.getfile(module)).resolve()
            record["file"] = str(file_path)
            record["sha256"] = sha256_file(file_path)
        except Exception as exc:  # noqa: BLE001 - diagnostic path only
            record["file_error"] = repr(exc)
    except Exception as exc:  # noqa: BLE001 - diagnostic path only
        record["import_error"] = repr(exc)
    return record


def params_to_jsonable(params: Any) -> dict[str, str]:
    return {str(k): str(v) for k, v in sorted(params.params.items(), key=lambda kv: str(kv[0]))}


def prepare_params(train_mod: Any, args: argparse.Namespace, torch: Any) -> tuple[Any, int, int]:
    params = train_mod.YParams(os.path.abspath(args.yaml_config), args.config)
    params["enable_amp"] = not args.no_amp
    params["vae_loss"] = False
    params["mode"] = "train"
    params["test_iterations"] = 30
    params["run_iter"] = 1
    params["use_legacy_model"] = args.use_legacy_model
    params["has_diagnostic"] = bool(hasattr(params, "diagnostic_variables") and len(params.diagnostic_variables) > 0)
    if not hasattr(params, "num_ensemble_members"):
        params["num_ensemble_members"] = 1
    params["just_validate"] = True
    params["validation_epochs"] = [args.validation_epoch]
    params["validate_before_train"] = False
    params["curriculum_learning"] = params["curriculum_learning"] if hasattr(params, "curriculum_learning") else False
    params["balanced_learning"] = params["balanced_learning"] if hasattr(params, "balanced_learning") else False
    params["debug"] = False

    if "WORLD_SIZE" in os.environ:
        params["world_size"] = int(os.environ["WORLD_SIZE"])
    else:
        params["world_size"] = torch.cuda.device_count() if torch.cuda.is_available() else 1

    dist_args = SimpleNamespace(global_seed=args.global_seed)
    world_rank, local_rank = train_mod.setup_distributed(params, dist_args)

    save_exp_dir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    load_base = params.load_exp_dir if hasattr(params, "load_exp_dir") else params.exp_dir
    load_exp_dir = os.path.join(load_base, args.config, str(args.run_num))

    params["experiment_dir"] = os.path.abspath(save_exp_dir)
    params["checkpoint_dir_save"] = os.path.join(save_exp_dir, "checkpoints")
    params["checkpoint_dir_load"] = os.path.join(load_exp_dir, "checkpoints")
    params["plots_dir"] = os.path.join(save_exp_dir, "plots")
    params["spectra_dir"] = os.path.join(params["plots_dir"], "spectra")
    params["acc_dir"] = os.path.join(params["plots_dir"], "acc")
    params["gif_dir"] = os.path.join(params["plots_dir"], "gif")
    params["bias_dir"] = os.path.join(params["plots_dir"], "bias")
    params["validation_data_dir"] = os.path.join(save_exp_dir, "validation_data")
    params["checkpoint_path_globstr_save"] = os.path.join(params["checkpoint_dir_save"], "ckpt_epoch_*.tar")
    params["checkpoint_path_globstr_load"] = os.path.join(params["checkpoint_dir_load"], "ckpt_epoch_*.tar")
    params["best_checkpoint_path_save"] = os.path.join(params["checkpoint_dir_save"], "best_ckpt.tar")
    params["best_checkpoint_path_load"] = os.path.join(params["checkpoint_dir_load"], "best_ckpt.tar")
    params["latest_checkpoint_path_save"] = os.path.join(params["checkpoint_dir_save"], "ckpt_latest.tar")
    params["latest_checkpoint_path_load"] = os.path.join(params["checkpoint_dir_load"], "ckpt_latest.tar")
    params["checkpoint_save_interval"] = getattr(params, "checkpoint_save_interval", 10)
    params["max_checkpoints_to_keep"] = getattr(params, "max_checkpoints_to_keep", 5)
    params["resuming"] = True
    params["finetuning"] = False
    params["local_rank"] = local_rank
    params["log_to_wandb"] = False
    params["log_to_screen"] = world_rank == 0

    torch.manual_seed(world_rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True

    if world_rank == 0:
        for dirname in (
            params["experiment_dir"],
            params["checkpoint_dir_save"],
            params["spectra_dir"],
            params["acc_dir"],
            params["gif_dir"],
            params["validation_data_dir"],
        ):
            os.makedirs(dirname, exist_ok=True)
        if params.long_validation:
            os.makedirs(params["bias_dir"], exist_ok=True)

    if train_mod.dist.is_initialized():
        train_mod.dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    train_mod.params = params
    train_mod.world_rank = world_rank
    return params, world_rank, local_rank


def get_git_record(path: Path) -> dict[str, str | None]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        status = None
    return {"commit": commit, "status_short": status}


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump a one-step SFNO provenance probe.")
    parser.add_argument("--upstream-tree", required=True)
    parser.add_argument("--yaml-config", required=True)
    parser.add_argument("--config", default="SFNO")
    parser.add_argument("--run-num", default="5410")
    parser.add_argument("--validation-epoch", type=int, default=50)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="local")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--use-legacy-model", action="store_true")
    parser.add_argument("--save-input-tensors", action="store_true")
    parser.add_argument("--max-hooks", type=int, default=0, help="0 means hook every non-root module")
    args = parser.parse_args()

    upstream_tree = Path(args.upstream_tree).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.chdir(upstream_tree)
    sys.path.insert(0, str(upstream_tree))

    import torch  # noqa: PLC0415

    train_mod = importlib.import_module("train")
    params, world_rank, local_rank = prepare_params(train_mod, args, torch)

    if hasattr(params, "use_sigma_levels") and params.use_sigma_levels:
        params["diagnostic_acc"] = False
        params["diagnostic_spectra"] = False
        params["diagnostic_gif"] = False

    trainer = train_mod.Trainer(params, world_rank)
    trainer.setup_model()
    checkpoint_path = params.checkpoint_path_globstr_load.replace("*", str(args.validation_epoch))
    trainer.restore_checkpoint(checkpoint_path)
    trainer.epoch = trainer.startEpoch

    trainer.model.eval()
    model = trainer.model.module if train_mod.dist.is_initialized() else trainer.model
    model.eval()
    assert not trainer.model.training
    assert not model.training

    data = next(iter(trainer.valid_data_loader))
    device = trainer.device
    if params.predict_delta:
        if params.has_diagnostic:
            (val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air,
             val_target_diagnostic, val_target_surface_delta, val_target_upper_air_delta,
             val_varying_boundary_data, times) = [
                x.to(device, dtype=torch.float32, non_blocking=True) for x in data
            ]
        else:
            (val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air,
             val_target_surface_delta, val_target_upper_air_delta, val_varying_boundary_data, times) = [
                x.to(device, dtype=torch.float32, non_blocking=True) for x in data
            ]
            val_target_diagnostic = None
    else:
        if params.has_diagnostic:
            (val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air,
             val_target_diagnostic, val_varying_boundary_data, times) = [
                x.to(device, dtype=torch.float32, non_blocking=True) for x in data
            ]
        else:
            (val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air,
             val_varying_boundary_data, times) = [
                x.to(device, dtype=torch.float32, non_blocking=True) for x in data
            ]
            val_target_diagnostic = None

    if params.num_ensemble_members > 1:
        items = [val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_varying_boundary_data]
        if val_target_diagnostic is not None:
            items.insert(4, val_target_diagnostic)
        ens = [train_mod.to_ensemble_batch(item, params.num_ensemble_members) for item in items]
        if val_target_diagnostic is not None:
            (val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air,
             val_target_diagnostic, val_varying_boundary_data) = ens
        else:
            val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_varying_boundary_data = ens

    constant_boundary = trainer.constant_boundary_data
    varying_boundary_step0 = val_varying_boundary_data[:, 0]
    model_inputs = [
        ("surface", val_input_surface),
        ("constant_boundary", constant_boundary),
        ("varying_boundary_step0", varying_boundary_step0),
        ("upper_air", val_input_upper_air),
    ]

    activations: list[dict[str, Any]] = []
    handles = []

    def make_hook(name: str, module: Any):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if world_rank != 0:
                return
            activations.append({
                "index": len(activations),
                "name": name,
                "class": module.__class__.__module__ + "." + module.__class__.__qualname__,
                "output": summarize_output(torch, output),
            })
        return hook

    hook_count = 0
    for name, module in model.named_modules():
        if name == "":
            continue
        if args.max_hooks and hook_count >= args.max_hooks:
            break
        handles.append(module.register_forward_hook(make_hook(name, module)))
        hook_count += 1

    with torch.no_grad():
        precision_context = train_mod.autocast(enabled=params.enable_amp, device_type="cuda")
        with precision_context:
            if params.use_legacy_model:
                if params.has_diagnostic:
                    val_output_surface, val_output_upper_air, val_output_diagnostic, _, _ = model(
                        val_input_surface, constant_boundary, varying_boundary_step0, val_input_upper_air)
                else:
                    val_output_surface, val_output_upper_air, _, _ = model(
                        val_input_surface, constant_boundary, varying_boundary_step0, val_input_upper_air)
                    val_output_diagnostic = None
            else:
                if params.has_diagnostic:
                    val_output_surface, val_output_upper_air, val_output_diagnostic, _, _, _, _ = model(
                        val_input_surface, constant_boundary, varying_boundary_step0, val_input_upper_air)
                else:
                    val_output_surface, val_output_upper_air, _, _, _, _ = model(
                        val_input_surface, constant_boundary, varying_boundary_step0, val_input_upper_air)
                    val_output_diagnostic = None

    for handle in handles:
        handle.remove()

    if world_rank == 0:
        parameter_summaries = []
        parameter_hash_items = []
        for name, param in model.named_parameters():
            summary = summarize_tensor(torch, param)
            summary["name"] = name
            parameter_summaries.append(summary)
            parameter_hash_items.append((name, param))

        buffer_summaries = []
        for name, buffer in model.named_buffers():
            summary = summarize_tensor(torch, buffer)
            summary["name"] = name
            buffer_summaries.append(summary)

        module_records = []
        for name, module in model.named_modules():
            module_records.append({
                "name": name,
                "class": module.__class__.__module__ + "." + module.__class__.__qualname__,
                "file": inspect.getfile(module.__class__) if hasattr(module.__class__, "__module__") else None,
            })

        important_modules = [
            "train",
            "utils.data_loader_multifiles",
            "utils.YParams",
            "utils.metrics",
            "utils.perturbation",
            "utils.integrate",
            "ensemble_inference",
            "networks.pangu",
            "networks.pangu_legacy",
            "networks.modulus_sfno.sfnonet",
            "networks.modulus_sfno.s2convolutions",
            "networks.modulus_sfno.layers",
            "networks.modulus_sfno.factorizations",
            "torch",
            "torch.cuda",
            "torch_harmonics",
            "torch_harmonics.distributed",
            "torch_harmonics.quadrature",
            "torch_harmonics.legendre",
            "torch_harmonics.sht",
            "numpy",
            "h5py",
            "xarray",
        ]

        input_summaries = {name: summarize_tensor(torch, tensor) for name, tensor in model_inputs}
        input_summaries["varying_boundary_full"] = summarize_tensor(torch, val_varying_boundary_data)
        input_summaries["target_surface"] = summarize_tensor(torch, val_target_surface)
        input_summaries["target_upper_air"] = summarize_tensor(torch, val_target_upper_air)
        if val_target_diagnostic is not None:
            input_summaries["target_diagnostic"] = summarize_tensor(torch, val_target_diagnostic)
        input_summaries["times"] = summarize_tensor(torch, times)

        normalization_tensors = {}
        for attr in (
            "surface_mean", "surface_std", "upper_air_mean", "upper_air_std",
            "varying_boundary_mean", "varying_boundary_std",
            "diagnostic_mean", "diagnostic_std",
        ):
            value = getattr(trainer.valid_dataset, attr, None)
            if torch.is_tensor(value):
                normalization_tensors[attr] = summarize_tensor(torch, value)

        output_summaries = {
            "surface": summarize_tensor(torch, val_output_surface),
            "upper_air": summarize_tensor(torch, val_output_upper_air),
        }
        if val_output_diagnostic is not None:
            output_summaries["diagnostic"] = summarize_tensor(torch, val_output_diagnostic)

        report = {
            "tag": args.tag,
            "probe_kind": "sfno_one_step_first_validation_forward",
            "upstream_tree": str(upstream_tree),
            "yaml_config": str(Path(args.yaml_config).resolve()),
            "yaml_sha256": sha256_file(Path(args.yaml_config).resolve()),
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": sha256_file(Path(checkpoint_path).resolve()),
            "run_num": args.run_num,
            "config": args.config,
            "validation_epoch": args.validation_epoch,
            "world_rank": world_rank,
            "local_rank": local_rank,
            "world_size": params.world_size,
            "global_seed": args.global_seed,
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "git": get_git_record(upstream_tree),
            "params": params_to_jsonable(params),
            "model_training": bool(model.training),
            "wrapper_training": bool(trainer.model.training),
            "model_repr": repr(model),
            "module_records": module_records,
            "imports": [import_record(name) for name in important_modules],
            "input_aggregate_sha256": aggregate_hash(torch, model_inputs),
            "inputs": input_summaries,
            "normalization_tensors": normalization_tensors,
            "parameter_aggregate_sha256": aggregate_hash(torch, parameter_hash_items),
            "parameters": parameter_summaries,
            "buffers": buffer_summaries,
            "outputs": output_summaries,
            "activations": activations,
        }

        json_path = out_dir / f"{args.tag}_rank0_probe.json"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

        if args.save_input_tensors:
            tensor_path = out_dir / f"{args.tag}_rank0_inputs_outputs.pt"
            torch.save({
                "surface": val_input_surface.detach().cpu(),
                "upper_air": val_input_upper_air.detach().cpu(),
                "constant_boundary": constant_boundary.detach().cpu(),
                "varying_boundary_step0": varying_boundary_step0.detach().cpu(),
                "varying_boundary_full": val_varying_boundary_data.detach().cpu(),
                "output_surface": val_output_surface.detach().cpu(),
                "output_upper_air": val_output_upper_air.detach().cpu(),
                "output_diagnostic": None if val_output_diagnostic is None else val_output_diagnostic.detach().cpu(),
                "times": times.detach().cpu(),
            }, tensor_path)

        print(json_path)

    if train_mod.dist.is_initialized():
        train_mod.dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)
        train_mod.dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
