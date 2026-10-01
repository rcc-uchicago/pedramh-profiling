#!/usr/bin/env python3
"""port_golden_infer.py — makani-port M0/M3/M4: golden inference + parameter hashes per checkpoint.

For every checkpoint in ``makani_port/golden_checkpoints.json``: build the wrapper exactly as the
eval path does, record the sha256 of the checkpoint file, of the restored parameters and of the
state-dict structure, run ``rollout_one_ic`` twice from the listed IC (K leads; lead 1 is the
1-step prediction) and record a sha256 per lead. ``--save-dir`` additionally writes the K-lead
prediction as ``<tag>_K<K>.npy`` (float32, physical units) for later diagnosis.

Writes one JSON manifest. Equivalence between two manifests is ``port_golden_compare.py``'s job,
bitwise only. Prints ``PORT_GOLDEN_INFER_OK n=<k>`` iff every checkpoint ran and every in-process
control rollout was bitwise equal to the first.

Launch under ``python -m torch.distributed.run --standalone --nproc_per_node=1``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _sha_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha_tensor(t) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sha_named(items) -> tuple[str, str]:
    """(values sha, structure sha) over (name, tensor) pairs in their given order."""
    hv, hs = hashlib.sha256(), hashlib.sha256()
    for name, t in items:
        t = t.detach().cpu().contiguous()
        hs.update(f"{name}|{tuple(t.shape)}|{t.dtype};".encode())
        hv.update(name.encode())
        hv.update(t.numpy().tobytes())
    return hv.hexdigest(), hs.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True, type=Path, help="golden_checkpoints.json")
    p.add_argument("--holdout", required=True, type=Path, help="dir holding <year>.h5")
    p.add_argument("--out", required=True, type=Path, help="manifest JSON to write")
    p.add_argument("--save-dir", type=Path, default=None, help="write <tag>_K<K>.npy here")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import numpy as np
    import torch
    from sfno_inference.checkpoint_loader import build_wrapper_from_checkpoint, load_eval_params
    from sfno_inference.rollout_driver import _load_run_norm_stats, rollout_one_ic
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader
    from eval_inference import _resolve_device

    spec = json.loads(args.list.read_text())
    root = Path(spec["root"])
    year, frame, K = spec["ic"]["year"], spec["ic"]["start_frame"], spec["ic"]["K"]

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = _resolve_device("auto")
    manifest = {
        "kind": "port_golden_infer",
        "ic": spec["ic"],
        "torch": torch.__version__,
        "flags": {
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "cudnn.deterministic": torch.backends.cudnn.deterministic,
            "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "checkpoints": {},
    }
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for c in spec["checkpoints"]:
        tag, run_dir = c["tag"], root / c["run"]
        ckpt = run_dir / "training_checkpoints" / c["file"]
        t0 = time.time()
        entry = {"file": str(ckpt), "file_bytes": ckpt.stat().st_size, "file_sha256": _sha_file(ckpt)}

        ep = load_eval_params(run_dir, K=K)
        wrapper = build_wrapper_from_checkpoint(ep, ckpt, device=device)
        entry["params_sha256"], entry["params_struct_sha256"] = _sha_named(wrapper.named_parameters())
        entry["state_sha256"], entry["state_struct_sha256"] = _sha_named(wrapper.state_dict().items())
        entry["n_params"] = sum(1 for _ in wrapper.named_parameters())

        out_bias, out_scale = _load_run_norm_stats(ep, device)
        _, ds, _ = _plasim_get_dataloader(ep, str(args.holdout), device, mode="eval")
        fidx = next(i for i, f in enumerate(ds.files_paths) if Path(f).stem == str(year))
        ic = int(ds.file_offsets[fidx]) + frame

        def roll():
            return rollout_one_ic(wrapper=wrapper, dataset=ds, ic_global_idx=ic, eval_params=ep,
                                  device=device, out_bias=out_bias, out_scale=out_scale).prediction

        pred = roll()
        control = roll()
        deterministic = bool(torch.equal(pred, control))
        del control
        entry.update({
            "ic_global_idx": ic,
            "shape": list(pred.shape),
            "dtype": str(pred.dtype),
            "lead_sha256": [_sha_tensor(pred[k]) for k in range(pred.shape[0])],
            "full_sha256": _sha_tensor(pred),
            "control_bitwise": deterministic,
        })
        if args.save_dir is not None:
            np.save(args.save_dir / f"{tag}_K{K}.npy", pred.numpy().astype(np.float32, copy=False))
        manifest["checkpoints"][tag] = entry
        all_ok = all_ok and deterministic
        print(f"GOLDEN_INFER {tag} epoch={c['epoch']} shape={list(pred.shape)} "
              f"lead1={entry['lead_sha256'][0][:12]} full={entry['full_sha256'][:12]} "
              f"params={entry['params_sha256'][:12]} control_bitwise={deterministic} "
              f"s={time.time() - t0:.1f}", flush=True)
        del wrapper, pred, ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    n = len(manifest["checkpoints"])
    if all_ok and n == len(spec["checkpoints"]):
        print(f"PORT_GOLDEN_INFER_OK n={n} out={args.out}")
        return 0
    bad = [t for t, e in manifest["checkpoints"].items() if not e["control_bitwise"]]
    print(f"ERROR PORT_GOLDEN_INFER_NONDETERMINISTIC tags={bad}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
