"""Rebuild an EMA-weights checkpoint from a per-epoch ``ckpt_mp0_vN.tar``.

Every epoch checkpoint of an EMA run carries ``ema_state`` next to the raw ``model_state``
(``PlasimTrainer.save_checkpoint``, plasim_trainer.py:858). The trainer writes
``best_ckpt_ema_mp0.tar`` only when the EMA validation loss improves, and it builds that
file with ``EMAModel.export_model_state``: the raw state_dict with every shadowed
trainable parameter replaced by its shadow, cast to the parameter's dtype (ema.py:268-301).
This script repeats that substitution from the file alone, so any epoch's EMA weights can
be rolled out by ``climate_rollout.py --ckpt <out>``.

Usage:
    python extract_ema_ckpt.py --ckpt ckpt_mp0_v22.tar --out ema_e23.tar
    python extract_ema_ckpt.py --ckpt ckpt_mp0_v22.tar --check-against best_ckpt_ema_mp0_e23_stable.tar

``--check-against`` compares every ``model_state`` tensor bitwise and prints
``EMA_EXTRACT_MATCH`` or ``EMA_EXTRACT_MISMATCH <first key> <max abs diff>``.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict

import torch


def _load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def ema_model_state(ckpt: dict) -> "OrderedDict[str, torch.Tensor]":
    """``model_state`` with the EMA shadow substituted, as ``export_model_state`` does."""
    if "ema_state" not in ckpt:
        raise KeyError("checkpoint has no ema_state (run trained without EMA?)")
    raw, ema = ckpt["model_state"], ckpt["ema_state"]
    unknown = sorted(set(ema) - set(raw))
    if unknown:
        raise KeyError(f"ema_state keys absent from model_state: {unknown[:5]}")
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, tensor in raw.items():
        if key in ema:
            if ema[key].shape != tensor.shape:
                raise ValueError(f"shape mismatch for {key}: {tuple(ema[key].shape)} vs {tuple(tensor.shape)}")
            out[key] = ema[key].to(tensor.dtype).detach().clone()
        else:
            out[key] = tensor.detach().clone()
    return out


def compare(a: dict, b: dict) -> tuple[bool, str, float]:
    if set(a) != set(b):
        missing = sorted(set(a) ^ set(b))
        return False, f"keys differ: {missing[:3]}", float("nan")
    for key in sorted(a):
        x, y = a[key], b[key]
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            diff = float((x.to(torch.complex128 if x.is_complex() else torch.float64)
                          - y.to(torch.complex128 if y.is_complex() else torch.float64)).abs().max())
            return False, key, diff
    return True, "", 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="per-epoch checkpoint holding ema_state")
    p.add_argument("--out", default=None, help="write the EMA checkpoint here")
    p.add_argument("--check-against", default=None, help="an EMA-best file to compare bitwise")
    args = p.parse_args()

    ckpt = _load(args.ckpt)
    state = ema_model_state(ckpt)
    n_ema = len(ckpt["ema_state"])
    print(f"source={args.ckpt} epoch={ckpt.get('epoch')} iters={ckpt.get('iters')} "
          f"ema_step={ckpt.get('ema_step')} shadowed={n_ema} tensors={len(state)}")

    if args.check_against:
        ref = _load(args.check_against)
        ok, key, diff = compare(state, ref["model_state"])
        tag = "EMA_EXTRACT_MATCH" if ok else f"EMA_EXTRACT_MISMATCH {key} {diff:.3e}"
        print(f"{tag} ref={args.check_against} ref_epoch={ref.get('epoch')}")
        if not ok:
            return 1

    if args.out:
        store = OrderedDict(model_state=state)
        for k in ("loss_state_dict", "comm_grid", "iters", "epoch"):
            if k in ckpt:
                store[k] = ckpt[k]
        store["ema_source"] = {"ckpt": args.ckpt, "ema_step": ckpt.get("ema_step"),
                               "ema_config": ckpt.get("ema_config")}
        torch.save(store, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
