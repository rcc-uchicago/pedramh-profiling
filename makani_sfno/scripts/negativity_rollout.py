#!/usr/bin/env python3
"""negativity_rollout.py — how often do moisture channels go negative in a rollout?

Question (2026-09-24, operator): ACE2 clamps its water and precipitation fields at
zero (``force_positive_names``, ACE2_retrain/config_polaris.yaml:172-188). Before
deciding whether makani needs the same, measure how often our rollouts produce
negative ``RELHUM_l00..l17``, ``RHREFHT``, ``PRECT`` (and ``TMQ``,
``SOILWATER_10CM``, which are physically >= 0 too).

Measurement only: the step body is ``climate_driver.stream_rollout``, unchanged, and
the prediction is never modified (CLAUDE.md #1). Per lead and per selected channel,
in physical units (``z * global_std + global_mean``):

  neg_frac   area-weighted fraction of grid cells < 0
  min        minimum value
  neg_mean   area-weighted mean of min(x, 0)   (how negative, on average)

A non-finite prediction stops the rollout (a measurement, as in the driver).

Output: one ``.npz`` per member + tokens
  NEGATIVITY_ROLLOUT_OK member=… steps=… out=…           rc 0
  NEGATIVITY_ROLLOUT_TRUNCATED step=… member=… out=…      rc 3 (output written)
  ERROR <reason>                                          rc 1

Launch like climate_rollout.py (torch.distributed.run, 1 GPU).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_CHANNELS = r"^(RELHUM_l\d\d|RHREFHT|PRECT|TMQ|SOILWATER_10CM)$"


def select_channels(names, pattern: str = DEFAULT_CHANNELS) -> list[int]:
    idx = [i for i, n in enumerate(names) if re.match(pattern, n)]
    if not idx:
        raise ValueError(f"NEGATIVITY_NO_CHANNELS: none of {len(names)} names match {pattern}")
    return idx


def negativity_stats(pred_z, idx, mean, std, w):
    """(neg_frac, min, neg_mean) per selected channel for one prediction.

    pred_z (1|B, C, H, W) z-space torch tensor; idx channel indices; mean/std
    float64 torch (C,); w float64 torch (H,) summing to 1. Returns three float64
    torch (len(idx),) tensors.
    """
    import torch

    x = pred_z.reshape(pred_z.shape[-3:])[idx].to(torch.float64)
    x = x * std[idx, None, None] + mean[idx, None, None]                # physical
    neg = (x < 0).to(torch.float64)
    frac = neg.mean(dim=-1) @ w
    mn = x.amin(dim=(-2, -1))
    negmean = torch.clamp(x, max=0.0).mean(dim=-1) @ w
    return frac, mn, negmean


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--start", required=True, type=int, nargs=2, metavar=("YEAR", "FRAME"))
    p.add_argument("--n-steps", required=True, type=int)
    p.add_argument("--member-id", required=True)
    p.add_argument("--chunk-len", type=int, default=40)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--years-dir", type=Path, required=True)
    p.add_argument("--channels", default=DEFAULT_CHANNELS, help="regex over channel names")
    p.add_argument("--dry-air-fix", choices=("config", "on", "off"), default="config")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    import torch
    from sfno_ensemble.scores import equiangular_weights
    from sfno_inference import climate_driver as cd
    from sfno_inference.checkpoint_loader import build_wrapper_from_checkpoint, load_eval_params
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader
    from eval_inference import _resolve_and_check_channel_names, _resolve_device

    device = _resolve_device(args.device)
    start = tuple(args.start)
    eval_params = load_eval_params(args.run_dir, K=1)
    if args.dry_air_fix != "config":
        eval_params.conserve_dry_air = args.dry_air_fix == "on"
    tdp = eval_params.train_data_path
    pack = Path(tdp[0] if isinstance(tdp, list) else tdp).parent
    years = cd.years_needed(start[0], start[1], args.n_steps)
    cd.build_year_dir(cd.locate_year_files(pack, years), args.years_dir)

    wrapper = build_wrapper_from_checkpoint(eval_params, args.ckpt, device=device)
    _, dataset, _ = _plasim_get_dataloader(eval_params, str(args.years_dir), device, mode="eval")
    dyears = cd.check_year_axis(dataset)
    names = _resolve_and_check_channel_names(args.run_dir, Path(dataset.files_paths[0]))
    n_out = int(eval_params.N_out_channels)
    idx = select_channels(names, args.channels)
    f64 = dict(dtype=torch.float64, device=device)
    mean = torch.as_tensor(cd.load_stats_f64(eval_params.global_means_path, n_out, "means"), **f64)
    std = torch.as_tensor(cd.load_stats_f64(eval_params.global_stds_path, n_out, "stds"), **f64)
    ic = int(dataset.file_offsets[dyears.index(start[0])]) + int(start[1])

    T, K = args.n_steps, len(idx)
    frac = np.full((T, K), np.nan)
    mn = np.full((T, K), np.nan)
    negm = np.full((T, K), np.nan)
    state = {"w": None, "trunc": -1, "last": 0}

    def on_step(k, pred):
        if not bool(torch.isfinite(pred).all()):
            state["trunc"] = k
            return False
        if state["w"] is None:
            state["w"] = torch.as_tensor(equiangular_weights(pred.shape[-2]), **f64)
        a, b, c = negativity_stats(pred, idx, mean, std, state["w"])
        frac[k - 1], mn[k - 1], negm[k - 1] = a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()
        state["last"] = k
        return True

    t0 = time.time()
    cd.stream_rollout(wrapper=wrapper, dataset=dataset, ic_global_idx=ic, n_steps=T,
                      chunk_len=args.chunk_len, eval_params=eval_params, device=device,
                      on_step=on_step)
    n = state["last"]
    vt = np.array([cd.valid_time(start[0], start[1], k) for k in range(1, n + 1)], dtype=np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, channels=np.array([names[i] for i in idx]), neg_frac=frac[:n], min=mn[:n],
             neg_mean=negm[:n], valid_year=vt[:, 0] if n else vt, valid_frame=vt[:, 1] if n else vt,
             start=np.array(start), truncated_at_step=state["trunc"], member_id=args.member_id,
             ckpt=str(args.ckpt), run_dir=str(args.run_dir),
             dry_air_fix=int(bool(eval_params.get("conserve_dry_air", False))))
    print(f"PERF s_per_step={(time.time() - t0) / max(n, 1):.3f} steps={n} channels={K}")
    if state["trunc"] > 0:
        print(f"NEGATIVITY_ROLLOUT_TRUNCATED step={state['trunc']} member={args.member_id} out={args.out}")
        return 3
    print(f"NEGATIVITY_ROLLOUT_OK member={args.member_id} steps={n} out={args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- one greppable line, then the traceback
        print(f"ERROR NEGATIVITY_ROLLOUT_FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
        raise
