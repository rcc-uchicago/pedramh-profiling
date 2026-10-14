#!/usr/bin/env python3
"""climate_driver_equiv.py — gate G2: streaming driver vs ``rollout_one_ic``.

Pre-registered (docs/2026-09-24_climate_driver_g2_prereg.md, committed before
the job ran): checkpoint A, test year 2048, start frame 1092, K=56.

  PASS iff  (1) for chunk_len 40, **every** lead's physical-unit prediction is
                bitwise equal to rollout_one_ic's, and
            (2) chunk_len 1, 7 and 40 are bitwise equal to each other.

TOLERANCE is "bitwise" and is not a parameter. A control (rollout_one_ic run
twice) separates GPU nondeterminism from a driver bug; a non-bitwise result is
reported with its max abs/rel error and where, never passed.

Launch under ``python -m torch.distributed.run --standalone --nproc_per_node=1``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TOLERANCE = "bitwise"   # pre-registered; see module docstring. Never loosen (CLAUDE.md #1).


def _diff(a, b, names):
    """Max abs/rel error and where, over (K, C, H, W)."""
    import torch
    d = (a.double() - b.double()).abs()
    k, c, i, j = [int(x) for x in torch.unravel_index(torch.argmax(d), d.shape)]
    rel = d / b.double().abs().clamp_min(1e-30)
    first = next(s for s in range(a.shape[0]) if not torch.equal(a[s], b[s])) + 1
    return (f"first_bad_lead={first} max_abs={float(d.max()):.3e} at lead={k + 1} "
            f"channel={names[c]} lat_idx={i} lon_idx={j}; max_rel={float(rel.max()):.3e}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--holdout", required=True, type=Path, help="dir holding <year>.h5")
    p.add_argument("--year", type=int, default=2048)
    p.add_argument("--start-frame", type=int, default=1092)
    p.add_argument("--K", type=int, default=56)
    p.add_argument("--chunks", type=int, nargs="+", default=[40, 7, 1])
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import torch
    from sfno_inference import climate_driver as cd
    from sfno_inference.checkpoint_loader import build_wrapper_from_checkpoint, load_eval_params
    from sfno_inference.rollout_driver import _load_run_norm_stats, rollout_one_ic
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader
    from eval_inference import _resolve_and_check_channel_names, _resolve_device

    device = _resolve_device("auto")
    ep_block = load_eval_params(args.run_dir, K=args.K)
    ep_stream = load_eval_params(args.run_dir, K=1)          # valid_autoreg_steps = 0

    wrapper = build_wrapper_from_checkpoint(ep_block, args.ckpt, device=device)
    out_bias, out_scale = _load_run_norm_stats(ep_block, device)
    _, ds_block, _ = _plasim_get_dataloader(ep_block, str(args.holdout), device, mode="eval")
    _, ds_stream, _ = _plasim_get_dataloader(ep_stream, str(args.holdout), device, mode="eval")
    fidx = next(i for i, f in enumerate(ds_block.files_paths) if Path(f).stem == str(args.year))
    ic = int(ds_block.file_offsets[fidx]) + args.start_frame
    names = _resolve_and_check_channel_names(args.run_dir, Path(ds_block.files_paths[fidx]))

    def block():
        return rollout_one_ic(wrapper=wrapper, dataset=ds_block, ic_global_idx=ic,
                              eval_params=ep_block, device=device,
                              out_bias=out_bias, out_scale=out_scale).prediction

    t0 = time.time()
    ref = block()
    t_block = time.time() - t0
    control = block()
    deterministic = torch.equal(ref, control)
    print(f"EQUIV setup: year={args.year} frame={args.start_frame} ic={ic} K={args.K} "
          f"ckpt={args.ckpt.name} tolerance={TOLERANCE} block_s={t_block:.1f}")
    print(f"EQUIV control rollout_one_ic x2 bitwise={deterministic}"
          + ("" if deterministic else " (" + _diff(control, ref, names) + ")"))
    del control

    runs, dtypes = {}, {}
    for chunk in args.chunks:
        preds = []

        def on_step(k, pred):
            preds.append((pred * out_scale + out_bias).cpu())
            return True

        t0 = time.time()
        info = cd.stream_rollout(wrapper=wrapper, dataset=ds_stream, ic_global_idx=ic,
                                 n_steps=args.K, chunk_len=chunk, eval_params=ep_stream,
                                 device=device, on_step=on_step)
        runs[chunk] = torch.cat(preds, 0)
        dtypes[chunk] = info.feedback_dtype
        same = torch.equal(runs[chunk], ref)
        print(f"EQUIV chunk_len={chunk}: steps={info.n_steps_run} s={time.time() - t0:.1f} "
              f"feedback_dtype={info.feedback_dtype} bitwise_vs_rollout_one_ic={same}"
              + ("" if same else " (" + _diff(runs[chunk], ref, names) + ")"))

    base = runs[args.chunks[0]]
    chunk_inv = all(torch.equal(base, runs[c]) for c in args.chunks[1:])
    vs_ref = all(torch.equal(runs[c], ref) for c in args.chunks)
    print(f"EQUIV chunk invariance {args.chunks} bitwise={chunk_inv}")
    if vs_ref and chunk_inv:
        print(f"CLIMATE_DRIVER_EQUIV_OK K={args.K} leads_checked={args.K} "
              f"chunks={args.chunks} tolerance={TOLERANCE}")
        return 0
    print(f"ERROR CLIMATE_DRIVER_EQUIV_MISMATCH vs_rollout_one_ic={vs_ref} "
          f"chunk_invariant={chunk_inv} control_deterministic={deterministic}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
