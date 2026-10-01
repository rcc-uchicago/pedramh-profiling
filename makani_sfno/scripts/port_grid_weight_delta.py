#!/usr/bin/env python3
"""port_grid_weight_delta.py — measurement only (api_delta §3.2, operator ruling 2026-10-01).

How much does makani's ``equiangular`` ("naive") quadrature, which our configs declare, differ
from the true area weights of our cell-centred 1° grid (rows 89.5 … −89.5), and what does that
do to A e243's l2? Changes nothing; prints numbers for jesswan.

(a) weights: makani's own ``GridQuadrature("naive", (180, 360), normalize=True)`` vs band areas
    ∝ sin(north edge) − sin(south edge), both normalised to sum 1.
(b) loss: A e243, one step, on the first ``--n`` samples of the valid split (the golden set's 64),
    z-space squared error integrated with each weighting, mean over samples; per channel and the
    channel mean (= constant channel weights). This is a single-step z-space l2, not the trainer's
    logged validation loss.

PASS token: ``GRID_WEIGHT_DELTA max_rel_w=… loss_naive=… loss_cell=… loss_rel=…``.
Launch under ``python -m torch.distributed.run --standalone --nproc_per_node=1``.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--n", type=int, default=64)
    args = p.parse_args()

    import torch
    from makani.utils.grids import GridQuadrature
    from sfno_inference.checkpoint_loader import build_wrapper_from_checkpoint, load_eval_params
    from sfno_inference.rollout_driver import _load_run_norm_stats, rollout_one_ic
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader
    from eval_inference import _resolve_device

    # build first: GridQuadrature asks makani's comm, which build_wrapper_from_checkpoint initialises
    device = _resolve_device("auto")
    ep = load_eval_params(args.run_dir, K=1)
    wrapper = build_wrapper_from_checkpoint(ep, args.ckpt, device=device)

    nlat, nlon = 180, 360
    w_naive = GridQuadrature("naive", (nlat, nlon), normalize=True).quad_weight.reshape(nlat, nlon).double()
    lat = torch.tensor([89.5 - i for i in range(nlat)], dtype=torch.float64)
    band = torch.sin(torch.deg2rad(lat + 0.5)) - torch.sin(torch.deg2rad(lat - 0.5))
    w_cell = (band / band.sum() / nlon).reshape(nlat, 1).expand(nlat, nlon)
    row_n, row_c = w_naive.sum(1), w_cell.sum(1)
    rel_row = (row_n - row_c).abs() / row_c
    i_max = int(torch.argmax(rel_row))
    print(f"GRID_WEIGHTS sum_naive={float(w_naive.sum()):.6f} sum_cell={float(w_cell.sum()):.6f} "
          f"max_abs_row={float((row_n - row_c).abs().max()):.3e} max_rel_row={float(rel_row.max()):.3f} "
          f"at lat={float(lat[i_max]):+.1f} rows_rel_gt_1pct={int((rel_row > 0.01).sum())} "
          f"rows_rel_gt_10pct={int((rel_row > 0.10).sum())} L1={float((row_n - row_c).abs().sum()):.4e}")

    bias, scale = _load_run_norm_stats(ep, device)
    _, ds, _ = _plasim_get_dataloader(ep, str(ep.valid_data_path), device, mode="eval")
    names = list(ep.channel_names)
    wn, wc = w_naive.to(device), w_cell.to(device)

    s_n = torch.zeros(len(names), dtype=torch.float64, device=device)
    s_c = torch.zeros_like(s_n)
    pole_n = torch.zeros_like(s_n)
    pole_c = torch.zeros_like(s_n)
    polar = (lat.abs() >= 85.0).to(device).reshape(nlat, 1).double()
    for i in range(args.n):
        r = rollout_one_ic(wrapper=wrapper, dataset=ds, ic_global_idx=i, eval_params=ep, device=device,
                           out_bias=bias, out_scale=scale)
        zp = ((r.prediction.to(device) - bias) / scale)[0].double()
        zt = ((r.truth.to(device) - bias) / scale)[0].double()
        e2 = (zp - zt) ** 2
        s_n += (e2 * wn).sum((-2, -1))
        s_c += (e2 * wc).sum((-2, -1))
        pole_n += (e2 * wn * polar).sum((-2, -1))
        pole_c += (e2 * wc * polar).sum((-2, -1))
    l_n, l_c = s_n / args.n, s_c / args.n
    rel_c = (l_c - l_n) / l_n
    L_n, L_c = float(l_n.mean()), float(l_c.mean())
    worst = sorted(range(len(names)), key=lambda k: -abs(float(rel_c[k])))[:3]
    worst_s = ",".join(f"{names[k]}:{float(rel_c[k]):+.4f}" for k in worst)
    print(f"GRID_WEIGHT_LOSS n={args.n} channels={len(names)} polar_share_naive={float(pole_n.sum() / s_n.sum()):.4e} "
          f"polar_share_cell={float(pole_c.sum() / s_c.sum()):.4e}")
    for k in worst:
        print(f"GRID_WEIGHT_CHANNEL {names[k]} loss_naive={float(l_n[k]):.6e} loss_cell={float(l_c[k]):.6e} "
              f"rel={float(rel_c[k]):+.4e}")
    print(f"GRID_WEIGHT_DELTA max_rel_w={float(rel_row.max()):.3f} loss_naive={L_n:.6e} loss_cell={L_c:.6e} "
          f"loss_rel={(L_c - L_n) / L_n:+.4e} worst3={worst_s} ckpt={args.ckpt.parent.parent.name}/{args.ckpt.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
