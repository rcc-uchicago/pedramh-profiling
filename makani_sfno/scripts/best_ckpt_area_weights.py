#!/usr/bin/env python3
"""best_ckpt_area_weights.py — measurement only. Does a run's best checkpoint stay best under band-area weights?

makani chose ``best_ckpt_mp0.tar`` by its validation loss, which integrates with the "naive" equiangular quadrature
(``sin(linspace(0, π, 180))``: zero weight on our ±89.5° rows, polar caps under-weighted). This scores several epoch
checkpoints of one run on the SAME validation start dates, with a K-step rollout that mirrors ``validate_one_epoch``
(``rollout_one_ic``, K = valid_autoreg_steps + 1), and integrates the z-space squared error with both makani's naive
weights and the true band areas of our cell-centred 1° grid (∝ sin(north edge) − sin(south edge)), each summing to 1.
Per IC: loss = mean over leads and channels of the area-weighted squared error. Changes nothing.

Writes one JSON per epoch with the per-IC losses, so the aggregator can compare epochs on their common ICs (paired).
ICs are visited in a fixed shuffled order, so a time-budget cut still samples the whole validation year.

Token per epoch: ``BEST_CKPT_EPOCH epoch=… n=… loss_naive=… loss_band=…``.
Launch under ``python -m torch.distributed.run --standalone --nproc_per_node=1`` (one process per GPU).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--epochs", required=True, type=int, nargs="+", help="1-based epochs; ckpt_mp0_v{epoch-1}.tar")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--k", type=int, default=4, help="rollout steps; the run's valid_autoreg_steps + 1")
    p.add_argument("--budget-s", type=float, default=2400.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    from makani.utils.grids import GridQuadrature
    from sfno_inference.checkpoint_loader import build_wrapper_from_checkpoint, load_eval_params
    from sfno_inference.rollout_driver import _load_run_norm_stats, rollout_one_ic
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader
    from eval_inference import _resolve_device

    device = _resolve_device("auto")
    ep = load_eval_params(args.run_dir, K=args.k)
    ckpts = {e: args.run_dir / "training_checkpoints" / f"ckpt_mp0_v{e - 1}.tar" for e in args.epochs}
    for e, c in ckpts.items():
        if not c.is_file():
            print(f"ERROR CKPT_MISSING epoch={e} path={c}", flush=True)
            return 2
    # build first: GridQuadrature asks makani's comm, which build_wrapper_from_checkpoint initialises
    wrappers = {e: build_wrapper_from_checkpoint(ep, c, device=device) for e, c in ckpts.items()}

    nlat, nlon = 180, 360
    w_naive = GridQuadrature("naive", (nlat, nlon), normalize=True).quad_weight.reshape(nlat, nlon).double()
    w_naive = w_naive / w_naive.sum()
    lat = torch.tensor([89.5 - i for i in range(nlat)], dtype=torch.float64)
    band = torch.sin(torch.deg2rad(lat + 0.5)) - torch.sin(torch.deg2rad(lat - 0.5))
    w_band = (band / band.sum() / nlon).reshape(nlat, 1).expand(nlat, nlon)
    print(f"AREA_WEIGHTS naive_sum={float(w_naive.sum()):.6f} band_sum={float(w_band.sum()):.6f} "
          f"naive_rows_zero={int((w_naive.sum(1) == 0).sum())} band_rows_zero={int((w_band.sum(1) == 0).sum())}",
          flush=True)
    wn, wb = w_naive.to(device), w_band.to(device)

    bias, scale = _load_run_norm_stats(ep, device)
    _, ds, _ = _plasim_get_dataloader(ep, str(ep.valid_data_path), device, mode="eval")
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)

    rec = {e: {"epoch": e, "ckpt": str(c), "k": args.k, "ics": [], "naive": [], "band": []} for e, c in ckpts.items()}
    t0 = time.time()
    with torch.no_grad():
        for ic in order:
            if time.time() - t0 > args.budget_s:
                break
            for e, wrapper in wrappers.items():
                r = rollout_one_ic(wrapper=wrapper, dataset=ds, ic_global_idx=ic, eval_params=ep, device=device,
                                   out_bias=bias, out_scale=scale, assert_contract=False)
                zp = ((r.prediction.to(device) - bias) / scale).double()  # (K, C, H, W)
                zt = ((r.truth.to(device) - bias) / scale).double()
                e2 = (zp - zt) ** 2
                rec[e]["ics"].append(ic)
                rec[e]["naive"].append(float((e2 * wn).sum((-2, -1)).mean()))
                rec[e]["band"].append(float((e2 * wb).sum((-2, -1)).mean()))

    args.out.mkdir(parents=True, exist_ok=True)
    for e, d in rec.items():
        (args.out / f"epoch_{e:04d}.json").write_text(json.dumps(d))
        n = len(d["ics"])
        ln = sum(d["naive"]) / max(n, 1)
        lb = sum(d["band"]) / max(n, 1)
        print(f"BEST_CKPT_EPOCH epoch={e} n={n} loss_naive={ln:.7e} loss_band={lb:.7e} "
              f"elapsed_s={time.time() - t0:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
