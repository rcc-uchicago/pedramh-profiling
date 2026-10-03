#!/usr/bin/env python3
"""build_true_climatology.py — the real 5-year (2045-2049) time-mean, from data.

For jesswan: scoring a 5-year free-running rollout against truth state-by-state
is meaningless past the ~2-week predictability horizon (any good model's RMSE
saturates to climatological noise and ACC decays to ~0 by then, regardless of
quality). What IS meaningful is climate fidelity: does the rollout's long-run
mean state match the real climate's? This builds the "real climate's" half of
that comparison -- the true time-mean field over the same window the 5-year
protocol scores (2045-01-01 through 2049-12-31), reusing the exact same
dataset/channel/normalization path the model's own eval pipeline uses (never
re-parses raw h5 layout by hand -- see climate_driver.py's own rationale).

channel_names / N_out_channels / normalization all come from --run-dir's
config.json (channel_names via the same h5-cross-checked resolver
climate_rollout.py uses for a member's own channel list), so the result lines
up channel-for-channel with that run's own member_*.nc:time_mean field.

Output: one .npz with `time_mean` (N_out, H, W; physical units), `channel_names`
(row-for-row with `time_mean`), `global_means`/`global_stds` (the exact
per-channel values used to de-normalise -- the scorer reads these instead of a
separately-specified stats file, so there's no way for the two to drift out of
alignment), `lat`/`lon` (for cross-checking a member's own grid), `years`,
`n_samples` (provenance).

PASS token: TRUE_CLIMATOLOGY_OK n_samples=<n> out=<path>

Usage (run on a compute node -- this imports torch/makani, never the login node):
    python build_true_climatology.py --run-dir <run, for its config.json> \\
        --years 2045 2046 2047 2048 2049 --out <out.npz>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Any run whose config.json gives the channel set to score "
                        "(e.g. nf4_prod_b16_r1 for the full 101-channel set)")
    p.add_argument("--pack", type=Path, default=None,
                   help="Pack root (default: the run's own train_data_path parent)")
    p.add_argument("--years", required=True, type=int, nargs="+",
                   help="Years to average over, e.g. 2045 2046 2047 2048 2049")
    p.add_argument("--years-dir", type=Path, default=None,
                   help="Directory of YYYY.h5 symlinks (default: <out dir>/years_truth)")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args(argv)


def compute_time_mean_z(dataset, n_out: int, *, n_total: int | None = None,
                         progress_every: int = 0):
    """Streaming z-space time-mean of ``dataset``'s target state, ``(n_out, H, W)``.

    ``n_total`` defaults to ``dataset.n_samples_total - 1``: index ``idx``'s
    ``tar`` is the frame AFTER ``idx`` (the same one-step convention
    ``stream_rollout`` uses, reading forcing from ``done + 1``), so querying
    ``idx = 0..n_total-1`` reads target frames ``1..n_total`` -- every frame
    EXCEPT THE FIRST one (not the last, despite how that reads at a glance:
    frame 0 is never a `tar`). One 6 h frame of several thousand, noise for a
    multi-year mean.

    Assumes ``n_history=0``, ``dt=1`` and no eval-time subsampling -- the same
    assumptions ``K=1`` in ``load_eval_params`` is meant to establish, but
    this function does not itself re-verify them.
    """
    import torch

    if n_total is None:
        n_total = int(dataset.n_samples_total) - 1
    if n_total < 1:
        raise ValueError(f"NO_FRAMES: dataset has {dataset.n_samples_total} samples, "
                         f"need at least 2")
    time_sum = None
    for idx in range(n_total):
        _, tar, _, _ = dataset[idx]
        t = tar.reshape(tar.shape[-3:]).to(torch.float64)  # (n_out, H, W), z-space
        if time_sum is None:
            if t.shape[0] != n_out:
                raise ValueError(f"CHANNEL_COUNT_MISMATCH: tar has {t.shape[0]} channels, "
                                 f"expected n_out={n_out}")
            time_sum = torch.zeros((n_out, *t.shape[-2:]), dtype=torch.float64)
        time_sum += t
        if progress_every and idx % progress_every == 0:
            print(f"  {idx}/{n_total}", flush=True)
    return (time_sum / n_total), n_total


def main(argv=None) -> int:
    args = _parse_args(argv)
    import numpy as np
    from sfno_inference import climate_driver as cd
    from sfno_inference.checkpoint_loader import load_eval_params
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader
    from eval_inference import _read_lat_lon_from_run, _resolve_and_check_channel_names

    eval_params = load_eval_params(args.run_dir, K=1)  # n_future=0: one frame per index
    pack = args.pack
    if pack is None:
        tdp = eval_params.train_data_path
        pack = Path(tdp[0] if isinstance(tdp, list) else tdp).parent

    year_files = cd.locate_year_files(pack, args.years)
    years_dir = args.years_dir or args.out.parent / f"years_{args.years[0]}_{args.years[-1]}_truth"
    cd.build_year_dir(year_files, years_dir)
    lat, lon = _read_lat_lon_from_run(args.run_dir, year_files[args.years[0]])

    device = "cpu"
    _, dataset, _ = _plasim_get_dataloader(eval_params, str(years_dir), device, mode="eval")
    cd.check_year_axis(dataset)

    n_out = int(eval_params.N_out_channels)
    out_idx = getattr(eval_params, "out_channels", None)
    mean = cd.load_stats_f64(eval_params.global_means_path, n_out, "global_means", out_idx)
    std = cd.load_stats_f64(eval_params.global_stds_path, n_out, "global_stds", out_idx)

    # config.json's channel_names is ALREADY the run's own N_out-sized list for a
    # subset run (eval_inference.py's resolver returns cfg_names as-is in that
    # case) -- reducing it again with out_idx here would double-index an
    # already-short list. Use the same resolver climate_rollout.py uses to build
    # a member's own channel_names, so the two can never independently drift
    # (review finding, 2026-10-03: an earlier version of this fix assumed
    # channel_names was the pack's full list, which is wrong for a subset run).
    names = _resolve_and_check_channel_names(args.run_dir, year_files[args.years[0]])
    if len(names) != n_out:
        raise ValueError(f"CHANNEL_NAME_COUNT_MISMATCH: {len(names)} names for "
                         f"N_out_channels={n_out}")

    time_mean_z, n_total = compute_time_mean_z(dataset, n_out, progress_every=500)
    time_mean_phys = time_mean_z.numpy() * std[:, None, None] + mean[:, None, None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, time_mean=time_mean_phys.astype(np.float64),
             channel_names=np.array(names, dtype=object),
             global_means=mean, global_stds=std,
             lat=np.asarray(lat, dtype=np.float64), lon=np.asarray(lon, dtype=np.float64),
             years=np.array(args.years), n_samples=n_total)
    print(f"TRUE_CLIMATOLOGY_OK n_samples={n_total} years={args.years} out={args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- one greppable line, then the traceback
        print(f"ERROR TRUE_CLIMATOLOGY_FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
        raise
