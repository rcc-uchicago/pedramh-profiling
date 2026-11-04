"""E3SM truth for the five-year figures, one year per call.

Writes, for every target frame of the year:
  * the global mean of every channel, weighted exactly as the rollouts weight theirs
    (sfno_ensemble.scores.equiangular_weights, via the zonal mean, climate_driver.py:523-525);
  * the zonal mean of the plotted variables;
and the year's time mean of every channel.

It goes through build_true_climatology.py's own dataset / channel / normalisation path. Dataset index i yields
target frame i+1, so the frames are 1..1459 and frame 0 (Jan 1 00:00) is not among them -- the convention of every
truth file here. The time mean is accumulated in the same order and precision as compute_time_mean_z, so for
2045-2047 it must equal window_truth/truth_<year>.npz exactly: that is the gate (relative 1e-12 per channel).

Usage (compute node, SFNO venv): truth_series.py --run-dir <G4D run dir> --year 2045 --out truth_series_2045.npz
PASS = TRUTH_SERIES_OK
"""
import argparse
import sys
from pathlib import Path

import numpy as np

EVAL = Path("/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/"
            "eval-protocol/makani_sfno")
for sub in ("src", "polaris", "scripts"):
    sys.path.insert(0, str(EVAL / sub))
WINDOW_TRUTH = Path("/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/window_truth")
ZONAL_VARS = ("TREFHT", "PRECT", "U10", "U_l17", "Z3_l12", "PS", "TMQ")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--years-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    import torch
    from sfno_inference import climate_driver as cd
    from sfno_inference.checkpoint_loader import load_eval_params
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader
    from sfno_ensemble.scores import equiangular_weights
    from eval_inference import _read_lat_lon_from_run, _resolve_and_check_channel_names

    eval_params = load_eval_params(a.run_dir, K=1)   # n_future=0: one frame per index, as the truth builder
    tdp = eval_params.train_data_path
    pack = Path(tdp[0] if isinstance(tdp, list) else tdp).parent
    year_files = cd.locate_year_files(pack, [a.year])
    cd.build_year_dir(year_files, a.years_dir)
    lat, lon = _read_lat_lon_from_run(a.run_dir, year_files[a.year])
    _, dataset, _ = _plasim_get_dataloader(eval_params, str(a.years_dir), "cpu", mode="eval")
    cd.check_year_axis(dataset)

    n_out = int(eval_params.N_out_channels)
    out_idx = getattr(eval_params, "out_channels", None)
    mean = cd.load_stats_f64(eval_params.global_means_path, n_out, "global_means", out_idx)
    std = cd.load_stats_f64(eval_params.global_stds_path, n_out, "global_stds", out_idx)
    names = _resolve_and_check_channel_names(a.run_dir, year_files[a.year])
    if len(names) != n_out:
        raise ValueError("CHANNEL_NAME_COUNT_MISMATCH: %d names for N_out_channels=%d" % (len(names), n_out))
    zi = [names.index(v) for v in ZONAL_VARS]
    H = len(lat)
    w = torch.as_tensor(equiangular_weights(H), dtype=torch.float64)

    n_total = int(dataset.n_samples_total) - 1
    gm_z = np.empty((n_total, n_out), dtype=np.float64)
    zm = np.empty((n_total, len(zi), H), dtype=np.float32)
    time_sum = None
    for idx in range(n_total):
        _, tar, _, _ = dataset[idx]
        t = tar.reshape(tar.shape[-3:]).to(torch.float64)            # (n_out, H, W), z-space
        if time_sum is None:
            if t.shape[0] != n_out:
                raise ValueError("CHANNEL_COUNT_MISMATCH: tar has %d channels, expected %d" % (t.shape[0], n_out))
            time_sum = torch.zeros((n_out,) + tuple(t.shape[-2:]), dtype=torch.float64)
        time_sum += t
        zonal = t.mean(dim=-1)                                          # (C, H)
        gm_z[idx] = (zonal @ w).numpy()
        zm[idx] = (zonal[zi].numpy() * std[zi, None] + mean[zi, None]).astype(np.float32)
        if idx % 500 == 0:
            print("  %d/%d" % (idx, n_total), flush=True)
    time_mean = (time_sum / n_total).numpy() * std[:, None, None] + mean[:, None, None]
    gm = gm_z * std[None, :] + mean[None, :]
    frames = np.arange(1, n_total + 1)
    months = np.array([cd.month_of_frame(int(f)) for f in frames])

    gate = "none (no window_truth/truth_%d.npz)" % a.year
    ref_path = WINDOW_TRUTH / ("truth_%d.npz" % a.year)
    rc = 0
    if ref_path.exists():
        ref = np.load(ref_path, allow_pickle=True)
        rnames = [str(c) for c in ref["channel_names"]]
        shared = [n for n in names if n in rnames]
        mine = time_mean[[names.index(n) for n in shared]]
        theirs = np.asarray(ref["time_mean"], dtype=np.float64)[[rnames.index(n) for n in shared]]
        scale = np.abs(theirs).max(axis=(1, 2))
        rel = (np.abs(mine - theirs).max(axis=(1, 2)) / np.where(scale > 0, scale, 1.0)).max()
        gate = "time mean vs %s: %d shared channels, max rel diff %.3e (n %d vs %d)" % (
            ref_path.name, len(shared), rel, n_total, int(ref["n_samples"]))
        if not (rel <= 1e-12 and int(ref["n_samples"]) == n_total and len(shared) >= len(ZONAL_VARS)):
            print("ERROR TRUTH_SERIES_GATE year %d: %s" % (a.year, gate))
            rc = 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, year=a.year, frames=frames, months=months, gm=gm, zm=zm, zonal_vars=np.array(ZONAL_VARS),
             time_mean=time_mean, channel_names=np.array(names, dtype="U16"), lat=np.asarray(lat, np.float64),
             lon=np.asarray(lon, np.float64), n_total=n_total, weights=w.numpy())
    print("gate: %s" % gate)
    if rc == 0:
        print("TRUTH_SERIES_OK year=%d frames=%d channels=%d out=%s" % (a.year, n_total, n_out, a.out))
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- one greppable line, then the traceback
        print("ERROR TRUTH_SERIES_FAILED %s: %s" % (type(exc).__name__, str(exc).splitlines()[0][:300]))
        raise
