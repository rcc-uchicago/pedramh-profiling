#!/usr/bin/env python3
"""eval_inference.py — batch SFNO emulator rollouts over the test split.

Implements docs/sfno_eval_plan.md §G (orchestration). Two modes:

  - ``--mode nwp`` (default): for each test-year h5, run K=56 rollouts
    from 12 monthly-spaced ICs (§A.4 ``nwp_ic_offsets``), 96 total.
  - ``--mode climate``: for each test-year h5, run a single
    K = n_samples - 1 rollout from sample 0. NWP non-leap files
    produce K=1454; leap files K=1458.

Outputs NetCDF per (file, IC) under
``{out_root}/inference/{mode}/{file_stem}_ic{nnn}.nc`` per §B.4.

Usage::

    scripts/eval_inference.py \\
        --run-dir $SCRATCH/SFNO_Climate_Emulator/runs/sfno_full/plasim_sim52_full/0 \\
        --ckpt   $SCRATCH/SFNO_Climate_Emulator/runs/sfno_full/plasim_sim52_full/0/training_checkpoints/best_ckpt_mp0.tar \\
        --test-holdout $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_full/test_holdout \\
        --out-root $WORK2/SFNO_Climate_Emulator/results/sfno_eval/$RUN_TAG \\
        --mode nwp \\
        --eval-sha7 abc1234 \\
        --data-sha7 58413cb \\
        --train-sha7 106d19d \\
        --run-tag 20260429_eval-abc1234_data-58413cb_train-106d19d_ckpt-best_ckpt_mp0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

# Make the in-repo src/ importable when invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

logger = logging.getLogger("eval_inference")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch SFNO emulator rollouts over the test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Training run directory (must contain config.json and global_means/stds.npy)")
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Path to best_ckpt_mp0.tar (or any legacy checkpoint)")
    p.add_argument("--test-holdout", required=True, type=Path,
                   help="Directory of test h5 files (output of build_test_split.py)")
    p.add_argument("--out-root", required=True, type=Path,
                   help="Output root; per-IC NetCDFs go under {out-root}/inference/{mode}/")
    p.add_argument("--mode", choices=["nwp", "climate"], default="nwp",
                   help="Rollout mode (default: nwp)")
    p.add_argument("--nwp-K", type=int, default=56,
                   help="Rollout horizon in NWP mode (default: 56 = 14 days)")
    p.add_argument("--nwp-n-ic", type=int, default=12,
                   help="ICs per file in NWP mode (default: 12)")
    p.add_argument("--eval-sha7", required=True, type=str)
    p.add_argument("--data-sha7", required=True, type=str)
    p.add_argument("--train-sha7", required=True, type=str)
    p.add_argument("--run-tag", required=True, type=str)
    p.add_argument("--device", type=str, default="auto",
                   help="Torch device ('auto' picks cuda:<current> if available, else cpu)")
    p.add_argument("--no-assert-contract", action="store_true",
                   help="Disable per-step 58→53 contract assertions (faster; default off)")
    p.add_argument("--limit-files", type=int, default=None,
                   help="Process only the first N files (debugging)")
    p.add_argument("--limit-ics", type=int, default=None,
                   help="Process only the first N ICs per file (debugging)")
    return p.parse_args()


def _resolve_device(spec: str):
    import torch
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _list_test_files(test_holdout: Path) -> list[Path]:
    files = sorted(test_holdout.glob("MOST.*.h5"))
    if not files:
        raise SystemExit(f"no MOST.*.h5 found under {test_holdout}")
    return files


def _read_n_samples(h5_path: Path) -> int:
    import h5py
    with h5py.File(h5_path, "r") as f:
        return int(f["time_plasim"].shape[0])


def _read_lat_lon_from_run(run_dir: Path) -> tuple[list[float], list[float]]:
    """Read lat/lon coords from the training run's metadata/data.json.

    The packager writes metadata/data.json with ``lat`` (Legendre-Gauss
    grid) and ``lon`` arrays; the training subset symlinks point at the
    same metadata. As a fallback for environments where this file is
    not co-located with the run dir, recompute from torch_harmonics.
    """
    cfg = json.loads((run_dir / "config.json").read_text())
    metadata_paths = []
    if "metadata_json_path" in cfg:
        metadata_paths.append(Path(cfg["metadata_json_path"]))
    # Try common neighbour location: $OUTPUT_ROOT/metadata/data.json sibling
    # of stats/. Use train_data_path as a starting point.
    train_path = cfg.get("train_data_path")
    if train_path:
        if isinstance(train_path, list):
            train_path = train_path[0]
        metadata_paths.append(Path(train_path).parent.parent / "metadata" / "data.json")

    for mp in metadata_paths:
        if mp.is_file():
            md = json.loads(mp.read_text())
            if "lat" in md and "lon" in md:
                return list(md["lat"]), list(md["lon"])

    # Fallback: compute from torch_harmonics.
    import torch_harmonics as th
    import numpy as np
    nlat = cfg["img_shape_x"]
    nlon = cfg["img_shape_y"]
    # Legendre-Gauss latitudes from -1..1 cosines.
    cos_thetas, _ = th.quadrature.legendre_gauss_weights(nlat, -1.0, 1.0)
    cos_thetas = cos_thetas.numpy() if hasattr(cos_thetas, "numpy") else cos_thetas
    lat = np.degrees(np.arcsin(cos_thetas))
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    return list(lat), list(lon)


def _channel_names_from_h5(h5_path: Path) -> list[str]:
    """Read ``channel_state ‖ channel_diagnostic`` from an h5 file.

    The h5 file written by the packager carries the authoritative
    per-position channel labels in these two attributes. Using them
    directly removes any chance of drift between (a) the training
    config's ``channel_names``, (b) the climatology's ``channel`` coord,
    and (c) the actual data layout — which is exactly what produced the
    v10.0/v10.1 contamination incident.
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        cs = [c.decode() if isinstance(c, bytes) else str(c)
              for c in f["channel_state"][:]]
        if "channel_diagnostic" in f:
            cd = [c.decode() if isinstance(c, bytes) else str(c)
                  for c in f["channel_diagnostic"][:]]
        else:
            cd = []
    return cs + cd


def _resolve_and_check_channel_names(
    run_dir: Path,
    h5_path: Path,
) -> list[str]:
    """Return h5-derived channel names; hard-fail if they disagree with config.

    The h5 attributes are authoritative for what's *in* the data tensor.
    The training config's ``channel_names`` records what the model was
    *trained against*. They must match, position by position. Any drift
    means the model and the data are talking past each other — score_nwp
    would silently produce nonsense (the v10.0/v10.1 incident).
    """
    h5_names = _channel_names_from_h5(h5_path)
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg_names = cfg.get("channel_names")
    if cfg_names is None:
        logger.warning(
            "run_dir/config.json has no channel_names; trusting h5 layout for %s.",
            h5_path.name,
        )
        return h5_names
    cfg_names = [str(c) for c in cfg_names]
    h5_names = [str(c) for c in h5_names]
    if cfg_names != h5_names:
        n = max(len(cfg_names), len(h5_names))
        diff_lines = []
        for i in range(n):
            cc = cfg_names[i] if i < len(cfg_names) else "<missing>"
            hh = h5_names[i] if i < len(h5_names) else "<missing>"
            mark = "" if cc == hh else "  <-- MISMATCH"
            diff_lines.append(f"  [{i:>2}] cfg={cc!r:<12} h5={hh!r:<12}{mark}")
        raise SystemExit(
            "channel-name mismatch between training config and test data h5.\n"
            f"  config       : {run_dir}/config.json\n"
            f"  h5 reference : {h5_path}\n"
            "Per-position diff:\n" + "\n".join(diff_lines) + "\n"
            "Refusing to run inference. The model was trained on a different "
            "channel layout than the data tensor it would be fed. Re-pack the "
            "data, or re-train, so that the two layouts agree."
        )
    return h5_names


# ---------------------------------------------------------------------------
# Per-mode runners
# ---------------------------------------------------------------------------

def run_nwp(args: argparse.Namespace) -> int:
    import torch
    from sfno_inference import (
        load_eval_params,
        build_wrapper_from_checkpoint,
        nwp_ic_offsets,
        rollout_one_ic,
        write_rollout_nc,
    )
    from sfno_inference.rollout_driver import _load_run_norm_stats
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader

    device = _resolve_device(args.device)
    logger.info("device: %s", device)

    test_files = _list_test_files(args.test_holdout)
    if args.limit_files:
        test_files = test_files[: args.limit_files]
    logger.info("test files: %s", [f.name for f in test_files])

    eval_params = load_eval_params(args.run_dir, K=args.nwp_K)
    wrapper = build_wrapper_from_checkpoint(eval_params, args.ckpt, device=device)
    out_bias, out_scale = _load_run_norm_stats(eval_params, device)

    # Channel names: derive from the first test h5's authoritative
    # channel_state ‖ channel_diagnostic; cross-check against training config.
    channel_names = _resolve_and_check_channel_names(args.run_dir, test_files[0])
    lat, lon = _read_lat_lon_from_run(args.run_dir)

    out_dir = args.out_root / "inference" / "nwp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # NWP mode: one dataset built per file (the dataset config has all
    # files listed but the IC index is into the global concat). We use
    # the test_holdout as the location and pick global indices.
    dataloader, dataset, _ = _plasim_get_dataloader(
        eval_params, str(args.test_holdout), device, mode="eval",
    )

    n_written = 0
    t0 = time.time()
    for fpath in test_files:
        n = _read_n_samples(fpath)
        # Map file → starting global index (file_offsets are sorted).
        file_idx = next(
            i for i, p in enumerate(dataset.files_paths) if Path(p).name == fpath.name
        )
        file_start_global = int(dataset.file_offsets[file_idx])

        offsets = nwp_ic_offsets(n, K=args.nwp_K, n_ic=args.nwp_n_ic)
        if args.limit_ics:
            offsets = offsets[: args.limit_ics]

        for ic_n, sample_idx in enumerate(offsets):
            global_idx = file_start_global + sample_idx
            t_start = time.time()
            result = rollout_one_ic(
                wrapper=wrapper,
                dataset=dataset,
                ic_global_idx=global_idx,
                eval_params=eval_params,
                device=device,
                out_bias=out_bias,
                out_scale=out_scale,
                assert_contract=not args.no_assert_contract,
            )
            result.rollout_mode = "nwp"
            out_nc = out_dir / f"{fpath.stem}_ic{ic_n:03d}.nc"
            write_rollout_nc(
                out_nc, result=result,
                channel_names=channel_names, lat=lat, lon=lon,
                ckpt_path=str(args.ckpt),
                eval_sha7=args.eval_sha7, data_sha7=args.data_sha7,
                train_sha7=args.train_sha7, run_tag=args.run_tag,
                rollout_mode="nwp",
            )
            elapsed = time.time() - t_start
            logger.info(
                "wrote %s  (file=%s sample_idx=%d, %.2fs)",
                out_nc.name, fpath.name, sample_idx, elapsed,
            )
            n_written += 1

    total = time.time() - t0
    logger.info("nwp mode done: %d files, %d ICs written, %.1f min",
                len(test_files), n_written, total / 60.0)
    return 0


def run_climate(args: argparse.Namespace) -> int:
    import torch
    from sfno_inference import (
        load_eval_params,
        build_wrapper_from_checkpoint,
        rollout_one_ic,
        write_rollout_nc,
    )
    from sfno_inference.rollout_driver import _load_run_norm_stats
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader

    device = _resolve_device(args.device)

    test_files = _list_test_files(args.test_holdout)
    if args.limit_files:
        test_files = test_files[: args.limit_files]

    channel_names = _resolve_and_check_channel_names(args.run_dir, test_files[0])
    lat, lon = _read_lat_lon_from_run(args.run_dir)

    out_dir = args.out_root / "inference" / "climate"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    t0 = time.time()
    for fpath in test_files:
        n = _read_n_samples(fpath)
        K = n - 1  # within-file maximum horizon (s + K < n_samples)
        logger.info("climate rollout: %s  K=%d  (n_samples=%d)", fpath.name, K, n)

        eval_params = load_eval_params(args.run_dir, K=K)
        wrapper = build_wrapper_from_checkpoint(eval_params, args.ckpt, device=device)
        out_bias, out_scale = _load_run_norm_stats(eval_params, device)

        dataloader, dataset, _ = _plasim_get_dataloader(
            eval_params, str(args.test_holdout), device, mode="eval",
        )
        file_idx = next(
            i for i, p in enumerate(dataset.files_paths) if Path(p).name == fpath.name
        )
        file_start_global = int(dataset.file_offsets[file_idx])
        global_idx = file_start_global + 0  # sample 0

        t_start = time.time()
        result = rollout_one_ic(
            wrapper=wrapper, dataset=dataset, ic_global_idx=global_idx,
            eval_params=eval_params, device=device,
            out_bias=out_bias, out_scale=out_scale,
            assert_contract=not args.no_assert_contract,
        )
        result.rollout_mode = "climate"
        out_nc = out_dir / f"{fpath.stem}_full.nc"
        write_rollout_nc(
            out_nc, result=result,
            channel_names=channel_names, lat=lat, lon=lon,
            ckpt_path=str(args.ckpt),
            eval_sha7=args.eval_sha7, data_sha7=args.data_sha7,
            train_sha7=args.train_sha7, run_tag=args.run_tag,
            rollout_mode="climate",
        )
        elapsed = time.time() - t_start
        logger.info("wrote %s  (K=%d, %.1f min)", out_nc.name, K, elapsed / 60.0)
        n_written += 1

        # Free GPU memory between files (each climate K varies, so we
        # rebuild the wrapper per file anyway).
        del wrapper, dataset, dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = time.time() - t0
    logger.info("climate mode done: %d ICs written, %.1f min", n_written, total / 60.0)
    return 0


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.mode == "nwp":
        return run_nwp(args)
    return run_climate(args)


if __name__ == "__main__":
    sys.exit(main())
