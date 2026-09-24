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
                   help="Rollout length in NWP mode, in 6 h frames: K predictions at leads "
                        "6 h … 6K h, no lead-0 (default: 56 = 14 days). In lagged mode also "
                        "the age of the oldest member; must be a multiple of --ic-stride.")
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
    p.add_argument("--test-file-glob", type=str, default=None,
                   help="Glob for holdout h5 files (default: any *.h5). The PLaSim packs "
                        "are MOST.*.h5 and the E3SM ALLDATA packs are YYYY.h5; pass this "
                        "only to narrow a directory holding more than one pack.")
    p.add_argument("--limit-ics", type=int, default=None,
                   help="Process only the first N ICs per file (debugging)")
    # --- lagged ensemble (end-to-end plan stage 1 / task 11) ---
    # Additive: --ic-mode defaults to `monthly`, which is nwp_ic_offsets and the
    # behaviour every existing caller gets. The SLURM path passes neither flag.
    p.add_argument("--ic-mode", choices=["monthly", "lagged"], default="monthly",
                   help="IC placement. 'monthly' = nwp_ic_offsets, far-apart starts for a "
                        "scorecard (default). 'lagged' = starts spaced --ic-stride apart so "
                        "rollouts OVERLAP on common targets, which is what a lagged ensemble "
                        "needs and what the monthly spacing deliberately prevents.")
    p.add_argument("--ic-stride", type=int, default=4,
                   help="Lagged mode: spacing d between starts, in 6 h frames, and so the "
                        "spacing in age between members at one target. Must divide --nwp-K; "
                        "members per target is K/d (default 4 = 24 h => 14 at K=56)")
    p.add_argument("--n-targets", type=int, default=32,
                   help="Lagged mode: how many consecutive target frames (valid times) must "
                        "receive all K/d members. Sets the rollout count, "
                        "ceil((n_targets + K)/d) - 1 (default 32 => 21 at K=56, d=4). "
                        "⚠ COST: ~1.95 GB and ~77 s per rollout at the production shape")
    p.add_argument("--first-start", type=int, default=0,
                   help="Lagged mode: frame index of the first start within each file. "
                        "Translates the whole sweep without changing any count; its residue "
                        "mod d picks which targets carry the on-lattice depths d, 2d, … K.")
    return p.parse_args()


def _resolve_device(spec: str):
    import torch
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _list_test_files(test_holdout: Path, pattern: str | None = None) -> list[Path]:
    """List the holdout's h5 files, in the order the dataset concatenates them.

    The glob used to be the literal ``MOST.*.h5``, which is the PLaSim packer's
    filename convention. The E3SM ALLDATA packer names files by year
    (``2048.h5``, ``2049.h5``), so this raised ``no MOST.*.h5 found`` before a
    single rollout ran (job 7630665). Default to *any* ``.h5`` and let
    ``--test-file-glob`` narrow it when a directory holds more than one pack.

    Sorted by name, which is what makes ``dataset.file_offsets`` line up with
    this list — both PLaSim's ``MOST.NNNN.h5`` and E3SM's ``YYYY.h5`` sort
    chronologically as strings, and the two are never mixed in one directory.
    """
    if pattern:
        files = sorted(test_holdout.glob(pattern))
        if not files:
            raise SystemExit(f"no files matching {pattern!r} under {test_holdout}")
        return files

    files = sorted(test_holdout.glob("*.h5"))
    if not files:
        raise SystemExit(f"no .h5 files found under {test_holdout}")
    return files


def _read_n_samples(h5_path: Path) -> int:
    import h5py
    with h5py.File(h5_path, "r") as f:
        return int(f["time_plasim"].shape[0])


def _read_lat_lon_from_run(run_dir: Path, h5_path: Path) -> tuple[list[float], list[float]]:
    """Read lat/lon coords in DATA order (descending, North-first).

    The authoritative source is the source h5 (``<file>['lat']`` /
    ``['lon']``) — the exact grid the model trains and forecasts on, and
    the order the data rows are written in (row 0 = +87.86 deg N). When
    the training metadata is co-located, cross-check the rounded
    value-set against ``coords/lat`` and fail loud on a genuine grid
    mismatch (orientation-agnostic, so it only fires on a real
    wrong-run/wrong-data pairing).

    NOTE: a prior version read a non-existent *top-level* ``lat`` key
    from ``metadata/data.json`` (the lat array lives under ``coords/lat``)
    and silently fell back to ascending torch-harmonics nodes, which
    hemisphere-flipped the written ``lat`` coordinate relative to the
    descending data. See docs/2026-06-02_eval_inference_latitude_flip*.md.
    """
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        lat = np.asarray(f["lat"][:], dtype=np.float64)
        lon = np.asarray(f["lon"][:], dtype=np.float64)

    # Cross-check against training metadata coords/lat when co-located.
    cfg = json.loads((run_dir / "config.json").read_text())
    meta_path = None
    if cfg.get("metadata_json_path"):
        cand = Path(cfg["metadata_json_path"])
        if cand.is_file():
            meta_path = cand
    if meta_path is None:
        # Neighbour location: <train_data_path>/../metadata/data.json
        # (train_data_path is ``{OUTPUT_ROOT}/train`` => parent is OUTPUT_ROOT).
        train_path = cfg.get("train_data_path")
        if isinstance(train_path, list):
            train_path = train_path[0] if train_path else None
        if train_path:
            cand = Path(train_path).parent / "metadata" / "data.json"
            if cand.is_file():
                meta_path = cand

    if meta_path is not None:
        md = json.loads(meta_path.read_text())
        meta_lat = md.get("coords", {}).get("lat")
        if meta_lat is not None:
            h5_set = sorted(round(float(x), 2) for x in lat)
            md_set = sorted(round(float(x), 2) for x in meta_lat)
            if h5_set != md_set:
                raise ValueError(
                    f"lat grid mismatch: h5 {h5_path.name} vs metadata {meta_path} "
                    f"have different latitude value-sets. Refusing to write a "
                    f"mislabeled grid (wrong run/data pairing?)."
                )

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
    from sfno_ensemble import plan_lagged_sweep
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader

    device = _resolve_device(args.device)
    logger.info("device: %s", device)

    test_files = _list_test_files(args.test_holdout, args.test_file_glob)
    if args.limit_files:
        test_files = test_files[: args.limit_files]
    logger.info("test files: %s", [f.name for f in test_files])

    eval_params = load_eval_params(args.run_dir, K=args.nwp_K)
    wrapper = build_wrapper_from_checkpoint(eval_params, args.ckpt, device=device)
    out_bias, out_scale = _load_run_norm_stats(eval_params, device)

    # Channel names: derive from the first test h5's authoritative
    # channel_state ‖ channel_diagnostic; cross-check against training config.
    channel_names = _resolve_and_check_channel_names(args.run_dir, test_files[0])
    lat, lon = _read_lat_lon_from_run(args.run_dir, test_files[0])

    # A lagged sweep writes to its own subdirectory: its members overlap in time and
    # would otherwise be scored as if they were the monthly sweep's independent ICs.
    sweep = "lagged" if args.ic_mode == "lagged" else "nwp"
    out_dir = args.out_root / "inference" / sweep
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

        if args.ic_mode == "lagged":
            plan = plan_lagged_sweep(
                n, K=args.nwp_K, stride=args.ic_stride,
                n_targets=args.n_targets, first_start=args.first_start,
            )
            offsets = plan.starts
            logger.info("%s: %s", fpath.name, plan.summary())
        else:
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
            result.rollout_mode = sweep
            # Lagged members are named by their START index, not by IC ordinal: the
            # start is what identifies a member, and alignment regroups by start+depth.
            out_nc = out_dir / (f"{fpath.stem}_s{sample_idx:05d}.nc" if sweep == "lagged"
                                else f"{fpath.stem}_ic{ic_n:03d}.nc")
            write_rollout_nc(
                out_nc, result=result,
                channel_names=channel_names, lat=lat, lon=lon,
                ckpt_path=str(args.ckpt),
                eval_sha7=args.eval_sha7, data_sha7=args.data_sha7,
                train_sha7=args.train_sha7, run_tag=args.run_tag,
                rollout_mode=sweep,
            )
            elapsed = time.time() - t_start
            logger.info(
                "wrote %s  (file=%s sample_idx=%d, %.2fs)",
                out_nc.name, fpath.name, sample_idx, elapsed,
            )
            n_written += 1

    total = time.time() - t0
    logger.info("%s mode done: %d files, %d ICs written, %.1f min",
                sweep, len(test_files), n_written, total / 60.0)
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

    test_files = _list_test_files(args.test_holdout, args.test_file_glob)
    if args.limit_files:
        test_files = test_files[: args.limit_files]

    channel_names = _resolve_and_check_channel_names(args.run_dir, test_files[0])
    lat, lon = _read_lat_lon_from_run(args.run_dir, test_files[0])

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
