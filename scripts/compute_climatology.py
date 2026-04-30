#!/usr/bin/env python3
"""compute_climatology.py — build the time-of-year-proleptic climatology.

Implements docs/sfno_eval_plan.md §C.2. Walks 100 training files at
``{train_dir}/MOST.NNNN.h5``, accumulates a Welford mean + std per
``(month, day, hour_quarter)`` × channel × lat × lon bin, and writes
``baselines/climatology_proleptic.nc`` containing the three arrays
``mean``, ``std``, ``n_contributors``.

CPU-only; no GPU needed. Memory budget ≈ 12.4 GB at H=64, W=128, 53
channels — fits comfortably on a Stampede3 ``skx`` node (191 GB).

Usage::

    scripts/compute_climatology.py \\
        --train-dir $SCRATCH/AI-RES/data/makani/sim52_full/train \\
        --out $WORK2/AI-RES/results/sfno_eval/$RUN_TAG/baselines/climatology_proleptic.nc
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make src/ importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build time-of-year-proleptic climatology from training pool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--train-dir", required=True, type=Path,
                   help="Directory of training MOST.*.h5 files (e.g. .../sim52_full/train)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output NetCDF path")
    p.add_argument("--n-chan", type=int, default=53,
                   help="Number of channels in the combined state||diagnostic field (default: 53)")
    p.add_argument("--H", type=int, default=64, help="Latitude grid size (default: 64)")
    p.add_argument("--W", type=int, default=128, help="Longitude grid size (default: 128)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N files (debugging)")
    p.add_argument("--channel-names", type=Path, default=None,
                   help="Optional path to a JSON list of channel names "
                        "(written into the NetCDF for downstream use)")
    p.add_argument("--source-files-out", type=Path, default=None,
                   help="If set, write the resolved (post-symlink) realpath of every "
                        "ingested file to this JSON for provenance (Q11)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("compute_climatology")

    import json
    import numpy as np
    import xarray as xr
    from sfno_eval.climatology import build_climatology

    train_dir = args.train_dir
    if not train_dir.is_dir():
        raise SystemExit(f"--train-dir does not exist: {train_dir}")

    files = sorted(train_dir.glob("MOST.*.h5"))
    if not files:
        raise SystemExit(f"no MOST.*.h5 found under {train_dir}")
    if args.limit:
        files = files[: args.limit]
    logger.info("ingesting %d training files", len(files))

    # Q11: log resolved (post-symlink) paths for provenance.
    if args.source_files_out is not None:
        resolved = [os.path.realpath(str(p)) for p in files]
        args.source_files_out.parent.mkdir(parents=True, exist_ok=True)
        args.source_files_out.write_text(json.dumps(
            [{"link": str(p), "realpath": r} for p, r in zip(files, resolved)],
            indent=2,
        ))
        logger.info("wrote source-file provenance to %s", args.source_files_out)

    out_dict = build_climatology(files, n_chan=args.n_chan, H=args.H, W=args.W)

    # Channel names (optional).
    channel_names: list[str] | None = None
    if args.channel_names is not None and args.channel_names.is_file():
        channel_names = json.loads(args.channel_names.read_text())
        if len(channel_names) != args.n_chan:
            raise SystemExit(
                f"--channel-names has {len(channel_names)} entries, expected {args.n_chan}"
            )

    # Build xarray Dataset.
    ds = xr.Dataset(
        data_vars=dict(
            mean=(("doy", "hour_quarter", "channel", "lat", "lon"), out_dict["mean"]),
            std=(("doy", "hour_quarter", "channel", "lat", "lon"), out_dict["std"]),
            n_contributors=(("doy", "hour_quarter"), out_dict["n_contributors"]),
        ),
        coords=dict(
            doy=("doy", np.arange(366, dtype=np.int32)),
            hour_quarter=("hour_quarter", np.array([0, 6, 12, 18], dtype=np.int32)),
            channel=("channel", channel_names if channel_names else np.arange(args.n_chan)),
            lat=("lat", np.arange(args.H)),
            lon=("lon", np.arange(args.W)),
        ),
        attrs=dict(
            indexing="time-of-year-proleptic (month, day, hour_quarter)",
            n_files_ingested=len(files),
            n_chan=int(args.n_chan),
            H=int(args.H),
            W=int(args.W),
        ),
    )
    ds["doy"].attrs["description"] = "0-indexed day of year in a 366-day calendar (Feb 29 = doy 59)"
    ds["hour_quarter"].attrs["units"] = "hour of day (UTC), 6h granularity"
    ds["n_contributors"].attrs["description"] = "Number of training samples in each bin"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ("mean", "std")}
    ds.to_netcdf(args.out, encoding=encoding, format="NETCDF4")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
