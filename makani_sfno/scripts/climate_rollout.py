#!/usr/bin/env python3
"""climate_rollout.py — one streaming multi-year member per process.

Sibling of ``eval_inference.py`` (whose ``run_climate`` is the single-file path
that OOMs; left alone). Drives :func:`sfno_inference.climate_driver.run_member`.

Launch under ``python -m torch.distributed.run --standalone --nproc_per_node=1``
on Polaris: makani's ``comm.init`` needs a process group even at world size 1
(``polaris_eval_inference.pbs`` header, job 7632577).

Output tokens (key on these, not on rc -- CLAUDE.md #14):
  CLIMATE_ROLLOUT_OK member=… steps=… out=…                     rc 0
  CLIMATE_ROLLOUT_TRUNCATED step=… channel=… member=… out=…     rc 3 (outputs written)
  ERROR <reason>                                                 rc 1/2

Usage::

    python -m torch.distributed.run --standalone --nproc_per_node=1 \\
        scripts/climate_rollout.py --run-dir $RUN --ckpt $RUN/training_checkpoints/best_ckpt_mp0.tar \\
        --start 2044 1092 --n-steps 7667 --score-start 2045 0 --member-id m00 \\
        --out $OUT/m00.nc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("climate_rollout")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--pack", type=Path, default=None,
                   help="Pack root holding train/ valid/ test/ YYYY.h5 "
                        "(default: parent of the run's train_data_path)")
    p.add_argument("--start", required=True, type=int, nargs=2, metavar=("YEAR", "FRAME"))
    p.add_argument("--n-steps", required=True, type=int,
                   help="Leads to roll (6 h each). 7667 = Oct 1 2044 → 2049-12-31 18:00")
    p.add_argument("--score-start", type=int, nargs=2, metavar=("YEAR", "FRAME"), default=None,
                   help="First scored valid time (default: Jan 1 00:00 of the year after --start)")
    p.add_argument("--member-id", required=True)
    p.add_argument("--chunk-len", type=int, default=40)
    p.add_argument("--out", required=True, type=Path, help="Output NetCDF path")
    p.add_argument("--years-dir", type=Path, default=None,
                   help="Directory of YYYY.h5 symlinks (default: <out dir>/years_<first>_<last>)")
    p.add_argument("--snapshot-every", type=int, default=0)
    p.add_argument("--zonal-mean", action="store_true")
    p.add_argument("--time-means", type=Path, default=None,
                   help="Default: the run config's time_means_path")
    p.add_argument("--git-sha", default="unknown")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-assert-contract", action="store_true")
    p.add_argument("--dry-air-fix", choices=("config", "on", "off"), default="config",
                   help="Dry-air mass fix (sfno_training/models/mass_fix.py; DIAGNOSTIC). "
                        "'config' = the run's conserve_dry_air (absent = off); 'on'/'off' "
                        "override it, e.g. to apply it post hoc to a checkpoint trained without")
    return p.parse_args(argv)


def _sha256_prefix(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _ckpt_epoch(path: Path):
    import torch
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 -- legacy (non-zip) format cannot mmap
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    ep = ck.get("epoch") if isinstance(ck, dict) else None
    del ck
    return int(ep) if ep is not None else -1


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import torch
    from sfno_inference import climate_driver as cd
    from sfno_inference.checkpoint_loader import build_wrapper_from_checkpoint, load_eval_params
    from sfno_training.trainer.plasim_trainer import _plasim_get_dataloader
    from eval_inference import (_read_lat_lon_from_run, _resolve_and_check_channel_names,
                                _resolve_device)

    device = _resolve_device(args.device)
    start = tuple(args.start)
    score_start = tuple(args.score_start) if args.score_start else (start[0] + 1, 0)

    eval_params = load_eval_params(args.run_dir, K=1)       # valid_autoreg_steps = 0
    trained_with_fix = bool(eval_params.get("conserve_dry_air", False))
    if args.dry_air_fix != "config":
        eval_params.conserve_dry_air = args.dry_air_fix == "on"
    dry_air_fix = bool(eval_params.get("conserve_dry_air", False))
    pack = args.pack
    if pack is None:
        tdp = eval_params.train_data_path
        pack = Path(tdp[0] if isinstance(tdp, list) else tdp).parent
    time_means = args.time_means or getattr(eval_params, "time_means_path", None)

    years = cd.years_needed(start[0], start[1], args.n_steps)
    year_files = cd.locate_year_files(pack, years)
    years_dir = args.years_dir or args.out.parent / f"years_{years[0]}_{years[-1]}"
    cd.build_year_dir(year_files, years_dir)

    wrapper = build_wrapper_from_checkpoint(eval_params, args.ckpt, device=device)
    _, dataset, _ = _plasim_get_dataloader(eval_params, str(years_dir), device, mode="eval")
    ts_mode = cd.timestamp_axis_mode(dataset)
    first_file = Path(dataset.files_paths[0])
    channel_names = _resolve_and_check_channel_names(args.run_dir, first_file)
    lat, lon = _read_lat_lon_from_run(args.run_dir, first_file)

    provenance = dict(
        member_id=args.member_id, run_dir=str(args.run_dir), ckpt=str(args.ckpt),
        ckpt_sha256_16=_sha256_prefix(args.ckpt), ckpt_epoch=_ckpt_epoch(args.ckpt),
        git_sha=args.git_sha, pack=str(pack), years_dir=str(years_dir),
        timestamp_axis=ts_mode, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        dry_air_fix=int(dry_air_fix), dry_air_fix_in_training=int(trained_with_fix),
    )
    if dry_air_fix and wrapper.preprocessor.dry_air_fix is None:
        raise RuntimeError("DRY_AIR_FIX requested but the wrapper's preprocessor has none")
    res = cd.run_member(
        wrapper=wrapper, dataset=dataset, eval_params=eval_params, device=device,
        start=start, n_steps=args.n_steps, score_start=score_start,
        chunk_len=args.chunk_len, out_path=args.out, channel_names=channel_names,
        lat=lat, lon=lon, time_means_path=time_means, snapshot_every=args.snapshot_every,
        zonal=args.zonal_mean, provenance=provenance,
        assert_contract=not args.no_assert_contract,
    )

    # --- concise read-out (≤10 lines) ---
    stab = res.stability
    peak = (torch.cuda.max_memory_allocated(device) / 2**30
            if torch.device(device).type == "cuda" else 0.0)
    print(f"member={args.member_id} ckpt_epoch={provenance['ckpt_epoch']} "
          f"sha={provenance['ckpt_sha256_16']} timestamps={ts_mode['mode']} "
          f"feedback_dtype={res.feedback_dtype} dry_air_fix={int(dry_air_fix)}")
    for h in res.handoffs:
        print(f"HANDOFF step={h['step']} {h['from']}->{h['to']} local={h['to_local_idx']} "
              f"forcing_direct_match={h['forcing_direct_match']}")
    print(f"PERF s_per_step={res.s_per_step:.3f} steps={res.n_steps_run} "
          f"peak_gpu_mem_gib={peak:.2f}")
    for m, name in enumerate(cd.STABILITY_METRICS):
        cross = {f"{thr:g}x": int(stab['median_cross_step'][m, t])
                 for t, thr in enumerate(cd.STABILITY_THRESHOLDS)}
        nbad = {f"{thr:g}x": int((stab['first_bad_step'][m, t] > 0).sum())
                for t, thr in enumerate(cd.STABILITY_THRESHOLDS)}
        print(f"STABILITY {name}: channels_past={json.dumps(nbad)} median_cross={json.dumps(cross)}")
    if res.truncated_at_step > 0:
        print(f"CLIMATE_ROLLOUT_TRUNCATED step={res.truncated_at_step} "
              f"channel={res.truncated_channel} member={args.member_id} out={res.out_path}")
        return 3
    print(f"CLIMATE_ROLLOUT_OK member={args.member_id} steps={res.n_steps_run} out={res.out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- one greppable line, then the traceback
        print(f"ERROR CLIMATE_ROLLOUT_FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
        raise
