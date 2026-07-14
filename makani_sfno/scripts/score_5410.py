#!/usr/bin/env python3
"""score_5410.py — top-level score driver for the 5410 NWP eval.

Per docs/2026-05-08_sfno_5410_scoring_plan.md (v4.4).

Pipeline:
  1. Build the run plan (96 (Y, s) tuples or a trimmed subset).
  2. Preflight (in order):
     1a. clim source present + canonical channels.
     1b. truth h5 per-IC presence (for each (Y, s+k), k=0..K).
     1c. raw outputs complete (mode='exact' for full plan, 'subset' for trimmed).
     1d. adapted out dir empty or FORCE=1 (deletes prior NCs).
  3. Build compat climatology at out_root/baselines/climatology_proleptic.nc
     (renames time_of_year -> doy so score_nwp.py:79 reads ds["doy"]).
  4. For each (Y, s) in plan: adapter writes
     out_root/inference/nwp/Y{Y}_s{s:04d}.nc.
  5. Invoke score_nwp.main() with sys.argv set to use the adapted dir.
     Produces out_root/scores/nwp_scorecard*.csv + bias_maps_*.npy.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sfno_inference_5410.ic_offsets import nwp_ic_offsets_5410  # noqa: E402
from sfno_inference_5410.preflight import (  # noqa: E402
    assert_K_explicit,
    assert_output_dir_complete,
)
from sfno_inference_5410.score_adapter import (  # noqa: E402
    adapt_5410_ic_to_score_nwp,
    canonical_channel_names,
)
from sfno_inference_5410.score_climatology_compat import (  # noqa: E402
    write_compat_clim,
)
from sfno_inference_5410.stampede3_yaml_override import TEST_YEARS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True,
                   help="5410 inference run-root (raws under inference/upstream_raw)")
    p.add_argument("--K", type=int, required=True,
                   help="Forecast-leads horizon. Canonical: 60.")
    p.add_argument("--years", type=int, nargs="+", default=list(TEST_YEARS),
                   help="Subset of test years to score (default: 121..128)")
    p.add_argument("--ic-subset", type=str, default=None,
                   help="Explicit Y:s,Y:s,... overrides --years/--limit-ics")
    p.add_argument("--limit-ics", type=int, default=None,
                   help="Smoke knob: cap the run plan at the first N ICs")
    p.add_argument("--truth-h5-dir", type=Path,
                   default=Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data"),
                   help="Derecho per-timestep truth h5 dir")
    p.add_argument("--clim-src", type=Path,
                   default=Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/baselines/climatology_proleptic_5410.nc"),
                   help="5410 climatology source NC (will be coord-renamed into out_root/baselines/)")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Per-eval scoring root (created fresh per RUN_TAG)")
    p.add_argument("--run-tag", type=str, required=True,
                   help="Run tag, e.g. 20260508_eval-<sha>_5410-<gsha>_ckpt_epoch_50")
    p.add_argument("--ckpt-path", type=str,
                   default="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar")
    p.add_argument("--eval-sha7", type=str, default="")
    p.add_argument("--data-sha7", type=str, default="5410-v2.0",
                   help="5410 data/group provenance label (data-sha7 slot in score_nwp metadata)")
    p.add_argument("--train-sha7", type=str, default="ckpt_epoch_50",
                   help="Model checkpoint label (train-sha7 slot)")
    p.add_argument("--force", action="store_true",
                   help="If out_root/inference/nwp/ is non-empty, DELETE prior NCs and rebuild")
    p.add_argument("--no-score", action="store_true",
                   help="Adapt only; skip the score_nwp.py invocation (debug)")
    return p.parse_args()


def _build_plan(args: argparse.Namespace) -> list[dict]:
    """Build the (Y, s) run plan from CLI args. Each entry has Y, s,
    ic_nc, save_basename keys (matches the orchestrator's build_run_plan
    output shape, minimally)."""
    K = args.K
    upstream_raw = args.run_root / "inference" / "upstream_raw"

    if args.ic_subset:
        pairs = []
        for token in args.ic_subset.split(","):
            t = token.strip()
            if not t:
                continue
            try:
                y_str, s_str = t.split(":")
                pairs.append((int(y_str), int(s_str)))
            except (ValueError, IndexError):
                raise ValueError(f"--ic-subset token {t!r} not in 'Y:s' format")
    else:
        n_samples_by_year = {Y: (1464 if Y in (124, 128) else 1460) for Y in args.years}
        pairs = []
        for Y in args.years:
            offsets = nwp_ic_offsets_5410(n_samples_by_year[Y], K=K)
            for s in offsets:
                pairs.append((Y, s))
        if args.limit_ics is not None:
            if args.limit_ics < 1:
                raise ValueError(f"--limit-ics must be >= 1, got {args.limit_ics}")
            pairs = pairs[: args.limit_ics]

    plan = []
    for Y, s in pairs:
        save_basename = f"Y{Y}_s{s:04d}"
        ic_nc = upstream_raw / f"{save_basename}_member000_y{Y:04d}.nc"
        plan.append({
            "Y": int(Y),
            "s": int(s),
            "save_basename": save_basename,
            "ic_nc": ic_nc,
        })
    return plan


def _preflight_clim(clim_src: Path) -> list[str]:
    """1a-1b: assert clim presence + canonical channels. Returns the 53
    channel names from the clim (canonical order is enforced)."""
    import xarray as xr

    if not clim_src.is_file():
        raise ValueError(f"clim source not found: {clim_src}")
    canonical = canonical_channel_names()
    with xr.open_dataset(clim_src) as ds:
        if "channel" not in ds.coords:
            raise ValueError(f"clim {clim_src} has no 'channel' coord")
        chs = list(map(str, ds["channel"].values))
        if chs != canonical:
            raise ValueError(
                f"clim {clim_src} channel coord != canonical 53-name list. "
                f"got first: {chs[:5]}, expected first: {canonical[:5]}; "
                f"got last: {chs[-3:]}, expected last: {canonical[-3:]}"
            )
        if ds.sizes["channel"] != 53:
            raise ValueError(
                f"clim has channel size {ds.sizes['channel']}, expected 53"
            )
    return canonical


def _preflight_truth_h5_for_plan(truth_h5_dir: Path, plan: list[dict],
                                  K: int) -> None:
    """1c (renamed): for each (Y, s) in plan, assert each (Y, s+k) h5
    file exists for k=0..K. Catches holey truth coverage at submit time.
    """
    import h5py

    if not truth_h5_dir.is_dir():
        raise ValueError(f"truth h5 dir not found: {truth_h5_dir}")

    sampled = False
    for entry in plan:
        Y, s = entry["Y"], entry["s"]
        for k in range(K + 1):   # k=0 (IC) through k=K (last lead)
            s_target = s + k
            fp = truth_h5_dir / f"{Y}_{s_target:04d}.h5"
            if not fp.is_file():
                raise FileNotFoundError(
                    f"missing truth h5 for Y={Y} s+{k}={s_target}: {fp}"
                )
        # Sample-check tas + sic shape on the IC file of the first plan entry.
        if not sampled:
            with h5py.File(truth_h5_dir / f"{Y}_{s:04d}.h5", "r") as f:
                tas_shape = f["input/tas"].shape
                if tas_shape != (64, 128):
                    raise ValueError(
                        f"truth h5 input/tas shape {tas_shape} != (64, 128)"
                    )
                if "input/sic" not in f:
                    raise ValueError(
                        f"truth h5 {Y}_{s:04d}.h5 missing input/sic "
                        "(required by tas_no_ice mask)"
                    )
                sic_shape = f["input/sic"].shape
                if sic_shape != (64, 128):
                    raise ValueError(
                        f"truth h5 input/sic shape {sic_shape} != (64, 128)"
                    )
                import numpy as _np
                sic_arr = _np.asarray(f["input/sic"][...], dtype=_np.float32)
                if not _np.all(_np.isfinite(sic_arr) | _np.isnan(sic_arr)):
                    raise ValueError(
                        f"truth h5 input/sic has +/-inf values in {Y}_{s:04d}.h5"
                    )
                finite_vals = sic_arr[_np.isfinite(sic_arr)]
                tol = 1e-4
                if finite_vals.size and (
                    finite_vals.min() < -tol or finite_vals.max() > 1 + tol
                ):
                    raise ValueError(
                        f"truth h5 input/sic finite values out of [0,1] "
                        f"(tol={tol}) in {Y}_{s:04d}.h5: "
                        f"min={float(finite_vals.min()):.6g}, "
                        f"max={float(finite_vals.max()):.6g}"
                    )
            sampled = True


def _preflight_raw_outputs(upstream_raw: Path, plan: list[dict], K: int,
                            full_plan_size: int) -> None:
    """1d: raws complete. mode='exact' if scoring full 96-IC plan;
    'subset' if scoring a trimmed plan (e.g. smoke against existing
    96-IC raw dir)."""
    mode = "exact" if len(plan) == full_plan_size else "subset"
    assert_output_dir_complete(upstream_raw, plan, K, mode=mode)


def _preflight_adapted_out(out_root: Path, force: bool) -> None:
    """1e: adapted out dir empty or force=True (active deletion).
    Codex round-7 fix #2 + #3: enforced INSIDE score_5410.py (not
    only the driver) so direct invocations are safe."""
    nwp_dir = out_root / "inference" / "nwp"
    if nwp_dir.is_dir():
        ncs = sorted(nwp_dir.glob("*.nc"))
        if ncs:
            if force:
                print(f"[score_5410] --force: deleting {len(ncs)} prior adapted NCs at {nwp_dir}")
                for nc in ncs:
                    nc.unlink()
            else:
                raise ValueError(
                    f"{nwp_dir} is non-empty ({len(ncs)} prior adapted NCs); "
                    f"score_nwp.py would silently include these in the scorecard. "
                    f"Pass --force to delete + rebuild, or use a fresh OUT_ROOT."
                )
    nwp_dir.mkdir(parents=True, exist_ok=True)


def _full_plan_size(K: int) -> int:
    """Full 96-IC plan size: 8 years × 12 ICs."""
    return 8 * 12


def main() -> int:
    args = _parse_args()
    K = args.K
    assert_K_explicit(K)

    out_root: Path = args.out_root
    print(f"[score_5410] RUN_ROOT={args.run_root}")
    print(f"[score_5410] OUT_ROOT={out_root}")
    print(f"[score_5410] K={K} run_tag={args.run_tag}")

    # --- build plan -------------------------------------------------
    plan = _build_plan(args)
    print(f"[score_5410] plan: {len(plan)} ICs "
          f"(first: {[(e['Y'], e['s']) for e in plan[:3]]}...)")

    # --- preflight --------------------------------------------------
    print(f"[score_5410] preflight 1a-1b: clim source + canonical channels")
    channel_names = _preflight_clim(args.clim_src)

    print(f"[score_5410] preflight 1c: per-IC truth h5 presence "
          f"({len(plan)} ICs × (K+1)={K + 1} files)")
    _preflight_truth_h5_for_plan(args.truth_h5_dir, plan, K)

    print(f"[score_5410] preflight 1d: raw outputs complete")
    upstream_raw = args.run_root / "inference" / "upstream_raw"
    _preflight_raw_outputs(upstream_raw, plan, K, _full_plan_size(K))

    print(f"[score_5410] preflight 1e: adapted out dir empty (or --force)")
    _preflight_adapted_out(out_root, args.force)

    # --- compat clim ------------------------------------------------
    compat_clim = out_root / "baselines" / "climatology_proleptic.nc"
    print(f"[score_5410] writing compat clim → {compat_clim}")
    write_compat_clim(args.clim_src, compat_clim)

    # --- adapt 96 ICs -----------------------------------------------
    nwp_dir = out_root / "inference" / "nwp"
    for i, entry in enumerate(plan, start=1):
        out_nc = nwp_dir / f"{entry['save_basename']}.nc"
        adapt_5410_ic_to_score_nwp(
            raw_nc_path=entry["ic_nc"],
            truth_h5_dir=args.truth_h5_dir,
            Y=entry["Y"], s=entry["s"], K=K,
            out_nc_path=out_nc,
            ckpt_path=args.ckpt_path,
            eval_sha7=args.eval_sha7,
            data_sha7=args.data_sha7,
            train_sha7=args.train_sha7,
            run_tag=args.run_tag,
        )
        print(f"[{i:>2}/{len(plan)}] adapted {out_nc.name}")

    print(f"[score_5410] adapter done: {len(plan)} NCs in {nwp_dir}")

    if args.no_score:
        print(f"[score_5410] --no-score: stopping after adapter")
        return 0

    # --- invoke score_nwp.main() in-process -------------------------
    print(f"[score_5410] invoking score_nwp.py …")
    score_argv = [
        "score_nwp.py",
        "--out-root", str(out_root),
        "--clim-nc", str(compat_clim),
    ]
    saved_argv = sys.argv
    try:
        sys.argv = score_argv
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        import score_nwp   # type: ignore[import-not-found]
        rc = score_nwp.main() if hasattr(score_nwp, "main") else 0
        # score_nwp.py uses module-level argparse; if it doesn't have a
        # main(), invoking it via subprocess is the safer path.
        if rc is None:
            rc = 0
    finally:
        sys.argv = saved_argv

    if rc != 0:
        # Fallback: spawn as subprocess. score_nwp.py may rely on
        # `if __name__ == "__main__":` guard.
        print(f"[score_5410] in-process invocation returned {rc}; "
              f"falling back to subprocess")
        proc = subprocess.run(
            [sys.executable, "-u", str(_REPO_ROOT / "scripts" / "score_nwp.py")] + score_argv[1:],
            cwd=str(_REPO_ROOT),
            env={**os.environ, "PYTHONPATH": f"{_REPO_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")},
        )
        if proc.returncode != 0:
            return proc.returncode

    print(f"[score_5410] DONE — outputs at {out_root}/scores/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
