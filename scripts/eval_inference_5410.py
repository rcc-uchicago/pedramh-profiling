#!/usr/bin/env python3
"""Orchestrator: in-process 96-IC SFNO-5410 NWP inference (v2.1).

Builds upstream's ``Stepper`` once (loading model + checkpoint exactly
ONCE), then loops over the 96 (Y, s) tuples calling
``Stepper.reconfigure_for_ic(...)`` (LP-004) and ``Stepper.predict()``
for each IC. Mirrors the own-track architecture
(``scripts/eval_inference.py``) which does the same: one process, model
loaded once, dataset rebuilt per IC.

Pre-2026-05-08, this orchestrator launched 96 separate ``python
long_inference.py`` subprocesses, paying ~95 s of cold-import + ckpt
reload **per IC**. Wall-clock was ~4 h, ~76 % of which was setup
overhead. The legacy subprocess path is preserved as a reference at
``tests/sfno_inference_5410/integration/_legacy_subprocess_launcher.py``
for A/B equivalence testing.

Pre-conditions (must be already satisfied before invocation):
  * Per-Y override yamls exist under ``$config_dir`` (built by
    ``scripts/build_5410_yaml_override.py --all-years --K 60 ...``).
  * Per-Y ckpt symlink shims exist under ``$exp_dir`` (same script).
  * ``<run_root>/inference/ic_source.json`` exists (§3 P-7 gate ran).
  * All 96 IC NCs at ``<run_root>/inference/ic_nc/<Y>_<ssss>.nc``.
  * Upstream ``long_inference.py`` carries LP-003 (4 allocator + 2
    continuation hunks) AND LP-004 (Stepper.reconfigure_for_ic).
  * Output dir (``<run_root>/inference/upstream_raw/``) is empty.

Usage
-----
Dry-run (print plan, no Stepper construction)::

    python scripts/eval_inference_5410.py \\
        --run-root $RESULTS/sfno_eval_5410/<run_tag> \\
        --config-dir $REPO_ROOT/config --K 60

Launch (in-process, one Python process)::

    python scripts/eval_inference_5410.py \\
        --run-root $RESULTS/sfno_eval_5410/<run_tag> \\
        --config-dir $REPO_ROOT/config --K 60 --launch
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import cftime

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sfno_inference_5410.ic_offsets import nwp_ic_offsets_5410  # noqa: E402
from sfno_inference_5410.ic_source import resolve_ic_nc_path  # noqa: E402
from sfno_inference_5410.preflight import (  # noqa: E402
    assert_K_explicit,
    assert_final_datetime_matches,
    assert_output_dir_complete,
    assert_output_dir_empty,
    assert_upstream_boundary_phase,
    assert_upstream_patched,
    assert_upstream_patched_lp004,
    assert_yaml_horizon,
    assert_yamls_share_static_arch,
)
from sfno_inference_5410.stampede3_yaml_override import (  # noqa: E402
    TEST_YEARS,
    config_basename_for_year,
)
from sfno_inference_5410.upstream_hydration import (  # noqa: E402
    hydrate_static_params,
    set_per_ic_params,
    set_per_y_params,
)


# Upstream entry point (set on Stampede3 by external env).
_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)
_UPSTREAM_LONG_INFERENCE = _UPSTREAM_REPO / "long_inference.py"


def init_datetime_for(Y: int, s: int) -> cftime.DatetimeProlepticGregorian:
    """Return the proleptic-gregorian init datetime for sample ``s`` of year ``Y``."""
    base = cftime.DatetimeProlepticGregorian(Y, 1, 1, 0, has_year_zero=True)
    return base + dt.timedelta(hours=s * 6)


def final_datetime_for(
    init_dt: cftime.DatetimeProlepticGregorian,
    K: int,
) -> cftime.DatetimeProlepticGregorian:
    """Return ``init_dt + (K + 1) * 6h`` — the upstream ``--final_datetime``.

    Defines the rollout horizon as ``K`` scored forecast leads. The
    upstream output buffer is sized at ``(final - init) / 6h = K + 1``
    rows: IC at index 0, forecast leads at indices 1..K. The 61st
    forward pass is computed but discarded by upstream's
    ``time_step_in_year + 1 < shape[1]`` save guard, so K=60 yields 61
    raw rows for 60 saved leads.
    """
    assert_K_explicit(K)
    return init_dt + dt.timedelta(hours=(K + 1) * 6)


def build_argv_for_ic(
    Y: int,
    s: int,
    *,
    K: int,
    run_root: Path,
    config_dir: Path,
) -> dict:
    """Return the per-IC entry dict for one (Y, s) invocation.

    Returns a dict with keys::

        {"Y": int, "s": int,
         "ic_nc": Path, "init_datetime": cftime.Datetime,
         "final_datetime": cftime.Datetime, "save_basename": str,
         "config": str, "yaml": Path, "output_dir": Path}

    Note: as of v2.1 (2026-05-08) the orchestrator runs the 96 ICs
    in-process via ``Stepper.reconfigure_for_ic`` (LP-004), so this
    helper no longer materializes a subprocess argv. The per-IC entries
    are consumed by ``main()`` (in-process) and by the legacy A/B
    fixture at ``tests/sfno_inference_5410/integration/_legacy_subprocess_launcher.py``
    (subprocess; reference path).
    """
    assert_K_explicit(K)
    yaml_name = f"SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y{Y}.yaml"
    yaml_path = Path(config_dir) / yaml_name
    config_basename = config_basename_for_year(Y)
    output_dir = Path(run_root) / "inference" / "upstream_raw"
    save_basename = f"Y{Y}_s{s:04d}"
    ic_nc = resolve_ic_nc_path(Y, s, run_root)
    if "," in str(ic_nc):
        raise ValueError(
            f"single-IC invariant violated: ic_nc path contains comma: {ic_nc!r}"
        )
    init_dt = init_datetime_for(Y, s)
    final_dt = final_datetime_for(init_dt, K)
    assert_final_datetime_matches(init_dt, final_dt, K)

    return {
        "Y": int(Y),
        "s": int(s),
        "ic_nc": ic_nc,
        "init_datetime": init_dt,
        "final_datetime": final_dt,
        "save_basename": save_basename,
        "config": config_basename,
        "yaml": yaml_path,
        "output_dir": output_dir,
    }


def build_run_plan(
    run_root: Path,
    config_dir: Path,
    *,
    K: int,
    years=TEST_YEARS,
    n_samples_by_year: dict[int, int] | None = None,
) -> list[dict]:
    """Return the run plan (one entry per (Y, s) tuple).

    ``K`` is the forecast-leads horizon (required keyword-only). Passed
    to both ``build_argv_for_ic`` (sets final_datetime) and
    ``nwp_ic_offsets_5410`` (validates that ``last_s + K < n_samples``).
    """
    assert_K_explicit(K)
    if n_samples_by_year is None:
        n_samples_by_year = {Y: (1464 if Y in (124, 128) else 1460) for Y in years}

    plan: list[dict] = []
    for Y in years:
        offsets = nwp_ic_offsets_5410(n_samples_by_year[Y], K=K)
        for s in offsets:
            plan.append(build_argv_for_ic(
                Y, s, K=K, run_root=run_root, config_dir=config_dir,
            ))

    for entry in plan:
        if "," in str(entry["ic_nc"]):
            raise ValueError(
                f"single-IC invariant violated in run plan: {entry['ic_nc']!r}"
            )
    return plan


def _run_one_ic(stepper, entry, K, *, val_year_changed: bool) -> None:
    """Reconfigure for one IC + predict + four post-reconfigure assertions.

    Codex round-1 blocker #3: the BCS loader's date_range length IS the
    rollout length. Asserting ``len(stepper.data_loader_bcs) == K + 1``
    after reconfigure proves the rollout itself is short, not just the
    saved file.

    Codex round-2 fix #2: explicit ``if ... raise ValueError`` instead
    of raw ``assert`` so ``python -O`` cannot disable these load-bearing
    checks.
    """
    boundary_leap_year = int(getattr(stepper.params, "boundary_leap_year", entry["Y"]))
    boundary_no_leap_year = int(
        getattr(stepper.params, "boundary_no_leap_year", entry["Y"])
    )

    stepper.reconfigure_for_ic(
        init_datetime=entry["init_datetime"],
        final_datetime=entry["final_datetime"],
        init_nc_filepaths=[entry["ic_nc"]],
        save_basename=entry["save_basename"],
        output_dir=entry["output_dir"],
        val_year_start=entry["Y"],
        val_year_end=entry["Y"] + 1,
        leap_year=boundary_leap_year,
        no_leap_year=boundary_no_leap_year,
        val_year_changed=val_year_changed,
    )

    if len(stepper.data_loader_bcs) != K + 1:
        raise ValueError(
            f"BCS loader length {len(stepper.data_loader_bcs)} != K+1={K+1} "
            f"(Y={entry['Y']}, s={entry['s']}); "
            f"prediction_duration_days may not have propagated"
        )
    if len(stepper.data_loader) != 1:
        raise ValueError(
            f"IC loader length {len(stepper.data_loader)} != 1 "
            f"(Y={entry['Y']}, s={entry['s']}); "
            f"single-IC invariant violated"
        )
    if len(stepper.params.init_nc_filepaths) != 1:
        raise ValueError(
            f"init_nc_filepaths len != 1: {stepper.params.init_nc_filepaths}"
        )
    if getattr(stepper.params, "nc_bc_offset", None) != 0:
        raise ValueError(
            f"nc_bc_offset must be 0 for 5410 NWP eval boundary alignment; "
            f"got {getattr(stepper.params, 'nc_bc_offset', None)!r}"
        )
    expected_final = entry["init_datetime"] + dt.timedelta(hours=(K + 1) * 6)
    if stepper.params.final_datetime != expected_final:
        raise ValueError(
            f"final_datetime {stepper.params.final_datetime} != "
            f"init + (K+1)*6h = {expected_final} "
            f"(Y={entry['Y']}, s={entry['s']})"
        )

    stepper.predict()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-root", type=Path, required=True,
                   help="Eval run root (parent of inference/, scores/, etc.)")
    p.add_argument("--config-dir", type=Path, required=True,
                   help="Where the per-Y override yamls live")
    p.add_argument("--launch", action="store_true",
                   help="Actually run inference (default: dry-run, print plan)")
    p.add_argument("--years", type=int, nargs="+", default=list(TEST_YEARS),
                   help="Subset of test years to run (default: 121..128)")
    p.add_argument("--K", type=int, required=True,
                   help="Forecast-leads horizon (positive int). Per-IC final_datetime "
                        "= init + (K+1)*6h. Canonical eval-track value: 60.")
    p.add_argument("--limit-ics", type=int, default=None,
                   help="Smoke-test knob: cap the run plan at the first N ICs "
                        "(after --years filtering). Useful for 1-IC sanity smokes.")
    p.add_argument("--ic-subset", type=str, default=None,
                   help="Test/debug knob: replace the run plan with an explicit "
                        "list of (Y, s) tuples. Format: 'Y:s,Y:s,...' "
                        "(e.g., '121:0,122:0' for the cross-year A/B test). "
                        "Overrides --years and --limit-ics. Each (Y, s) must "
                        "still be in the IC offsets schedule for that Y.")
    p.add_argument("--async-save", action="store_true",
                   help="Pass async_save=True to upstream Stepper. v2.1 default is "
                        "False (synchronous saves) so post-IC assertions and "
                        "reconfigure can't race a still-pending save thread.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    K = args.K
    assert_K_explicit(K)
    run_root = args.run_root
    raw_steps = K + 1
    raw_hours = raw_steps * 6
    print(f"[orchestrator] forecast_K={K} raw_steps={raw_steps} raw_hours={raw_hours} "
          f"prediction_duration_days={raw_hours / 24.0}")

    if args.ic_subset is not None:
        # --ic-subset overrides --years and --limit-ics. Build entries
        # directly from the explicit (Y, s) list.
        ic_pairs = []
        for token in args.ic_subset.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                Y_str, s_str = token.split(":")
                ic_pairs.append((int(Y_str), int(s_str)))
            except (ValueError, IndexError):
                raise ValueError(
                    f"--ic-subset token {token!r} not in 'Y:s' format"
                )
        if not ic_pairs:
            raise ValueError(f"--ic-subset is empty: {args.ic_subset!r}")
        plan = [
            build_argv_for_ic(
                Y, s, K=K, run_root=run_root, config_dir=args.config_dir,
            )
            for Y, s in ic_pairs
        ]
        print(f"[orchestrator] --ic-subset → run plan = {len(plan)} explicit IC(s): "
              f"{[(e['Y'], e['s']) for e in plan]}")
    else:
        plan = build_run_plan(
            run_root, args.config_dir, K=K, years=tuple(args.years),
        )
        if args.limit_ics is not None:
            if args.limit_ics < 1:
                raise ValueError(f"--limit-ics must be >= 1, got {args.limit_ics}")
            plan = plan[: args.limit_ics]
            print(f"[orchestrator] --limit-ics {args.limit_ics} → run plan trimmed to "
                  f"{len(plan)} IC(s)")
    out_dir = plan[0]["output_dir"]

    # === preflights (run on dry-run too — except output_dir_empty). ===
    assert_upstream_patched(_UPSTREAM_LONG_INFERENCE)
    assert_upstream_patched_lp004(_UPSTREAM_LONG_INFERENCE)
    assert_upstream_boundary_phase(_UPSTREAM_LONG_INFERENCE)
    print(f"[preflight] upstream LP-003+LP-004 markers + boundary phase OK at {_UPSTREAM_LONG_INFERENCE}")

    yaml_paths = sorted({Path(e["yaml"]) for e in plan})
    for yp in yaml_paths:
        assert_yaml_horizon(yp, K)
    print(f"[preflight] yaml-horizon OK for {len(yaml_paths)} per-Y yamls "
          f"(ensemble_inference_hours={raw_hours}, "
          f"prediction_duration_days={raw_hours / 24.0})")

    if len(yaml_paths) >= 2:
        assert_yamls_share_static_arch(yaml_paths)
        print(f"[preflight] static-arch invariant OK across {len(yaml_paths)} yamls")

    for entry in plan:
        print(f"[ic Y={entry['Y']:>3} s={entry['s']:04d}] "
              f"init={entry['init_datetime']} final={entry['final_datetime']} "
              f"forecast_K={K} raw_steps={raw_steps}")

    if not args.launch:
        print("[orchestrator] dry-run (no --launch); plan above")
        return 0

    # === launch-only preflights (Codex round-2 fix #3: not on dry-run). ===
    assert_output_dir_empty(out_dir)
    print(f"[preflight] output dir {out_dir} is empty")

    # === one-time setup (paid ONCE across all 96 ICs). ===
    # cd into upstream so `from utils.YParams import YParams` etc. resolve.
    os.chdir(_UPSTREAM_REPO)
    sys.path.insert(0, str(_UPSTREAM_REPO))

    print(f"[orchestrator] hydrating params from {yaml_paths[0]}...")
    params = hydrate_static_params(
        yaml_paths[0], K=K, upstream_repo=_UPSTREAM_REPO,
    )
    set_per_y_params(params, Y=plan[0]["Y"])
    set_per_ic_params(
        params,
        init_datetime=plan[0]["init_datetime"],
        final_datetime=plan[0]["final_datetime"],
        init_nc_filepaths=[plan[0]["ic_nc"]],
        save_basename=plan[0]["save_basename"],
        output_dir=plan[0]["output_dir"],
    )

    # Mirror upstream main()'s torch-level globals (long_inference.py:1463-1465).
    # Critical: cudnn.benchmark=True selects autotuned kernels; without it
    # cudnn picks conservative kernels and fp16/AMP output diverges from
    # the legacy path (verified by gate-A failure + legacy-vs-legacy
    # bit-exact baseline 2026-05-08).
    import torch  # already imported transitively, re-exposed here for clarity
    torch.manual_seed(0)  # world_rank=0 in single-rank inference
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    print(f"[orchestrator] constructing Stepper (one-time model + ckpt load)...")
    t_setup = time.time()
    from long_inference import Stepper  # type: ignore
    stepper = Stepper([params], world_rank=0, async_save=args.async_save)
    print(f"[orchestrator] Stepper ready in {time.time() - t_setup:.1f}s")

    # === per-IC loop. ===
    t0 = time.time()
    # First IC (val_year_changed=True so constant_boundary_data is populated).
    t_ic = time.time()
    _run_one_ic(stepper, plan[0], K, val_year_changed=True)
    print(f"[1/{len(plan)}] {plan[0]['save_basename']} ({time.time() - t_ic:.1f}s)")

    # Remaining ICs.
    for i, entry in enumerate(plan[1:], start=2):
        t_ic = time.time()
        prev_Y = plan[i - 2]["Y"]
        val_year_changed = entry["Y"] != prev_Y
        if val_year_changed:
            set_per_y_params(stepper.params, Y=entry["Y"])
        set_per_ic_params(
            stepper.params,
            init_datetime=entry["init_datetime"],
            final_datetime=entry["final_datetime"],
            init_nc_filepaths=[entry["ic_nc"]],
            save_basename=entry["save_basename"],
            output_dir=entry["output_dir"],
        )
        _run_one_ic(stepper, entry, K, val_year_changed=val_year_changed)
        print(f"[{i}/{len(plan)}] {entry['save_basename']} "
              f"({time.time() - t_ic:.1f}s"
              f"{', Y boundary' if val_year_changed else ''})")

    elapsed_min = (time.time() - t0) / 60.0
    print(f"[orchestrator] all {len(plan)} ICs done in {elapsed_min:.1f} min")

    # === postflight: exact filenames + time==K+1 + var set check. ===
    assert_output_dir_complete(out_dir, plan, K)
    print(f"[postflight] {len(plan)} NetCDFs match expected filenames + time-dim + var set")

    return 0


if __name__ == "__main__":
    sys.exit(main())
