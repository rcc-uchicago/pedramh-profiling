#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Combine and score a lagged ensemble -- stages 4 and 5 of the end-to-end plan.

Consumes a stagger-`d` sweep (`eval_inference.py --ic-mode lagged`) plus the depth
curve `sigma(k)` from `score_rollout_nc.py`'s `k56_metrics.h5`, and answers plan
§4.3's four questions in one pass:

  1. Does the ensemble mean beat the best single member?  (the blunt value test)
  2. Is the spread-skill ratio near 1, or is the ensemble over-confident?
  3. Is the rank histogram flat?  (diagnostic only -- members are not exchangeable)
  4. Does CRPS beat the deterministic baseline?

THE HEADLINE IS ACE2 EQ 8, and the four above are now secondary.  Watt-Meyer et al.,
arXiv:2411.11268 §4.3 defines the metric this lagged ensemble is meant to be scored
under, and it is not an RMSE:

    alpha = (1/C) sum_c sqrt( sum_{phi,lambda} w ( MEAN_{t,ens}[y_c - yhat_c] )^2 )

The time-and-ensemble average sits INSIDE the square, so alpha is the RMS of the
*mean* error -- a bias metric.  Squaring per snapshot first, as checks 1-4 do, keeps
the random component too, and the two answer different questions: a construction that
loses on instantaneous RMSE can still win on alpha, because averaging signed errors is
precisely what cancels random error.

Two further details of the paper's framework that checks 1-4 do not follow:

  * ACE2's own ensemble in eq 8 IS a lagged ensemble -- "eight 5-year long simulations,
    initialized at evenly spaced intervals", each supplying its prediction "for the
    corresponding time, from a simulation initialized at some previous time" -- and it
    is combined with a PLAIN, UNWEIGHTED mean.  No inverse-variance weighting appears
    anywhere in it.  We report alpha under both rules so the weighting has to earn
    itself against the paper's default rather than being assumed.
  * eq 8 is defined on NORMALISED fields, which is what licenses its arithmetic
    `(1/C) sum_c` channel reduction.  Checks 1-4 work in physical units and are
    therefore stuck with a median over 101 disparate scales -- a reduction that hides
    a single blown-up channel.  `ace2_dominant_channels` in the readout is the
    replacement for that blind spot.

Every number is latitude-weighted on the **equiangular** grid, never Gauss-Legendre
(change G: GL over-weights the polar row by 1.50x here and the only guard downstream
is a shape check that 180-vs-180 passes).

PASS = `LAGGED_ENSEMBLE_OK`.  Per CLAUDE.md #14 rc=0 is not the signal: a run that
scored no targets, or whose members disagreed about the truth field, exits 0 from the
OS' point of view and must not read as a result.

Run under the SFNO venv on a compute node (CLAUDE.md #3):

    python polaris/score_lagged_ensemble.py \\
        --nc-dir     $MEMBER_ROOT/runs/makani_eval/<run>_lagged_K56_d4/inference/lagged \\
        --sigma-h5   $MEMBER_ROOT/runs/makani_eval/<run>_K56/scores/k56_metrics.h5 \\
        --out-dir    $MEMBER_ROOT/runs/makani_eval/<run>_lagged_K56_d4/scores
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_MAKANI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MAKANI_ROOT / "src"))

logger = logging.getLogger("score_lagged_ensemble")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine and score a lagged ensemble.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--nc-dir", required=True, type=Path,
                   help="Directory of lagged rollout NetCDFs (inference/lagged)")
    p.add_argument("--sigma-h5", required=True, type=Path,
                   help="k56_metrics.h5 from score_rollout_nc.py; supplies sigma(k)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--min-members", type=int, default=None,
                   help="Keep only targets with at least this many members "
                        "(default: the modal count, i.e. the fully covered ones)")
    p.add_argument("--channel-chunk", type=int, default=16,
                   help="Channels per read block; bounds peak memory (default 16)")
    p.add_argument("--limit-targets", type=int, default=None, help="Smoke: score N targets")
    p.add_argument("--all-residues", action="store_true",
                   help="Score every covered target rather than only the modal depth "
                        "pattern. ⚠ Mixes ensembles with different minimum leads, so the "
                        "aggregate is not one construction -- diagnostic only.")
    p.add_argument("--truth-check-every", type=int, default=8,
                   help="Verify member truth agreement on every Nth target (0 = never). "
                        "Doubles that target's reads; it is the check that proves the "
                        "alignment against real data, so 0 is for smokes only.")
    p.add_argument("--provenance", default="")
    return p.parse_args()


def _load_sigma(path: Path, channels: list[str]) -> np.ndarray:
    """Return `(K, C)` RMSE-by-depth, checked against the ensemble's channel order.

    The curve is positional, so a channel-order difference between the sweep that
    produced sigma and the sweep being weighted would mis-weight every member with no
    error -- the v10.0/v10.1 contamination pattern.
    """
    import h5py

    with h5py.File(path, "r") as f:
        sigma = np.asarray(f["rmse_mean"][:], dtype=np.float64)
        sig_chan = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][:]]
    if sig_chan != channels:
        raise SystemExit(
            f"CHANNEL_ORDER_MISMATCH: {path} was computed on a different channel order "
            f"than the lagged sweep ({sig_chan[:3]}… vs {channels[:3]}…). Refusing to "
            "score: sigma is indexed positionally and would mis-weight every member."
        )
    return sigma


def _load_norm_sigma(path: Path, n_channels: int) -> np.ndarray:
    """Return the `(C,)` standard scaling for ACE2 eq 8's *normalised* units.

    Eq 8 is defined on standard-scaled fields, but the scaling **mean cancels** in the
    difference `y - yhat`, so only a per-channel sigma is needed and no new input file
    is required: `a_truth_mean` in `k56_metrics.h5` is already the truth's own
    lat-weighted anomaly amplitude about the pack climatology -- the project's existing
    normalisation convention (`score_rollout_nc.py` divides by it to form NRMSE).

    Averaged over the 56 leads because it is a property of the truth field, not of the
    forecast: it varies with lead only through which validation times each lead lands
    on, and the mean over an independent 24-IC sweep is the steadiest estimate we have.
    Using the lagged sweep's own 8 targets instead would make the normaliser depend on
    the sample being scored.
    """
    import h5py

    with h5py.File(path, "r") as f:
        a_truth = np.asarray(f["a_truth_mean"][:], dtype=np.float64)   # (K, C)
    sig = a_truth.mean(axis=0)
    if sig.shape != (n_channels,):
        raise SystemExit(
            f"NORM_SHAPE_MISMATCH: a_truth_mean reduced to {sig.shape}, expected "
            f"({n_channels},). Eq 8 would normalise the wrong axis."
        )
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
        raise SystemExit(
            "NORM_DEGENERATE: a_truth_mean has a non-positive or non-finite channel; "
            "that channel's contribution to alpha would be infinite."
        )
    return sig


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import xarray as xr
    from sfno_ensemble import align_members_by_target, coverage_report, index_rollout_dir
    from sfno_ensemble.combine import combine_target, load_target_stack
    from sfno_ensemble import scores as S

    idxs = index_rollout_dir(args.nc_dir)
    logger.info("indexed %d rollouts", len(idxs))

    table = align_members_by_target(idxs, min_members=1)
    modal = max({len(v) for v in table.values()},
                key=lambda c: sum(1 for v in table.values() if len(v) == c))
    keep = args.min_members if args.min_members is not None else modal
    table = {t: m for t, m in table.items() if len(m) >= keep}
    rep = coverage_report(table)
    logger.info("coverage: %s", rep)
    if not table:
        print(f"ERROR NO_FULL_TARGETS: no target has >= {keep} members", file=sys.stderr)
        return 2

    # Keep only the MODAL DEPTH PATTERN, and this is a correctness constraint rather
    # than tidiness.  A target's depths are T - s, so they depend on
    # (T - first_start) mod stride: at stride 4, target 56 gets depths 4, 8 … 56 while
    # target 53 gets 1, 5 … 53.  A lagged ensemble valid at T can only use forecasts
    # ALREADY ISSUED, so every member must share one minimum lead -- which is the
    # plan's figure (§3.1, "depths 4, 8, … 56").  A depth-1 member is a 6 h forecast
    # of a target the 24 h-lead ensemble could not have seen, and comparing an
    # ensemble against a 6 h baseline on one target and a 24 h baseline on the next
    # makes the headline meaningless.  Grouping by depth tuple picks the uniform
    # construction without needing to know the stride.
    if not args.all_residues:
        patterns: dict[tuple, list[int]] = {}
        for t, m in table.items():
            patterns.setdefault(tuple(x.depth_steps for x in m), []).append(t)
        # Tie-break on the DEEPEST minimum lead, not on encounter order. All `stride`
        # patterns have equally many targets, so a plain max() picked depths 1..53 --
        # an ensemble whose freshest member is a 6 h forecast, and whose headline is
        # then "can 14 members beat a 6 h forecast?". The plan's construction is
        # `d, 2d … K` (§3.1), the pattern with the largest minimum depth.
        best = max(patterns, key=lambda p: (len(patterns[p]), p[0]))
        dropped = len(table) - len(patterns[best])
        logger.info("depth patterns: %d (min depths %s); keeping %s..%s on %d targets "
                    "(dropped %d)", len(patterns), sorted(p[0] for p in patterns),
                    best[0], best[-1], len(patterns[best]), dropped)
        table = {t: table[t] for t in patterns[best]}

    targets = sorted(table)
    if args.limit_targets:
        targets = targets[: args.limit_targets]

    with xr.open_dataset(idxs[0].path, decode_timedelta=False) as ds0:
        channels = [str(c) for c in ds0["channel"].values]
        n_lat = int(ds0.sizes["lat"])
        n_lon = int(ds0.sizes["lon"])
    sigma = _load_sigma(args.sigma_h5, channels)
    lat_w = S.equiangular_weights(n_lat)
    C, E = len(channels), len(table[targets[0]])
    norm_c = _load_norm_sigma(args.sigma_h5, C)
    logger.info("scoring %d targets x %d members x %d channels", len(targets), E, C)

    # --- ACE2 eq 8 accumulators (arXiv:2411.11268 §4.3) ---------------------------
    # The overbar is a time- AND ensemble-average of the SIGNED error, taken before
    # squaring, so scalars-per-target are not enough: the error FIELD has to be
    # summed over targets.  We accumulate one running sum per MEMBER rather than one
    # per ensemble configuration, because `member_weights` depends only on (depth,
    # channel) and never on the target -- so every combination rule (uniform, the
    # 1/sigma^2 weights, any nested prefix) is an exact linear recombination of these
    # afterwards, at no extra read and no extra memory per rule.
    member_sum = np.zeros((E, C, n_lat, n_lon))
    truth_sum = np.zeros((C, n_lat, n_lon))
    weights_full = np.zeros((E, C))

    acc = {k: np.zeros((len(targets), C)) for k in
           ("rmse_weighted", "rmse_uniform", "rmse_shallowest", "rmse_best_member",
            "crps_ensemble", "crps_shallowest", "spread", "skill", "ssr", "ssr_raw")}
    n_ssr_undefined = 0
    # Nested sub-ensembles: the m freshest members, for m = 1 … E. Almost free (one
    # einsum each, the stack is already resident) and it turns a pass/fail on check 1
    # into the actionable question -- how many members are worth keeping, given that
    # the oldest are 300 h forecasts?
    nested_rmse = np.zeros((len(targets), E, C))
    rank_hist = np.zeros((C, E + 1))
    truth_disagreement = 0.0
    t0 = time.time()

    for n, target in enumerate(targets):
        members = table[target]
        check = bool(args.truth_check_every) and (n % args.truth_check_every == 0)
        for c0 in range(0, C, args.channel_chunk):
            sl = slice(c0, min(c0 + args.channel_chunk, C))
            st = load_target_stack(members, channels=sl, check_truth=check)
            truth_disagreement = max(truth_disagreement, st.truth_max_disagreement)

            comb = combine_target(st, sigma[:, sl])
            w, truth = comb["weights"], st.truth
            mem_rmse = S.member_rmse(st.members, truth, lat_w)      # (E, C_chunk)

            acc["rmse_weighted"][n, sl] = S.rmse(comb["mean_weighted"], truth, lat_w)
            acc["rmse_uniform"][n, sl] = S.rmse(comb["mean_uniform"], truth, lat_w)
            acc["rmse_shallowest"][n, sl] = mem_rmse[0]
            acc["rmse_best_member"][n, sl] = mem_rmse.min(axis=0)

            ss = S.spread_skill(st.members, truth, comb["mean_weighted"], w, lat_w)
            for k in ("spread", "skill", "ssr", "ssr_raw"):
                acc[k][n, sl] = ss[k]
            n_ssr_undefined += ss["n_undefined"]

            acc["crps_ensemble"][n, sl] = S.weighted_crps(st.members, truth, w, lat_w)
            # The deterministic baseline: the single best forecast available for this
            # target. A one-member "ensemble" has no spread, so its CRPS is its MAE.
            acc["crps_shallowest"][n, sl] = S.area_mean(
                np.abs(st.members[0] - truth), lat_w)
            rank_hist[sl] += S.rank_histogram(st.members, truth, lat_w)

            # Re-normalise the weights within each nested subset so every m is a
            # proper weighted mean rather than a partial sum.
            from sfno_ensemble.combine import weighted_mean as _wm
            for m in range(1, E + 1):
                wm = w[:m] / w[:m].sum(axis=0, keepdims=True)
                nested_rmse[n, m - 1, sl] = S.rmse(_wm(st.members[:m], wm), truth, lat_w)

            # ACE2 eq 8: sum the raw fields, not any scalar reduction of them.
            member_sum[:, sl] += st.members
            truth_sum[sl] += truth
            weights_full[:, sl] = w

        if n % 8 == 0 or n == len(targets) - 1:
            logger.info("target %d (%d/%d, %.1f min)", target, n + 1, len(targets),
                        (time.time() - t0) / 60)

    # --- reduce over targets, then over channels (median: 101 disparate scales) ---
    mean_over_t = {k: np.nanmean(v, axis=0) for k, v in acc.items()}
    med = {k: float(np.nanmedian(v)) for k, v in mean_over_t.items()}
    gain_vs_shallow = 1.0 - mean_over_t["rmse_weighted"] / mean_over_t["rmse_shallowest"]
    gain_vs_uniform = 1.0 - mean_over_t["rmse_weighted"] / mean_over_t["rmse_uniform"]
    crps_gain = 1.0 - mean_over_t["crps_ensemble"] / mean_over_t["crps_shallowest"]

    # --- ACE2 eq 8 (arXiv:2411.11268 §4.3) ----------------------------------------
    # Every rule below is an exact recombination of `member_sum`, so the uniform mean
    # the PAPER prescribes and the 1/sigma^2 mean this scorecard has been using can be
    # compared on identical data with no second pass.
    nt = len(targets)

    def _alpha(w_rule: np.ndarray, cw: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        pred = np.einsum("ec,echw->chw", w_rule, member_sum)
        return S.ace2_alpha(S.time_mean_bias(pred, truth_sum, nt), lat_w, norm_c,
                            channel_weights=cw)

    def _prefix(w_rule: np.ndarray, m: int) -> np.ndarray:
        """`m` freshest members, renormalised, zero-padded back to `(E, C)`."""
        out = np.zeros_like(w_rule)
        out[:m] = w_rule[:m] / w_rule[:m].sum(axis=0, keepdims=True)
        return out

    uniform_full = np.full((E, C), 1.0 / E)
    alpha_uniform, alpha_uniform_per_c = _alpha(uniform_full)      # the ACE2 headline
    alpha_sigma, _ = _alpha(weights_full)
    alpha_det, _ = _alpha(_prefix(uniform_full, 1))                # single freshest
    alpha_nested_uniform = [_alpha(_prefix(uniform_full, m))[0] for m in range(1, E + 1)]
    alpha_nested_sigma = [_alpha(_prefix(weights_full, m))[0] for m in range(1, E + 1)]
    best_m_alpha = int(np.argmin(alpha_nested_uniform)) + 1
    dom = np.argsort(alpha_uniform_per_c)[::-1][:5]

    # ACE2's one documented carve-out, applied to whichever channel dominates HERE:
    # "we downweighted the contribution of q0 to the calculation of alpha by a factor
    # of 10, since our poor skill in predicting the time-mean of this variable
    # otherwise dominated alpha".  Reported as a sensitivity, never as the headline --
    # if the verdict flips when one channel is quietened, that is the finding.
    cw_dw = np.ones(C)
    cw_dw[dom[0]] = 0.1
    alpha_uniform_dw, _ = _alpha(uniform_full, cw_dw)
    alpha_det_dw, _ = _alpha(_prefix(uniform_full, 1), cw_dw)
    dom_share = float(alpha_uniform_per_c[dom[0]] / (alpha_uniform_per_c.sum()))

    verdict = {
        "beats_shallowest_member": bool(med["rmse_weighted"] < med["rmse_shallowest"]),
        "weighting_beats_uniform": bool(med["rmse_weighted"] < med["rmse_uniform"]),
        "crps_beats_deterministic": bool(med["crps_ensemble"] < med["crps_shallowest"]),
        "median_rmse_gain_vs_shallowest_pct": float(np.median(gain_vs_shallow) * 100),
        "median_rmse_gain_vs_uniform_pct": float(np.median(gain_vs_uniform) * 100),
        "median_crps_gain_pct": float(np.median(crps_gain) * 100),
        "channels_improved_vs_shallowest": int((gain_vs_shallow > 0).sum()),
        "n_channels": C,
        "median_ssr": med["ssr"],
        "median_ssr_raw": med["ssr_raw"],
        "ssr_undefined_cells": n_ssr_undefined,
        "member_depths": [int(m.depth_steps) for m in table[targets[0]]],
        "truth_max_disagreement": truth_disagreement,
        # --- ACE2 eq 8, the paper-aligned framework ---
        "ace2_alpha_uniform": alpha_uniform,
        "ace2_alpha_sigma_weighted": alpha_sigma,
        "ace2_alpha_deterministic": alpha_det,
        "ace2_alpha_gain_vs_deterministic_pct": float((1.0 - alpha_uniform / alpha_det) * 100),
        "ace2_alpha_beats_deterministic": bool(alpha_uniform < alpha_det),
        "ace2_uniform_beats_sigma_weighted": bool(alpha_uniform < alpha_sigma),
        "ace2_alpha_nested_uniform": alpha_nested_uniform,
        "ace2_alpha_nested_sigma_weighted": alpha_nested_sigma,
        "ace2_best_n_members": best_m_alpha,
        "ace2_dominant_channels": [[channels[i], float(alpha_uniform_per_c[i])] for i in dom],
        "ace2_dominant_channel_share": dom_share,
        "ace2_alpha_uniform_downweighted": alpha_uniform_dw,
        "ace2_alpha_deterministic_downweighted": alpha_det_dw,
        "ace2_alpha_gain_downweighted_pct": float((1.0 - alpha_uniform_dw / alpha_det_dw) * 100),
        "ace2_verdict_survives_downweighting": bool(
            (alpha_uniform < alpha_det) == (alpha_uniform_dw < alpha_det_dw)),
        "ace2_n_time_samples": nt,
    }

    # Which nested sub-ensemble is best, per channel, then the modal choice.
    nested_med = np.median(nested_rmse.mean(axis=0), axis=1)       # (E,) over channels
    best_m = int(np.argmin(nested_med)) + 1
    per_channel_best = nested_rmse.mean(axis=0).argmin(axis=0) + 1
    verdict.update(
        best_n_members=best_m,
        best_n_members_depth_h=int(verdict["member_depths"][best_m - 1] * 6),
        nested_median_rmse=[float(x) for x in nested_med],
        channels_preferring_single_member=int((per_channel_best == 1).sum()),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import h5py
    with h5py.File(args.out_dir / "lagged_metrics.h5", "w") as f:
        for k, v in acc.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        f.create_dataset("rank_histogram", data=rank_hist)
        f.create_dataset("nested_rmse", data=nested_rmse, compression="gzip",
                         compression_opts=4)
        f.create_dataset("ace2_alpha_per_channel", data=alpha_uniform_per_c)
        f.create_dataset("ace2_alpha_nested_uniform", data=np.asarray(alpha_nested_uniform))
        f.create_dataset("ace2_alpha_nested_sigma", data=np.asarray(alpha_nested_sigma))
        f.create_dataset("ace2_norm_sigma", data=norm_c)
        # The eq-8 bias map itself: the per-channel field whose area-weighted RMS is
        # alpha. Kept because "which channel, and where" is the actionable part and it
        # cannot be recovered from the scalars.
        f.create_dataset("ace2_bias_uniform", compression="gzip", compression_opts=4,
                         data=S.time_mean_bias(
                             np.einsum("ec,echw->chw", uniform_full, member_sum),
                             truth_sum, nt))
        f.create_dataset("target", data=np.asarray(targets, dtype=np.int64))
        f.create_dataset("depth", data=np.asarray([m.depth_steps for m in table[targets[0]]]))
        f.create_dataset("channel", data=np.array(channels, dtype=h5py.string_dtype()))
        f.attrs.update(n_targets=len(targets), n_members=E, provenance=args.provenance,
                       nc_dir=str(args.nc_dir), sigma_h5=str(args.sigma_h5))

    with (args.out_dir / "lagged_summary.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["channel", "metric", "mean_over_targets"])
        for k, v in mean_over_t.items():
            for c, name in enumerate(channels):
                wr.writerow([name, k, f"{v[c]:.8g}"])

    (args.out_dir / "lagged_readout.json").write_text(json.dumps(
        {"verdict": verdict, "medians": med, "coverage": rep,
         "provenance": args.provenance}, indent=2, sort_keys=True, default=str) + "\n")

    print(f"targets {len(targets)}  members {E}  channels {C}  "
          f"({(time.time() - t0) / 60:.1f} min)")
    print(f"truth disagreement across members: {truth_disagreement:.3e} "
          "(must be 0 -- all members of a target predict the same valid time)")
    print(f"median RMSE   weighted {med['rmse_weighted']:.5g}  uniform "
          f"{med['rmse_uniform']:.5g}  shallowest member {med['rmse_shallowest']:.5g}")
    print(f"  vs shallowest: {verdict['median_rmse_gain_vs_shallowest_pct']:+.2f}%  "
          f"({verdict['channels_improved_vs_shallowest']}/{C} channels improved)")
    print(f"  vs uniform:    {verdict['median_rmse_gain_vs_uniform_pct']:+.2f}%")
    print(f"median CRPS   ensemble {med['crps_ensemble']:.5g}  shallowest "
          f"{med['crps_shallowest']:.5g}  ({verdict['median_crps_gain_pct']:+.2f}%)")
    print(f"median SSR    {med['ssr']:.4f}  raw {med['ssr_raw']:.4f}  "
          f"(1 = calibrated, <1 = over-confident; {n_ssr_undefined} (channel,target) "
          "cells undefined where the fair correction went non-positive)")
    print(f"member depths {verdict['member_depths'][0]}..{verdict['member_depths'][-1]} "
          f"({verdict['member_depths'][0] * 6}..{verdict['member_depths'][-1] * 6} h lead)")
    print("nested sub-ensembles (median RMSE over channels, m freshest members):")
    print("  m:    " + "  ".join(f"{m + 1:7d}" for m in range(E)))
    print("  rmse: " + "  ".join(f"{x:7.4f}" for x in nested_med))
    print(f"  best m = {best_m} (deepest member {verdict['best_n_members_depth_h']} h); "
          f"{verdict['channels_preferring_single_member']}/{C} channels prefer m=1")
    print(f"ACE2 eq 8 (arXiv:2411.11268 §4.3) -- RMS of the time+ensemble-mean error, "
          f"normalised, channel-MEAN over {C}; time average is {nt} targets:")
    print(f"  alpha  uniform(ACE2) {alpha_uniform:.5g}   1/sigma^2-weighted "
          f"{alpha_sigma:.5g}   single freshest member {alpha_det:.5g}")
    print(f"  vs deterministic: {verdict['ace2_alpha_gain_vs_deterministic_pct']:+.2f}%"
          f"   (lower alpha is better)")
    print("  alpha by m (uniform): " + "  ".join(f"{x:.4f}" for x in alpha_nested_uniform))
    print(f"  best m = {best_m_alpha};  dominant channels: " +
          ", ".join(f"{channels[i]} {alpha_uniform_per_c[i]:.3f}" for i in dom[:3]))
    print(f"  sensitivity, ACE2's q0 rule applied to {channels[dom[0]]} "
          f"({dom_share * 100:.0f}% of alpha) at 0.1x: ensemble {alpha_uniform_dw:.5g} vs "
          f"deterministic {alpha_det_dw:.5g} ({verdict['ace2_alpha_gain_downweighted_pct']:+.2f}%"
          f", verdict {'holds' if verdict['ace2_verdict_survives_downweighting'] else 'FLIPS'})")
    print(f"wrote {args.out_dir}/lagged_metrics.h5, lagged_summary.csv, lagged_readout.json")

    if truth_disagreement > 0:
        print(f"ERROR LAGGED_TRUTH_MISMATCH: members of a target disagree about truth by "
              f"{truth_disagreement:.3e}; the alignment put different valid times in one "
              "bucket and every number above is meaningless", file=sys.stderr)
        return 3
    print(f"LAGGED_ENSEMBLE_OK targets={len(targets)} members={E} "
          f"gain_vs_shallowest={verdict['median_rmse_gain_vs_shallowest_pct']:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
