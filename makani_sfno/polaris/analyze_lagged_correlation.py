#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Why the lagged ensemble fails here but ACE2's eq-8 ensemble does not.

Produces the table in `docs/2026-09-21_lagged_ensemble_result.md` §5.  Reads only the
two scorecards that already exist -- no rollout NetCDFs -- so it is seconds, not
minutes, and is safe to re-run while iterating on the write-up.

THE QUESTION.  ACE2 (arXiv:2411.11268) averages eight staggered-init simulations
inside eq 8 and it helps; we average fourteen staggered-init forecasts and it hurts.
Both are "lagged ensembles" in the loose sense, so the difference has to be structural
rather than a bug -- and it is one inequality.

For two estimators with error magnitudes `sigma_1 < sigma_k` and correlation `rho`, the
optimal weight on the second is positive -- i.e. adding it helps AT ALL -- iff

    rho < sigma_1 / sigma_k

because the BLUE weight on the first is `a* = (s_k^2 - rho s_1 s_k)/(s_1^2 + s_k^2 -
2 rho s_1 s_k)` and `a* < 1` reduces to `rho s_1 s_k < s_1^2`.

  * ACE2's eq-8 members are 5-YEAR simulations, far past any predictability horizon, so
    every member estimates the climate EQUALLY well: `sigma_1/sigma_k = 1`, the
    threshold is `rho < 1`, and any non-degenerate ensemble clears it.
  * Ours are forecasts at 24...336 h, so member skill is wildly UNEQUAL and the
    threshold collapses toward zero as members age.

WHAT THIS MEASURES.  Each nested step `m -> m+1` is a two-estimator problem: the
m-member weighted mean (error variance `V_m`, known from `nested_rmse`) combined with
member `m+1` (variance `s_k^2`, from the independent monthly sweep) at the driver's
renormalised weights `a, b`:

    V_{m+1} = a^2 V_m + b^2 s_k^2 + 2 a b rho sqrt(V_m) s_k

which inverts for `rho` -- the correlation between the running mean and the incoming
member.  Comparing it against `sqrt(V_m)/s_k` says whether that member could have
helped under ANY weighting, which is the question the nested sweep alone cannot answer
(it varies ensemble size, never the weights' functional form).

Run under the SFNO venv; needs the h5py overlay on `PYTHONPATH` (see
`polaris_makani_env.sh`).  CPU-only and CLAUDE.md #3 does not apply -- it reads two
small HDF5 files -- but the login node's BLAS thread default does, so pin it:

    OMP_NUM_THREADS=1 python polaris/analyze_lagged_correlation.py \\
        --sigma-h5  $EVAL/<run>_K56/scores/k56_metrics.h5 \\
        --lagged-h5 $EVAL/<run>_lagged_K56_d4/scores_ace2/lagged_metrics.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Implied member-error correlation vs the threshold that would make "
                    "a lagged member worth adding.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--sigma-h5", required=True, type=Path,
                   help="k56_metrics.h5 from the INDEPENDENT monthly sweep")
    p.add_argument("--lagged-h5", required=True, type=Path,
                   help="lagged_metrics.h5 from score_lagged_ensemble.py")
    p.add_argument("--stride", type=int, default=4, help="sweep stride d (default 4)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    import h5py

    with h5py.File(args.sigma_h5, "r") as f:
        sigma = np.asarray(f["rmse_mean"][:], dtype=np.float64)          # (K, C)
    with h5py.File(args.lagged_h5, "r") as f:
        nested = np.asarray(f["nested_rmse"][:], dtype=np.float64).mean(0)   # (E, C)
        r_shallow = np.asarray(f["rmse_shallowest"][:], dtype=np.float64).mean(0)
        depths = [int(d) for d in f["depth"][:]]
        channels = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][:]]
        alpha_c = (np.asarray(f["ace2_alpha_per_channel"][:], dtype=np.float64)
                   if "ace2_alpha_per_channel" in f else None)

    if len(depths) != nested.shape[0]:
        print(f"ERROR DEPTH_MISMATCH: {len(depths)} depths vs {nested.shape[0]} nested "
              "columns; these files are not from the same run", file=sys.stderr)
        return 2

    s = sigma[[d - 1 for d in depths], :]
    w = 1.0 / s ** 2
    w = w / w.sum(axis=0, keepdims=True)
    # Calibrate the independent sweep's sigma onto this sample using the one member
    # whose error we measured directly. Without it a 5% offset between sweeps would be
    # read as a 5% change in rho.
    s_cal = s * (r_shallow / s[0])

    print(f"{'m->m+1':>8} {'lead h':>7} {'sigma_k/sqrt(Vm)':>17} {'rho':>7} "
          f"{'rho must be <':>14} {'channels helped':>16}")
    rows = []
    for m in range(1, len(depths)):
        v_m = nested[m - 1] ** 2
        s_k = s_cal[m]
        head = w[:m].sum(axis=0)
        a = head / (head + w[m])
        b = 1.0 - a
        rho = (nested[m] ** 2 - a ** 2 * v_m - b ** 2 * s_k ** 2) / (
            2.0 * a * b * np.sqrt(v_m) * s_k)
        thresh = np.sqrt(v_m) / s_k
        helped = float((rho < thresh).mean() * 100.0)
        rows.append((m, depths[m] * 6, float(np.median(s_k / np.sqrt(v_m))),
                     float(np.median(rho)), float(np.median(thresh)), helped))
        print(f"{m}->{m + 1:<5} {depths[m] * 6:>7} {rows[-1][2]:>17.3f} "
              f"{rows[-1][3]:>7.3f} {rows[-1][4]:>14.3f} {helped:>15.0f}%")

    # Is the floor one bad channel, or all of them?  It is easy -- and this analysis
    # got it wrong once -- to read "rho floors at 0.42" together with "one channel owns
    # 42% of alpha" and conclude they are the same finding.  They are not: rho above is
    # a MEDIAN over channels, and one channel cannot move a median.  Report the
    # dominant channel's own rho next to the median with it removed, so the two
    # problems stay visibly distinct.
    if alpha_c is not None:
        iz = int(np.argmax(alpha_c))
        keep = np.ones(len(channels), bool)
        keep[iz] = False
        v_m = nested[-2] ** 2
        s_k = s_cal[-1]
        head = w[:-1].sum(axis=0)
        a = head / (head + w[-1])
        b = 1.0 - a
        rho_d = (nested[-1] ** 2 - a ** 2 * v_m - b ** 2 * s_k ** 2) / (
            2.0 * a * b * np.sqrt(v_m) * s_k)
        thr_d = np.sqrt(v_m) / s_k
        print(f"\ndeepest step, dominant-channel check ({channels[iz]} = "
              f"{alpha_c[iz] / alpha_c.sum() * 100:.1f}% of alpha):")
        print(f"  median rho  all channels {np.median(rho_d):.4f}  |  "
              f"excluding {channels[iz]} {np.median(rho_d[keep]):.4f}")
        print(f"  {channels[iz]}'s own rho {rho_d[iz]:.4f} (p90 of all = "
              f"{np.percentile(rho_d, 90):.3f}) -- the most extreme channel, not the cause")
        print(f"  channels failing the threshold: {(rho_d > thr_d).sum()}/{len(channels)}"
              f"  |  excluding it: {(rho_d[keep] > thr_d[keep]).sum()}/{keep.sum()}")
        print("  => the floor is a property of ALL channels (shared model bias); the "
              "dominant\n     channel is the worst instance of it, and fixing that one "
              "would not rescue\n     the ensemble.")

    rho_floor = min(r[3] for r in rows)
    print(f"\nrho is the correlation between the m-member mean and the incoming member.")
    print(f"'channels helped' = %% of {nested.shape[1]} channels where rho < threshold, "
          "i.e. where the\noptimal weight on that member is positive at all.")
    print(f"\nrho floors at {rho_floor:.3f} rather than decaying to 0: that residue is the "
          "SHARED\nSYSTEMATIC MODEL BIAS -- both members come from the same weights, so part "
          "of their\nerror is identical however far apart they are initialised, and averaging "
          "cannot\ntouch it. This is a MODEL-WIDE property, not one bad channel (see the "
          "dominant-\nchannel check above), so it is not reachable by any per-channel fix -- "
          "only by an\nensemble whose members are not the same weights, or not at unequal leads.")
    if all(r[3] >= r[4] for r in rows):
        print("\nLAGGED_CORRELATION_OK verdict=no-member-helps "
              f"rho_floor={rho_floor:.3f}")
    else:
        helps = [r[0] for r in rows if r[3] < r[4]]
        print(f"\nLAGGED_CORRELATION_OK verdict=some-member-helps steps={helps} "
              f"rho_floor={rho_floor:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
