#!/usr/bin/env python
"""Audit a makani E3SM pack's normalization against the corrected reference moments.

WHY THIS EXISTS.  The 2026-08-04 SST/fill audit found the normalization defect
class lives only where an externally-precomputed stats file can disagree with a
separately-configured fill -- PanguWeather and ai-rossby, whose stats were
regenerated (mask-aware moments, 2026-08-05:
``$PANGU_AUX/moments_2015-2050.json`` + ``data_2015-2050_{mean,std_corr}.nc``,
old ones preserved under ``pre_fix/``).  makani computes its stats in-stream
from the packed data, so it cannot have that bug -- but "immune to the bug" and
"numerically consistent with the corrected reference" are different claims.
This script proves the second one, per channel, so it can be re-run whenever
either side regenerates (new source years, new fill convention, new pack).

WHAT IT COMPARES.  Every channel makani trains on (52 state + PRECT) against
the corrected Pangu moments derived from the same source archive:
surface/diagnostic directly; upper-air by mapping makani's 10 plev channels to
the LAST 10 of the reference's 18 hybrid levels (~200..1000 hPa -- the same
source datasets the converter packed).  The moments JSON stores
``[count_total, count_valid, sum(x-shift), sum((x-shift)^2)]`` per (var, level)
with a per-var ``shift``; mean/std reconstruct exactly from those.

READING THE OUTPUT.  ``std`` agreement is the claim that matters (std is what
scales the training signal); expect a few % from the pack's shorter year
window.  Near-zero-mean channels (V*, U1000) show huge *percent* mean
deviations that are physically negligible -- the verdict line therefore scores
mean error in UNITS OF STD, not percent.  Fill-affected channels (SST, ICE --
makani forcings; TSOI/SOILWATER -- absent from makani) are intentionally out of
scope: makani's forcing normalization is self-consistent by construction and
SST/ICE are inputs only, never scored.

Verified 2026-08-26 on the 2-year scaling pack: all 53 stds within 2.14%,
all means within 0.02 std.  Run under the SFNO venv (needs numpy only):
    $SFNO_VENV/bin/python audit_makani_pack_vs_reference.py \
        --pack $MEMBER_ROOT/data/e3sm_makani_scaling \
        --moments $MEMBER_ROOT/../mehta5/pangu_polaris_data/moments_2015-2050.json
Prints ``MAKANI_PACK_AUDIT_OK`` or ``ERROR MAKANI_PACK_AUDIT_FAILED``.
"""

import argparse
import json
import sys

import numpy as np

PLEVS = [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
BASE = {"T": "T", "U": "U", "V": "V", "RH": "RELHUM", "Z": "Z3"}
# Makani's 10 plev channels are the last 10 of the reference's 18 hybrid levels.
LVL_IDX = list(range(8, 18))
STD_TOL_PCT = 5.0    # sampling-window drift allowance (2 vs 35 years measured 2.14%)
MEAN_TOL_STD = 0.05  # mean agreement in units of the channel std


def ref_stats(moments, shifts, key, row=None):
    rows = np.array(moments[key], dtype=float)
    if rows.ndim == 1:
        rows = rows[None, :]
    _tot, valid, s, sq = rows[row if row is not None else 0]
    mean = s / valid + shifts.get(key, 0.0)
    return mean, np.sqrt(sq / valid - (s / valid) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="makani pack root (has stats/)")
    ap.add_argument("--moments", required=True, help="corrected moments JSON")
    args = ap.parse_args()

    m = json.load(open(args.moments))
    moments, shifts = m["moments"], m["shifts"]
    means = np.load(args.pack + "/stats/global_means.npy").reshape(-1)
    stds = np.load(args.pack + "/stats/global_stds.npy").reshape(-1)

    if means.shape[0] == 53:
        # Locked PlaSim-shaped pack: nominal-hPa channel names mapping onto the
        # LAST 10 of the reference's 18 hybrid levels.
        names = ["PS", "TREFHT"] + [
            f"{v}{p}" for v in ("T", "U", "V", "RH", "Z") for p in PLEVS
        ] + ["PRECT"]
        skip_gate = set()
    elif means.shape[0] == 101:
        # ALLDATA pack: Pangu-parity contract, all 18 levels by index, surface
        # names identical to the reference's own keys. Two channels are
        # EXPECTED to disagree and are reported but not gated: the reference's
        # moments for TSOI_10CM/SOILWATER_10CM are MASK-AWARE (land-only) while
        # the pack's stats are filled-field (ocean filled 270.0 / 0.0) -- a
        # convention difference, not a defect (the pack's stats match its own
        # fills by construction, which is the property that matters).
        names = ["PS", "TREFHT", "U10", "RHREFHT", "PSL", "TMQ", "FSNT",
                 "FSNTOA", "SOILWATER_10CM", "TSOI_10CM"] + [
            f"{v}_l{i:02d}" for v in ("T", "U", "V", "Z3", "RELHUM")
            for i in range(18)
        ] + ["PRECT"]
        skip_gate = {"SOILWATER_10CM", "TSOI_10CM"}
    else:
        print(f"ERROR MAKANI_PACK_AUDIT_FAILED: pack has {means.shape[0]} channels, "
              f"expected 53 or 101 -- contract drifted, fix the audit or the pack")
        return 1

    bad = []
    print(f"{'channel':16s} {'pack std':>11s} {'ref std':>11s} {'d_std%':>7s} {'d_mean/std':>10s}")
    for i, n in enumerate(names):
        # Order matters: direct-name lookup FIRST, because surface names like
        # U10 end in digits and would otherwise be misparsed as nominal-hPa
        # upper-air channels (the bug that crashed the first alldata audit).
        if n in moments:
            rm, rs = ref_stats(moments, shifts, n)
        elif "_l" in n:
            v, idx = n.rsplit("_l", 1)
            rm, rs = ref_stats(moments, shifts, v, int(idx))
        else:
            v = "".join(c for c in n if c.isalpha())
            rm, rs = ref_stats(moments, shifts, BASE[v], LVL_IDX[PLEVS.index(int(n[len(v):]))])
        d_std = 100.0 * (stds[i] - rs) / rs
        d_mean = (means[i] - rm) / rs
        flag = ""
        if n in skip_gate:
            flag = "  (mask-aware ref vs filled pack: informational, not gated)"
        elif abs(d_std) > STD_TOL_PCT or abs(d_mean) > MEAN_TOL_STD:
            flag = " <<< OUT OF TOLERANCE"
            bad.append(n)
        print(f"{n:16s} {stds[i]:11.5g} {rs:11.5g} {d_std:7.2f} {d_mean:10.4f}{flag}")

    if bad:
        print(f"ERROR MAKANI_PACK_AUDIT_FAILED: {len(bad)} channels out of tolerance: {bad}")
        return 1
    print(f"MAKANI_PACK_AUDIT_OK {len(names)} channels within "
          f"{STD_TOL_PCT}% std / {MEAN_TOL_STD} std-units mean of the corrected reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
