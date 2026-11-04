"""Check the re-centred bias-map CIs against the pushed figures (figures_v2_uncentred/), stdlib only.

Re-centring subtracts one constant per panel from the same bootstrap draws, so it must leave:
  * every row that is not a bias_maps row bit-identical (same RNG stream, same numbers);
  * every bias_maps estimate, n and significant area bit-identical;
  * every bias_maps interval's width unchanged (relative 1e-9), shifted down (old - new > 0) and containing its
    estimate.
Usage: check_recentre.py [NEW_DIR] [OLD_DIR]      PASS = RECENTRE_CHECK_OK
"""
import csv
import os
import sys

M = os.path.dirname(os.path.abspath(__file__))
NEW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(M, "figures")
OLD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(M, "figures_v2_uncentred")


def rows(d):
    out = {}
    for r in csv.DictReader(open(os.path.join(d, "summary_table.csv"))):
        out[(r["figure"], r["variable"], r["model"], r["lead_h"], r["window"])] = r
    return out


def main():
    new, old = rows(NEW), rows(OLD)
    bad = []
    if set(new) != set(old):
        bad.append("row keys differ: %d new vs %d old" % (len(new), len(old)))
    other = [k for k in old if k[0] != "bias_maps"]
    for k in other:
        if new.get(k) != old[k]:
            bad.append("non-bias row changed: %s" % (k,))
    bias = sorted(k for k in old if k[0] == "bias_maps")
    shifts, wrel, old_out, old_edge = [], [], [], []
    for k in bias:
        n, o = new[k], old[k]
        for f in ("value", "n", "sig_area_pct", "unit"):
            if n[f] != o[f]:
                bad.append("bias %s changed: %s" % (f, k))
        v = float(o["value"])
        nlo, nhi, olo, ohi = (float(x) for x in (n["ci_lo"], n["ci_hi"], o["ci_lo"], o["ci_hi"]))
        wrel.append(abs((nhi - nlo) - (ohi - olo)) / (ohi - olo))
        s_lo, s_hi = olo - nlo, ohi - nhi
        if abs(s_lo - s_hi) > 1e-9 * max(abs(s_lo), 1e-30) + 1e-12 * (ohi - olo):
            bad.append("unequal shift %s: %g vs %g" % (k, s_lo, s_hi))
        shifts.append(s_lo / (ohi - olo))
        if s_lo <= 0:
            bad.append("shift not downward: %s" % (k,))
        if not nlo <= v <= nhi:
            bad.append("new CI excludes its estimate: %s" % (k,))
        if not olo <= v <= ohi:
            old_out.append(k)
        elif (v - olo) / (ohi - olo) < 0.1:
            old_edge.append(k)
    print("non-bias rows identical: %d/%d" % (len(other) - sum(1 for b in bad if b.startswith("non-bias")), len(other)))
    print("bias rows: %d; max relative width change %.2e; shift / old width %.3f-%.3f" %
          (len(bias), max(wrel), min(shifts), max(shifts)))
    print("old interval excluded its estimate: %d %s" % (len(old_out), ["%s/%s" % (k[1], k[2]) for k in old_out]))
    print("old estimate in the bottom tenth of its interval: %d %s" %
          (len(old_edge), ["%s/%s" % (k[1], k[2]) for k in old_edge]))
    for b in bad[:10]:
        print("ERROR " + b)
    if bad:
        print("ERROR RECENTRE_CHECK_FAILED n=%d" % len(bad))
        return 1
    print("RECENTRE_CHECK_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
