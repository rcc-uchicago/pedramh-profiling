"""Synthetic test of fp32_rerank.py's ranking rule (prereg 2026-10-13 fp32 re-roll). Stdlib only.
PASS = TEST_FP32_RERANK_OK
"""
import csv
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp32_rerank as fr  # noqa: E402
import precision_cross_analysis as pca  # noqa: E402

OFFSETS = [0.1 * (i - 3.5) for i in range(8)]          # shared per start date -> a paired design


def write_tag(d, tag, arms, rng):
    """arms: {arm: dict(A=, neg=, dps=, drift=)}"""
    with open(os.path.join(d, tag + "_channels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "member", "channel", "group", "e", "rmse_sigma", "r", "steps"])
        for arm, s in arms.items():
            for m in range(8):
                noise = rng.gauss(0, 0.02)                  # per-rollout noise, as in real paired data
                for g in range(24):
                    w.writerow([arm, "member_%s%02d.nc" % (arm, m), "c%d" % g, "G%02d" % g, 0, 0,
                                s["A"] + OFFSETS[m] + noise, 7667])
    with open(os.path.join(d, tag + "_budget.txt"), "w") as fh:
        for arm, s in arms.items():
            for m in range(8):
                fh.write("%s%02d  n= %d dPS= %7.2f dDRY=   -0.10 g*dTMQ=   -0.20 hPa | TMQ gm  27.10->  25.00 "
                         "(min   24.00) kg/m2 | max|step dDRY|=0.100 hPa | monthly TMQ min=   -9.00 lastmonth neg= %4.1f%%\n"
                         % (arm, m, pca.N_STEPS0 - pca.STRIDE * m, s.get("dps", -0.3), s.get("neg", 0.2)))
    with open(os.path.join(d, tag + "_census.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "members", "drifting_groups", "groups", "trefht_drift_K", "ps_drift_hPa"])
        for arm, s in arms.items():
            w.writerow([arm, 8, s.get("drift", 10), "", 0, 0])


def case(new_arms, seed=0):
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as d:
        write_tag(d, "g1dry_c", {"G1D39": dict(A=1.5, drift=14)}, rng)
        write_tag(d, "g1fs", {"FSE30": dict(A=3.0, dps=-0.5), "G1E15": dict(A=1.6, dps=-2.4)}, rng)
        write_tag(d, "t", new_arms, rng)
        devnull = open(os.devnull, "w")
        stdout, sys.stdout = sys.stdout, devnull
        try:
            return fr.run(d, "t", "")
        finally:
            sys.stdout = stdout
            devnull.close()


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    return cond


def main():
    ok = True
    w, out, tie, rows = case({"FSEF30": dict(A=1.0, drift=20)})
    ok &= check("clearly lower A -> wins outright", w == "FSEF30" and out)
    w, out, tie, rows = case({"FSEF30": dict(A=1.497, drift=20)})
    ok &= check("within 3.3 SE -> tie, broken by fewer drifting groups (incumbent keeps it)",
                w == "G1D39" and not out and set(tie) == {"FSEF30", "G1D39"})
    w, out, tie, rows = case({"FSEF30": dict(A=1.0, neg=5.0)})
    ok &= check("lowest A but fails the water gate -> excluded", w == "G1D39" and not rows["FSEF30"]["ok"])
    ok &= check("bf16 original not in the fp32 pool (FSE30 excluded, G1E15 kept but fails gate 3)",
                "FSE30" not in rows and "G1E15" in rows and not rows["G1E15"]["ok"])
    ok &= check("original() maps every prereg arm", [fr.original(a) for a in ("FSEF23", "FDEF32", "BF22", "BF24fix", "DRYAIRF24")]
                == ["FSE23", "FDE32", "B22", "B24dryair", "DRYAIR24"])
    print("TEST_FP32_RERANK_OK" if ok else "ERROR TEST_FP32_RERANK_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
