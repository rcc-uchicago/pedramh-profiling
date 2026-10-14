"""Synthetic test of precision_cross_analysis.py: known effects in, the pre-registered outcome out.

Stdlib only; runs on a login node.  PASS = TEST_PRECISION_CROSS_OK
"""
import csv
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precision_cross_analysis as pca  # noqa: E402

GROUPS = ["G%02d" % i for i in range(24)]


def write_budget(path, arm_values):
    """Budget lines in dryair_budget_readout.py's format; a non-finite base = a truncated, blown member."""
    with open(path, "w") as fh:
        for arm, base in arm_values.items():
            for m in range(8):
                n = pca.N_STEPS0 - pca.STRIDE * m
                if not math.isfinite(base):
                    n //= 3
                fh.write("%s%02d  n= %d dPS=   -0.30 dDRY=   -0.10 g*dTMQ=   -0.20 hPa | TMQ gm  27.10->  25.00 "
                         "(min   24.00) kg/m2 | max|step dDRY|=0.100 hPa | monthly TMQ min=   -9.00 lastmonth neg=  0.2%%\n"
                         % (arm, m, n))


def write_channels(path, arm_values, rng):
    """arm_values: {arm: base A}; per-member offsets shared across arms (paired design) + small noise."""
    offsets = [rng.uniform(-0.3, 0.3) for _ in range(8)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "member", "channel", "group", "e", "rmse_sigma", "r", "steps"])
        for arm, base in arm_values.items():
            for m in range(8):
                for g in GROUPS:
                    r = base + offsets[m] + rng.gauss(0, 0.02) if math.isfinite(base) else 1e14
                    w.writerow([arm, "member_%s%02d.nc" % (arm, m), g + "_c", g, 0, 0, r, 7667])


def build(tmp, r_eff, t_eff, gate=True, blow=None, seed=0):
    rng = random.Random(seed)
    existing, new = {}, {}
    for e in pca.EPOCHS:
        base = 3.0 + 0.05 * (e - 15)
        existing["FSE%02d" % e] = base                         # bf16 train, bf16 rollout
        existing["G1E%02d" % e] = base + r_eff + t_eff         # fp32 train, fp32 rollout
        new["FSEF%02d" % e] = base + r_eff                     # bf16 train, fp32 rollout
        new["G1EB%02d" % e] = base + t_eff                     # fp32 train, bf16 rollout
    if blow:
        existing[blow] = float("inf")
    ex, nw, gt, bx, bn = (os.path.join(tmp, f) for f in ("ex.csv", "new.csv", "gate.txt", "bx.txt", "bn.txt"))
    write_channels(ex, existing, rng)
    write_channels(nw, new, rng)
    write_budget(bx, existing)
    write_budget(bn, new)
    with open(gt, "w") as fh:
        fh.write("AMP_BF16_EQUIV_OK members=8 steps=120\n" if gate else "ERROR AMP_BF16_NOT_BITWISE\n")
    return ex, nw, gt, [bx, bn]


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    return cond


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        devnull = open(os.devnull, "w")
        stdout, sys.stdout = sys.stdout, devnull
        try:
            o1, e1 = pca.run(*build(tmp, r_eff=-1.5, t_eff=0.0), out_path="")
            o2, _ = pca.run(*build(tmp, r_eff=0.0, t_eff=-1.5, seed=1), out_path="")
            o3, _ = pca.run(*build(tmp, r_eff=-0.8, t_eff=-0.8, seed=2), out_path="")
            o4, e4 = pca.run(*build(tmp, r_eff=-1.5, t_eff=0.0, gate=False, seed=3), out_path="")
            o5, e5 = pca.run(*build(tmp, r_eff=-1.5, t_eff=0.0, blow="FSE17", seed=4), out_path="")
        finally:
            sys.stdout = stdout
            devnull.close()
    ok &= check("rollout-only effect -> outcome 1", o1 == 1)
    ok &= check("training-only effect -> outcome 2", o2 == 2)
    ok &= check("both effects -> outcome 3", o3 == 3)
    ok &= check("gate missing -> outcome 4 and the G1EB cell void", o4 == 4 and e4["R|T=fp32"]["n"] == 0)
    ok &= check("Total = R|T=bf16 + T|R=fp32 (additive identity)",
                abs(e1["Total"]["mean"] - e1["R|T=bf16"]["mean"] - e1["T|R=fp32"]["mean"]) < 1e-9)
    ok &= check("recovered R|T=bf16 within 0.05 of -1.5", abs(e1["R|T=bf16"]["mean"] + 1.5) < 0.05)
    ok &= check("truncated member (huge finite A) excluded via the budget, not propagated", o5 == 1 and math.isfinite(e5["R|T=bf16"]["mean"])
                and 17 not in e5["R|T=bf16"]["per_epoch"])
    print("TEST_PRECISION_CROSS_OK" if ok else "ERROR TEST_PRECISION_CROSS_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
