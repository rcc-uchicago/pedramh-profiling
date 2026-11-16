"""table_trgateD.py -- notes Part XVII H's gate D tables for tilerows_if25 (job 7843532), from the committed
dose_summary.json / history files. Standard library; python >= 3.6.

  python3 table_trgateD.py            # drift / overfit / absorption tables, best and last
  python3 table_trgateD.py --cross    # post hoc: alpha = 0 seed-1 arm of job 7835786 vs this job's arms

dose_summary.json's keys delta_1000 / delta_10000 hold the 860 / 8,600 arms in this job.
"""
import argparse
import json
import math
import os
import statistics

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
TR, XV = os.path.join(RES, "polaris_trgateD_7843532"), os.path.join(RES, "polaris_gateD_7835786")
LO, HI = 860.0, 8600.0
ARMS = ["a0_sA", "a0_sB", "a860_sA", "a8600_sA"]


def rng(v, f="{:+.3f}"):
    return f"{f.format(min(v))} to {f.format(max(v))}"


def summary(job, which):
    return json.load(open(os.path.join(job, which, "dose_summary.json")))


def tables():
    pat = json.load(open(os.path.join(TR, "bias_pattern_tilerows_if25.json")))
    asign = {f"k{a}-{b}": ("+" if x > 0 else "-") for (a, b), x in zip(pat["bands"], pat["alpha"])}
    for which in ("best", "last"):
        d = summary(TR, which)
        print(f"===== {which}: verdict {d['verdict']}")
        print("| quantity | R_0A | N(w) | D_860 | D_8600 | |D_8600|>N | |D_860|>N | max real | max real/(0.1N) |"
              " D_8600 +/- | D_8600/D_860 (median) | alpha_b |")
        tot_hi = tot_lo = cells = 0
        worst = (0.0, None)
        for q, ws in d["drift"].items():
            N = [w["noise"] for w in ws]
            dl = [w["delta_1000"] for w in ws]
            dh = [w["delta_10000"] for w in ws]
            re = [w["real_bias_effect"] for w in ws]
            rat = [r / (0.1 * n) for r, n in zip(re, N)]
            assert all(w["pass"] == (r < 0.1 * n) for w, r, n in zip(ws, re, N))
            assert all(math.isclose(r, abs(x) / LO, rel_tol=1e-9) for r, x in zip(re, dl))
            for wi, r in enumerate(rat):
                if r > worst[0]:
                    worst = (r, f"{q} {ws[wi]['window_TU'][0]:g}-{ws[wi]['window_TU'][1]:g} TU")
            nh = sum(abs(x) > n for x, n in zip(dh, N))
            nl = sum(abs(x) > n for x, n in zip(dl, N))
            tot_hi, tot_lo, cells = tot_hi + nh, tot_lo + nl, cells + len(ws)
            pos = sum(x > 0 for x in dh)
            lin = [h / l for h, l in zip(dh, dl) if l != 0]
            print(f"| {q} | {rng([w['R_ref'] for w in ws])} | {rng(N, '{:.3f}')} | {rng(dl)} | {rng(dh)} |"
                  f" {nh}/{len(ws)} | {nl}/{len(ws)} | {max(re):.1e} | {max(rat):.3f} | {pos}/{len(ws) - pos} |"
                  f" {statistics.median(lin):+.1f} | {asign.get(q, '')} |")
        print(f"cells |D_8600|>N {tot_hi}/{cells}; |D_860|>N {tot_lo}/{cells}; tightest {worst[1]}: "
              f"real/(0.1N) {worst[0]:.4f} = {1 / worst[0]:.0f}x margin")
        for k, v in d["overfit"].items():
            print(f"overfit {k}: val_min {v['val_min']:.6f} @ {v['val_min_step']}, final/min {v['final_over_min']:.4f}, "
                  f"train final {v['train_final']:.6f}, val/train {v['val_over_train_final']:.2f}")
        print("absorption", json.dumps(d["absorption"]))
        print("blown", {k: v["blown"] for k, v in d["stability"].items()}, "usable", d["usable_steps"], d["windows"])
    vs = {}
    for a in ARMS:
        h = json.load(open(os.path.join(TR, a + "_history.json")))["hist"]
        s = [x["s_step"] for x in h]
        vs[a] = h[-1]["val_clean"]
        print(f"{a}: {len(s)} val intervals, mean s/step {statistics.mean(s):.4f}")
    print(f"final val_clean vs a0_sA: seed {vs['a0_sB'] / vs['a0_sA'] - 1:+.4f}, "
          f"860 {vs['a860_sA'] / vs['a0_sA'] - 1:+.4f}, 8600 {vs['a8600_sA'] / vs['a0_sA'] - 1:+.4f}")


def cross():
    """Same frames, ICs, climatology and seed in both jobs; training is not bitwise reproducible."""
    med = statistics.median
    for which in ("best", "last"):
        x, y = summary(XV, which), summary(TR, which)
        assert x["climatology"]["band_mean"] == y["climatology"]["band_mean"]
        print(f"===== {which} (medians over windows)")
        print("| quantity | XV a0_sA - this a0_sA | this D_860 | this a860_sA - XV a0_sA | this a8600_sA - XV a0_sA |")
        for q in y["drift"]:
            X, Y = x["drift"][q], y["drift"][q]
            rx = [w["R_ref"] for w in X]
            print(f"| {q} | {med([a - w['R_ref'] for a, w in zip(rx, Y)]):+.3f} |"
                  f" {med([w['delta_1000'] for w in Y]):+.3f} |"
                  f" {med([w['R_ref'] + w['delta_1000'] - a for w, a in zip(Y, rx)]):+.3f} |"
                  f" {med([w['R_ref'] + w['delta_10000'] - a for w, a in zip(Y, rx)]):+.3f} |")
        ex, ey = x["drift"]["E_low"], y["drift"]["E_low"]
        print("E_low per window: " + " ".join(
            f"[{b['window_TU'][0]:g}] XV {a['R_ref']:+.3f} this {b['R_ref']:+.3f} N {b['noise']:.3f}"
            for a, b in zip(ex, ey)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", action="store_true")
    cross() if ap.parse_args().cross else tables()
