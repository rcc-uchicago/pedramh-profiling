"""score_warp.py <warp_probe.json> [<round-1 warp_probe.json>] -- score PREDICTIONS_warp.md mechanically (stdlib):
P1-P14 on the given run; P15-P20 (the round-2 addendum) when the run has round 2's configs (P20 needs round 1's json).

Reading notes fixed before the run's numbers were seen (wording, not thresholds):
  P6's "51 % (Part X)" is a fraction of the A100-40GB DRAM peak (1,555 GB/s); its "35-65 %" is of BW_copy. Both
  clauses are scored with their own reference; the measured torch fft(dim=0) is printed next to them.
  P7's "strips" = the two shared-memory strip kernels (strided, tstore); tview loads through a strided view.
"""
import json
import sys

PEAK = 1555.0                                                        # GB/s, A100-SXM4-40GB HBM2


def main():
    R = json.load(open(sys.argv[1]))
    P = R["phases"]
    rows, cols, fused, acc, br, cu = (P.get(k, {}) for k in ("rows", "cols", "fused", "accuracy", "bridge", "cufft"))
    proj = P.get("projection", {})
    out = []

    def say(n, verdict, detail, *needed):
        """needed: the measured values the prediction reads; any None -> untested (the probe did not produce it)."""
        out.append((n, "untested" if any(v is None for v in needed) else verdict, detail))

    def best(t, prefix, key="GBs"):
        c = [(v[key], k) for k, v in t.items() if k.startswith(prefix) and isinstance(v, dict) and key in v]
        return max(c) if c else (None, None)

    env = P.get("env", {})
    r256 = rows.get("row_f32_R1_B256", {})
    say(1, "confirmed" if env.get("mathdx") == 1 and "med_us" in r256 else "refuted",
        f"mathdx={env.get('mathdx')}, row_f32_R1_B256 {'ran' if 'med_us' in r256 else r256.get('error', 'missing')}")
    say(2, "confirmed" if env and not any(e["where"] == "ENV" for e in R.get("errors", [])) else "refuted",
        f"torch {env.get('torch')}, jax {env.get('jax')}, warp {env.get('warp')}")
    j, s = br.get("jit_rel_l2"), br.get("scan_rel_l2")
    say(3, "confirmed" if j is not None and s is not None and j <= 1e-6 and s <= 1e-5 else "refuted",
        f"jit {j}, scan {s}, callbacks/iter {br.get('tiny_callbacks_per_iter')}", j, s)
    o = br.get("overhead_us_per_call")
    say(4, "confirmed" if o is not None and 5 <= o <= 40 else "refuted", f"{o} us per call", o)
    bw, k = best(rows, "row_f32_R")
    fr = rows[k]["frac_copy"] if k else None
    tr = cu.get("torch_row_c64", {}).get("med_us")
    ratio = rows[k]["med_us"] / tr if k and tr else None
    c1, c2 = fr is not None and 0.70 <= fr <= 0.90, ratio is not None and abs(ratio - 1) <= 0.15
    say(5, "confirmed" if c1 and c2 else ("half" if c1 or c2 else "refuted"),
        f"{k}: {fr} of copy, {ratio} x torch row")
    bw, k = best(cols, "col_f32")
    fr = cols[k]["frac_copy"] if k else None
    fp = bw / PEAK if bw else None
    tc = cu.get("torch_col_c64", {})
    c1, c2 = fr is not None and 0.35 <= fr <= 0.65, fp is not None and fp <= 0.61
    say(6, "confirmed" if c1 and c2 else ("half" if c1 or c2 else "refuted"),
        f"{k}: {fr} of copy, {fp} of DRAM peak; torch fft(dim=0) {tc.get('GBs')} GB/s")
    k4 = [cols.get(n, {}) for n in ("col_f32_strided_K4_B512", "col_f32_tstore_K4_B512")]
    nfail = sum("error" in x for x in k4)
    say(7, "confirmed" if nfail == 2 else ("half" if nfail == 1 else "refuted"),
        f"{nfail}/2 K=4 strips failed: " + "; ".join(x.get("error", "ran")[:120] for x in k4))
    bw, k = best(fused, "fused3_f32")
    fr = fused[k]["frac_copy"] if k else None
    say(8, "confirmed" if fr is not None and fr >= 0.60 else "refuted", f"{k}: {fr} of copy", fr)
    bw, kr = best(rows, "row_f32_R")
    e_row, e2d = acc.get(kr), [acc.get("warp_2d_row_colstrided_c64"), acc.get("warp_2d_nvidia_3pass_c64")]
    e2d = [x for x in e2d if x is not None]
    c1 = e_row is not None and e_row <= 2 * acc.get("torch_row_c64", 0)
    c2 = bool(e2d) and min(e2d) <= 2 * acc.get("torch_fft2_c64", 0)
    say(9, "confirmed" if c1 and c2 else ("half" if c1 or c2 else "refuted"),
        f"row {e_row} vs torch {acc.get('torch_row_c64')}; 2-D {e2d} vs torch {acc.get('torch_fft2_c64')}")
    bw, k = best(rows, "row_f64")
    fr = rows[k]["frac_copy"] if k else None
    tr = cu.get("torch_row_c128", {}).get("med_us")
    ratio = rows[k]["med_us"] / tr if k and tr else None
    c1, c2 = fr is not None and fr >= 0.50, ratio is not None and abs(ratio - 1) <= 0.25
    say(10, "confirmed" if c1 and c2 else ("half" if c1 or c2 else "refuted"),
        f"{k}: {fr} of c128 copy, {ratio} x torch c128 row")
    fb, m = rows.get("fallback_row_f32_R1_B256", {}).get("med_us"), r256.get("med_us")
    say(11, "confirmed" if fb and m and fb >= 1.5 * m else "refuted", f"fallback {fb} us vs cuFFTDx {m} us")
    jf = cu.get("jax_fft2_c64", {}).get("med_us")
    say(12, "confirmed" if jf is not None and 460 <= jf <= 620 else "refuted", f"{jf} us")
    t32, t64 = proj.get("t_step_ms_f32"), proj.get("t_step_ms_f64")
    say(13, "confirmed" if t32 is not None and 0.9 <= t32 <= 1.6 else "refuted", f"{t32} ms", t32)
    say(14, "confirmed" if t64 is not None and 1.8 <= t64 <= 3.5 else "refuted", f"{t64} ms", t64)
    if "rowT_f32_R4_B512" in rows:                                   # round 2's configs
        say(15, "confirmed" if br.get("BRIDGE_OK") else "refuted", f"BRIDGE_OK {br.get('BRIDGE_OK')}")
        cs = cols.get("col_f32_strided_K1_B256", {}).get("med_us")
        ct = cols.get("col_f32_tstore_K1_B256", {}).get("med_us")
        say(16, "confirmed" if cs and ct and cs <= 1.15 * ct else "refuted", f"strided {cs} us vs tstore {ct} us", cs, ct)
        tv = {n: v for n, v in cols.items() if n.startswith("col_f32_tview") and isinstance(v, dict)}
        k12 = [tv.get(n, {}) for n in ("col_f32_tview_K1_B256", "col_f32_tview_K2_B256")]
        launched = all("med_us" in x for x in k12)
        btv = max([v["frac_copy"] for v in tv.values() if "frac_copy" in v], default=None)
        say(17, "confirmed" if launched and btv is not None and btv >= 0.35 else ("half" if launched else "refuted"),
            f"K<=2 launched {launched}; best tview {btv} of copy")
        bw, k = best(fused, "fused3_f32")
        fr = fused[k]["frac_copy"] if k else None
        say(18, "confirmed" if fr is not None and fr >= 0.60 else "refuted", f"{k}: {fr} of copy", fr)
        fr = rows.get("rowT_f32_R4_B512", {}).get("frac_copy")
        say(19, "confirmed" if fr is not None and fr >= 0.50 else "refuted", f"rowT R4 {fr} of copy", fr)
        if len(sys.argv) > 2:
            R1 = json.load(open(sys.argv[2]))["phases"]
            pairs = [(rows.get("row_f32_R1_B256", {}).get("med_us"), R1["rows"].get("row_f32_R1_B256", {}).get("med_us")),
                     (cols.get("col_f32_tstore_K1_B256", {}).get("med_us"),
                      R1["cols"].get("col_f32_tstore_K1_B256", {}).get("med_us"))]
            okr = all(a and b and abs(a / b - 1) <= 0.05 for a, b in pairs)
            say(20, "confirmed" if okr else "refuted", f"round 2 / round 1: {[a / b if a and b else None for a, b in pairs]}")
    for n, v, d in out:
        print(f"P{n:<2d} {v:9s} {d}")
    tally = {v: sum(1 for _, x, _ in out if x == v) for v in ("confirmed", "half", "refuted", "untested")}
    print("WARP_SCORE " + " ".join(f"{k}={v}" for k, v in tally.items())
          + f" | verdict {proj.get('verdict', {}).get('f32')} / {proj.get('verdict', {}).get('f64')}")


if __name__ == "__main__":
    main()
