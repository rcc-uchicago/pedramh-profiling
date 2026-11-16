"""score_warpupd.py <results dir> -- score PREDICTIONS_warpupd.md's 11 predictions mechanically (stdlib).
Reads warpupd_probe.json and the ncu_<arm>_raw.csv.summary.json files in the dir."""
import glob
import json
import os
import sys

ARMS = ("xla_compact", "xla_compact_if", "warp_full", "warp_compact", "warp_compact_if", "warp_simt", "warp_simt_if")


def main():
    d = sys.argv[1]
    R = json.load(open(os.path.join(d, "warpupd_probe.json")))
    g, b, v = R.get("gates", {}), R.get("bench_ms_per_step", {}), R.get("verdict", {})
    ncu = {}
    for p in glob.glob(os.path.join(d, "ncu_*_raw.csv.summary.json")):
        ncu[os.path.basename(p)[4:-len("_raw.csv.summary.json")]] = json.load(open(p))
    out = []

    def say(n, ok, detail, *needed):
        out.append((n, "untested" if any(x is None for x in needed) else ("confirmed" if ok else "refuted"), detail))

    fin = [g.get(n, {}).get("finite", {}).get(a) for n in ("1", "40", "400") for a in ARMS]
    say(1, all(x is True for x in fin), f"finite all: {all(x is True for x in fin)}")
    say(2, all(all(x is True for x in v.get(a, [])) for a in ARMS), f"gates {v}")
    bw = lambda a: all(g.get(n, {}).get("bitwise_vs_base", {}).get(a) is True for n in ("1", "40", "400"))  # noqa: E731
    say(3, bw("xla_compact"), "xla_compact bitwise at 1/40/400")
    say(4, bw("warp_full"), "warp_full bitwise at 1/40/400")
    f, wf, xc, wc, ws = (b.get(k) for k in ("lean32ek_fast", "warp_full", "xla_compact", "warp_compact", "warp_simt"))
    say(5, wf is not None and f is not None and abs(wf / f - 1) <= 0.03, f"warp_full {wf} vs {f}", wf, f)
    say(6, xc is not None and f is not None and 0.15 <= f - xc <= 0.35, f"lean32ek_fast - xla_compact = "
        f"{(f - xc) if xc and f else None}", xc, f)
    say(7, wc is not None and xc is not None and abs(wc - xc) <= 0.05, f"warp_compact - xla_compact = "
        f"{(wc - xc) if wc and xc else None}", wc, xc)
    arms_t = {k: b[k] for k in ("lean32ek_fast",) + ARMS if k in b}
    fastest = min(arms_t, key=arms_t.get) if arms_t else None
    say(8, fastest in ("warp_simt", "warp_simt_if") and ws is not None and 2.2 <= ws <= 2.5,
        f"fastest {fastest} ({arms_t.get(fastest)}), warp_simt {ws}", ws)
    pairs = [("xla_compact_if", "xla_compact"), ("warp_compact_if", "warp_compact"), ("warp_simt_if", "warp_simt")]
    say(9, all(b.get(i) and b.get(a) and b[i] <= 1.02 * b[a] for i, a in pairs),
        "; ".join(f"{i}/{a} {b[i] / b[a]:.4f}" for i, a in pairs if b.get(i) and b.get(a)))
    upd = [e for arm in ("warp_compact", "warp_simt") for k, e in ncu.get(arm, {}).items()
           if "upd_compact" in k or "upd_split" in k]
    if upd:
        ok = all(e.get("ld_sec_per_req", 99) <= 8.5 and e.get("ld_use", 0) >= 85 for e in upd)
        say(10, ok, "; ".join(f"{e.get('ld_sec_per_req'):.2f} sec/req, {e.get('ld_use'):.1f}% used" for e in upd))
    else:
        say(10, False, "no ncu data", None)
    bwk = [e for k, e in ncu.get("warp_simt", {}).items() if "build_w" in k]
    say(11, bool(bwk) and bwk[0].get("ld_sec_per_req", 99) <= 10,
        f"build_w {bwk[0].get('ld_sec_per_req') if bwk else None} sec/req", bwk[0] if bwk else None)
    for n, vv, dd in out:
        print(f"P{n:<2d} {vv:9s} {dd}")
    tally = {x: sum(1 for _, y, _ in out if y == x) for x in ("confirmed", "refuted", "untested")}
    adm = [a for a in ARMS if a in b and all(x is True for x in v.get(a, [])) and a.startswith("warp")]
    best = min(adm, key=lambda a: b[a]) if adm else None
    worth = bool(best and b[best] <= 0.97 * f and b[best] <= 0.97 * xc)
    print("WU_SCORE " + " ".join(f"{k}={x}" for k, x in tally.items())
          + f" | best Warp arm {best} {b.get(best)} | Warp worth it (pre-set): {worth}")


if __name__ == "__main__":
    main()
