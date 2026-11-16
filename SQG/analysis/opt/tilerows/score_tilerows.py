"""score_tilerows.py <tilerows_probe.json> -- score PREDICTIONS_tilerows.md's 11 predictions mechanically (stdlib)."""
import json
import sys


def main():
    R = json.load(open(sys.argv[1]))
    gates, bench, ver = R.get("gates", {}), R.get("bench_ms_per_step", {}), R.get("verdict", {})
    out = []

    def say(n, ok, detail, *needed):
        out.append((n, "untested" if any(x is None for x in needed) else ("confirmed" if ok else "refuted"), detail))

    fin = gates.get("1", {}).get("rel", {}).get("finite", {})
    say(1, bool(fin) and all(fin.values()) and not R.get("errors"), f"finite {fin}; errors {len(R.get('errors', []))}")
    g32, gif = ver.get("G32"), ver.get("GIF")
    say(2, g32 is not None and all(x is True for x in g32), f"G32 {g32}", g32)
    say(3, gif is not None and all(x is True for x in gif), f"GIF {gif}", gif)
    b = gates.get("4000", {}).get("bands", {})
    if "tilerows32_vs_lean_u1" in b:
        r = [x / y for k in ("sub", "af") for x, y in zip(b["tilerows32_vs_lean_u1"][k], b["lean32ek_fast_vs_lean_u1"][k])]
        say(4, all(0.5 <= x <= 2 for x in r), f"ratio {min(r):.3f} .. {max(r):.3f}")
    else:
        say(4, False, "no bands", None)
    t, f = bench.get("tilerows32_eager"), bench.get("lean32ek_fast")
    say(5, t is not None and 1.4 <= t <= 2.2, f"tilerows32 {t} ms", t)
    say(6, t is not None and f is not None and t <= 0.97 * f, f"{t} vs lean32ek_fast {f} ({f / t if t else 0:.3f}x)", t, f)
    sp = bench.get("tilerows32_split_ms") or {}
    tr = sp.get("tile_rows")
    say(7, tr is not None and 0.30 <= tr <= 0.55, f"tile_rows {tr} ms", tr)
    cw, cz = sp.get("cufft_cols_w"), sp.get("cufft_cols_z")
    say(8, cw is not None and cz is not None and all(0.18 <= x <= 0.30 for x in (cw, cz)), f"cols {cw}, {cz} ms", cw, cz)
    c2r = sp.get("cufft_c2r")
    say(9, c2r is not None and 0.25 <= c2r <= 0.40, f"c2r {c2r} ms", c2r)
    g = bench.get("tilerows32_graph")
    say(10, g is not None and t is not None and abs(g / t - 1) <= 0.03, f"graph {g} vs {t}", g, t)
    ti = bench.get("tilerows32_if_eager")
    say(11, ti is not None and t is not None and ti <= 1.03 * t, f"if {ti} vs {t}", ti, t)
    for n, v, d in out:
        print(f"P{n:<2d} {v:9s} {d}")
    tally = {v: sum(1 for _, x, _ in out if x == v) for v in ("confirmed", "refuted", "untested")}
    worth = bool(g32 and all(x is True for x in g32) and t is not None and f is not None and t <= 0.97 * f)
    print("TR_SCORE " + " ".join(f"{k}={v}" for k, v in tally.items()) + f" | worth integrating (pre-set): {worth}")


if __name__ == "__main__":
    main()
