"""score_hybrid.py <hybrid_probe.json> -- score PREDICTIONS_hybrid.md's 12 predictions mechanically (stdlib)."""
import json
import sys


def main():
    R = json.load(open(sys.argv[1]))
    env, micro, gates, bench, ver = (R.get(k, {}) for k in ("env", "micro", "gates", "bench_ms_per_step", "verdict"))
    out = []

    def say(n, ok, detail, *needed):
        v = "untested" if any(x is None for x in needed) else ("confirmed" if ok else "refuted")
        out.append((n, v, detail))

    say(1, env.get("cufft_version", 0) >= 11300, f"cuFFT {env.get('cufft_version')}, nvmath {env.get('nvmath')}",
        env.get("cufft_version"))
    built = not any(e["where"] == "HYBRID_BUILD" for e in R.get("errors", []))
    say(2, built and "1" in gates and "rel" in gates["1"], "callbacks compiled, plans built" if built else "build failed")
    g32, gif, g64 = ver.get("G32"), ver.get("GIF"), ver.get("G64")
    say(3, g32 is not None and all(x is True for x in g32), f"G32 {g32}", g32)
    say(4, gif is not None and all(x is True for x in gif), f"GIF {gif}", gif)
    say(5, g64 is True, f"G64 {g64}", g64)
    b = gates.get("4000", {}).get("bands", {})
    if "hybrid32_vs_lean_u1" in b and "lean32ek_fast_vs_lean_u1" in b:
        r = [x / y for k in ("sub", "af") for x, y in zip(b["hybrid32_vs_lean_u1"][k], b["lean32ek_fast_vs_lean_u1"][k])]
        say(6, all(0.5 <= x <= 2 for x in r), f"ratio {min(r):.3f} .. {max(r):.3f}")
    else:
        say(6, False, "no 4,000-step bands", None)
    h32, g = bench.get("hybrid32_eager"), bench.get("hybrid32_graph")
    say(7, h32 is not None and 1.5 <= h32 <= 2.2, f"hybrid32 {h32} ms (lean32ek_fast {bench.get('lean32ek_fast')})", h32)
    say(8, g is not None and h32 is not None and abs(g / h32 - 1) <= 0.03 and bench.get("hybrid32_graph_finite"),
        f"graph {g} vs eager {h32}", g, h32)
    h64 = bench.get("hybrid64_eager")
    say(9, h64 is not None and 3.0 <= h64 <= 4.5, f"hybrid64 {h64} ms (lean_u1 {bench.get('lean_u1')})", h64)
    hif = bench.get("hybrid32_if_eager")
    say(10, hif is not None and h32 is not None and hif <= 1.03 * h32, f"if {hif} vs {h32}", hif, h32)
    sp = bench.get("hybrid32_split_ms") or {}
    if sp:
        fft = sum(v for k, v in sp.items() if k != "warp_update")
        say(11, fft / sum(sp.values()) >= 0.70, f"cuFFT share {fft / sum(sp.values()):.3f}; split {sp}")
    else:
        say(11, False, "no split", None)
    a, t = micro.get("nvmath_c2c_complex64", {}).get("med_ms"), micro.get("torch_fft2_complex64", {}).get("med_ms")
    say(12, a is not None and t is not None and abs(a / t - 1) <= 0.05, f"nvmath {a} vs torch {t} ms", a, t)
    for n, v, d in out:
        print(f"P{n:<2d} {v:9s} {d}")
    tally = {v: sum(1 for _, x, _ in out if x == v) for v in ("confirmed", "refuted", "untested")}
    worth = (h32 is not None and h32 <= 2.2) or (h64 is not None and h64 <= 2.70)
    print("HYB_SCORE " + " ".join(f"{k}={v}" for k, v in tally.items()) + f" | worth integrating (pre-set): {worth}")


if __name__ == "__main__":
    main()
