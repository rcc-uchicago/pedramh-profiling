"""score_inrow.py <inrow_probe_prod.json> -- PREDICTIONS_inrow.md's 15 predictions scored mechanically (stdlib;
python >= 3.6). A prediction without its data is UNSCORED. Prints IR_SCORE ... last.

  python3 score_inrow.py <json>
  python3 score_inrow.py --self-test
"""
import json
import sys


def score(R):
    g, b, ver, V = R.get("gates", {}), R.get("bench_ms_per_step", {}), R.get("verdict", {}), R.get("validity", {})
    build = R.get("build", {})
    th = V.get("theta", {})
    out = []

    def say(n, ok, detail, *needed):
        out.append((n, "UNSCORED" if any(x is None for x in needed) else ("CONFIRMED" if ok else "REFUTED"), detail))

    tile = [k for k in build if k.startswith(("tilerows", "inrow"))]
    fin = [all(x.get("rel", {}).get("finite", {}).values()) for x in g.values() if "rel" in x]
    say(1, bool(tile) and all(build[k]["ok"] for k in tile) and bool(fin) and all(fin),
        f"built {sum(build[k]['ok'] for k in tile)}/{len(tile)}; finite {fin}", build or None)
    e_ir, e_tr = th.get("rms_rel_inrow"), th.get("rms_rel_c2r")
    say(2, e_ir is not None and e_ir <= 1e-5, f"in-tile q_u rms rel {e_ir}", e_ir)
    say(3, e_ir is not None and e_tr is not None and e_ir <= 1.5 * e_tr, f"{e_ir} vs C2R {e_tr}", e_ir, e_tr)
    hz = V.get("HI_ZERO_ROWS")
    say(4, bool(V.get("HI_PRESERVED")) and bool(hz) and all(hz.values()), f"preserved {V.get('HI_PRESERVED')}, "
        f"zero rows {hz}", V.get("HI_PRESERVED"), hz)
    g32, gif = ver.get("G32"), ver.get("GIF")
    say(5, g32 is not None and all(x is True for x in g32), f"G32 {g32}", g32)
    say(6, gif is not None and all(x is True for x in gif), f"GIF {gif}", gif)
    bands = g.get("4000", {}).get("bands", {})
    if "inrow32_vs_lean_u1" in bands and "tilerows32_vs_lean_u1" in bands:
        r = [x / y for k in ("sub", "af") for x, y in zip(bands["inrow32_vs_lean_u1"][k], bands["tilerows32_vs_lean_u1"][k])]
        say(7, all(0.5 <= x <= 2 for x in r), f"ratio {min(r):.3f} .. {max(r):.3f}")
    else:
        say(7, False, "no bands", None)
    wf = g.get("40", {}).get("variants", {}).get("inrow32_wfirst")
    say(8, bool(wf and wf.get("bitwise_vs_inrow32")), f"wfirst {wf}", wf)
    ti, tt = b.get("inrow32_eager"), b.get("tilerows32_eager")
    ratio = ti / tt if (ti and tt) else None
    say(9, ratio is not None and ratio <= 0.97, f"inrow32 {ti} / tilerows32 {tt} = {ratio}", ratio)
    say(10, ratio is not None and ratio <= 0.93, f"ratio {ratio}", ratio)
    sp = b.get("inrow32_split_ms") or {}
    rows, cth, h0 = sp.get("tile_rows_ir"), sp.get("cufft_cols_th"), sp.get("halve0")
    say(11, rows is not None and 0.33 <= rows <= 0.45, f"tile_rows_ir {rows} ms", rows)
    say(12, cth is not None and 0.10 <= cth <= 0.18, f"cufft_cols_th {cth} ms", cth)
    say(13, h0 is not None and h0 <= 0.01, f"halve0 {h0} ms", h0)
    t32, tif = ver.get("TR_G32"), ver.get("TR_GIF")
    say(14, t32 is not None and tif is not None and all(x is True for x in t32 + tif), f"TR_G32 {t32} TR_GIF {tif}",
        t32, tif)
    t512 = b.get("inrow32_b512_eager")
    say(15, t512 is not None and ti is not None and t512 > ti, f"b512 {t512} vs b256 {ti}", t512, ti)
    return out, bool(ver.get("worth"))


def self_test():
    R = {"build": {"tilerows32": {"ok": True}, "inrow32": {"ok": True}},
         "validity": {"theta": {"rms_rel_inrow": 1e-7, "rms_rel_c2r": 1e-7}, "THETA": True, "HI_PRESERVED": True,
                      "HI_ZERO_ROWS": {"inrow32": True, "inrow32_if": True}},
         "gates": {"1": {"rel": {"finite": {"inrow32": True}}},
                   "40": {"rel": {"finite": {"inrow32": True}},
                          "variants": {"inrow32_wfirst": {"bitwise_vs_inrow32": True}}},
                   "4000": {"rel": {"finite": {"inrow32": True}},
                            "bands": {"inrow32_vs_lean_u1": {"sub": [1.0] * 8, "af": [1.0] * 8},
                                      "tilerows32_vs_lean_u1": {"sub": [1.0] * 8, "af": [1.0] * 8}}}},
         "verdict": {"G32": [True, True], "GIF": [True, True], "TR_G32": [True, True], "TR_GIF": [True, True],
                     "worth": True},
         "bench_ms_per_step": {"inrow32_eager": 1.40, "tilerows32_eager": 1.578, "inrow32_b512_eager": 1.5,
                               "inrow32_split_ms": {"tile_rows_ir": 0.38, "cufft_cols_th": 0.14, "halve0": 0.004}}}
    out, worth = score(R)
    assert worth and all(v == "CONFIRMED" for _, v, _ in out), out
    out, worth = score({})
    assert not worth and all(v == "UNSCORED" for n, v, _ in out if n != 7), out
    print("IR_SCORE_TEST_OK")


def main():
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    out, worth = score(json.load(open(sys.argv[1])))
    for n, v, d in out:
        print(f"IR_PRED P{n:<2d} {v:9s} {d}")
    tally = {v: sum(1 for _, x, _ in out if x == v) for v in ("CONFIRMED", "REFUTED", "UNSCORED")}
    print("IR_SCORE " + " ".join(f"{k.lower()}={v}" for k, v in tally.items()) + f" | worth (pre-set): {worth}")


if __name__ == "__main__":
    main()
