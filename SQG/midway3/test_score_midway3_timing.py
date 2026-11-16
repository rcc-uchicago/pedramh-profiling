"""test_score_midway3_timing.py -- the Midway3 scorer on real Polaris result files. Stdlib only (login node OK).

  python3 SQG/midway3/test_score_midway3_timing.py            # prints MIDWAY3_SCORER_TEST_OK

  A  the A100 results scored as if they were H100: every validity gate that has data PASSES, every speed prediction
     is REFUTED (speedup 1.0 is outside every interval), the cuFFT version and split-sum predictions CONFIRM.
  B  a synthetic H100: the same files scaled to the preregistered point values (and batch-32 arms built from them):
     every prediction CONFIRMS and no gate fails.
"""
import copy
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "analysis", "results")
sys.path.insert(0, HERE)
import score_midway3_timing as SC  # noqa: E402

P = SC.load_predictions()


def stage(tmp):
    """Polaris files laid out as the Midway jobs write them."""
    d = {k: os.path.join(tmp, k) for k in ("inrow", "o12", "tr1", "tr2", "b32")}
    for x in d.values():
        os.makedirs(x)
    shutil.copy(os.path.join(RES, "polaris_opt_round1c_7815637", "seed1.json"), d["inrow"])
    shutil.copy(os.path.join(RES, "polaris_inrow_7853317", "inrow_probe_prod.json"), d["inrow"])
    for src, dst in (("polaris_o12_7835278", "o12"), ("polaris_o12tr_7840909", "tr1"), ("polaris_o12tr_7853365", "tr2")):
        for f in os.listdir(os.path.join(RES, src)):
            if f == "o12_extrapolation.json" or f.endswith(".log"):
                shutil.copy(os.path.join(RES, src, f), d[dst])
    return d


def verdicts(res):
    g = {x["id"]: x["verdict"] for x in res["gates"]}
    p = {x["id"]: x["verdict"] for x in res["predictions"]}
    return g, p


def case_a(d):
    res = SC.score(P, None, d["inrow"], d["o12"], [d["tr1"], d["tr2"]], None)
    g, p = verdicts(res)
    for k in ("S1", "S2", "I1", "I2") + tuple(f"O1.{a}" for a in SC.ARMS) + tuple(f"O2.{a}" for a in SC.TR_ARMS):
        assert g[k] == "PASS", (k, g[k])
    assert g["E1"] == "MISSING" and g["O3.ref_b32"] == "MISSING"
    for a in SC.ARMS:
        assert p[f"P1.{a}"] == "REFUTED" and p[f"P2.{a}"] == "REFUTED" and p[f"P3.{a}"] == "REFUTED", a
    assert p["P1.cufft_version"] == "CONFIRMED" and p["P1.split_sum"] == "CONFIRMED"
    assert res["gates_failed"] == 0
    return res


def rewrite(path, fn):
    with open(path) as fh:
        obj = json.load(fh)
    fn(obj)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def case_b(d):
    sp = {a: P["solver_speedup"]["f64" if a in P["f64_arms"] else "f32"][0] for a in SC.ARMS}
    ml_sp = P["ml_speedup"][0]
    A = P["a100"]

    def s1(o):
        for a in ("ref", "lean_u1"):
            o["bench"]["ms_per_step"][a] = A["ms"][a] / sp[a]
    rewrite(os.path.join(d["inrow"], "seed1.json"), s1)

    def ir(o):
        b = o["bench_ms_per_step"]
        b["lean_u1"] = A["ms"]["lean_u1"] / sp["lean_u1"]
        for arm, key in (("lean32ek_fast", "lean32ek_fast"), ("if32_25", "if32"), ("tilerows_if25", "tilerows32_if_eager"),
                         ("tilerows_inrow_if25", "inrow32_if_eager")):
            b[key] = A["ms"][arm] / sp[arm]
        b["inrow32_split_ms"] = {k: v / 2.1 for k, v in b["inrow32_split_ms"].items()}
        b["inrow32_eager"] /= 2.1
    rewrite(os.path.join(d["inrow"], "inrow_probe_prod.json"), ir)

    def h100(a, v, batch=16, ml_factor=1.0):
        frames, ml = 12 * A["s_frame"][a] / sp[a], (A["s_step"][a] - 12 * A["s_frame"][a]) / ml_sp * ml_factor
        v["pieces"]["s_frame"], v["pieces"]["s_step"] = A["s_frame"][a] / sp[a], frames + ml
        steps = P["steps_1node_b16"] if batch == 16 else P["steps_1node_b32"]
        v["projections"] = {f"1 node, 4 GPUs, batch {batch}": {"hours": steps * (frames + ml) / 3600 + 1.0}}
        v["parsed"]["peak"] = 40.0 if batch == 32 else v["parsed"].get("peak")

    def ex(o):
        for a, v in o["solvers"].items():
            if a in SC.ARMS:
                h100(a, v)
    for k in ("o12", "tr1", "tr2"):
        rewrite(os.path.join(d[k], "o12_extrapolation.json"), ex)
    # batch 32: the b16 entries of ref and tilerows_inrow_if25, ML x 1.9
    src = {}
    for k in ("o12", "tr2"):
        with open(os.path.join(d[k], "o12_extrapolation.json")) as fh:
            src.update(json.load(fh)["solvers"])
    b32 = {"solvers": {}}
    for a in ("ref", "tilerows_inrow_if25"):
        v = copy.deepcopy(src[a])
        h100(a, v, batch=32, ml_factor=1.9)
        b32["solvers"][a + "_b32"] = v
    with open(os.path.join(d["b32"], "o12_extrapolation.json"), "w") as fh:
        json.dump(b32, fh)
    shutil.copy(os.path.join(d["tr2"], "tilerows_inrow_if25.log"), os.path.join(d["b32"], "tilerows_inrow_if25_b32.log"))
    env = os.path.join(d["inrow"], "midway_env_extra.json")
    with open(env, "w") as fh:
        json.dump({"torch_gpus": ["NVIDIA H100 NVL"] * 4, "jax_devices": ["gpu:NVIDIA H100 NVL"] * 4,
                   "warp": {"mathdx": 1}, "tile_fft": {"rel_l2_vs_torch": 3e-7}, "nvmath": {"cufft_version": 11002}}, fh)
    res = SC.score(P, d["inrow"], d["inrow"], d["o12"], [d["tr1"], d["tr2"]], d["b32"])
    g, p = verdicts(res)
    bad_p = {k: v for k, v in p.items() if v != "CONFIRMED"}
    bad_g = {k: v for k, v in g.items() if v != "PASS"}
    assert not bad_p, bad_p
    assert not bad_g, bad_g
    return res


def main():
    with tempfile.TemporaryDirectory() as tmp:
        a = case_a(stage(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        b = case_b(stage(tmp))
    print(f"MIDWAY3_SCORER_TEST_OK A: {a['counts']} gates_failed=0; B: all {b['counts']['CONFIRMED']} predictions "
          f"confirmed, all {len(b['gates'])} gates pass")


if __name__ == "__main__":
    main()
