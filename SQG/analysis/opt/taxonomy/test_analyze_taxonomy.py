"""Tests of analyze_taxonomy.py on synthetic spectra (numpy only; python3 test_analyze_taxonomy.py). Prints
TAXO_ANALYSIS_TEST_OK.

1. Statistics, 16 clusters x 4 pairs:
   - null (e = noise): level 1 |z| < 4 in every band; the fraction of shells with p < 0.0027 is <= 2%;
   - a band-uniform bias planted at 8 SE in one band: recovered within 25%, z >= 5, other bands quiet;
   - a mode-specific bias (random sign per shell, as float32's quantized damping) in one band: level 1 sees only the
     residual of its signs (< half its size); level 2 flags >= 50% of that band's shells and estimates its learnable
     size within 30%.
2. End to end: main() on a small fake job directory (2 workers x 2 states x 2 pairs, every arm's keys) whose controls
   are built to pass; checks the summary, the sealed file and its sha256.
"""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_taxonomy as AT                                         # noqa: E402

RNG = np.random.RandomState(20261106)


def cnormal(*shape):
    return (RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)) / np.sqrt(2)


def synth(P, ncl, sigma_e=1.0):
    G = AT.geometry()
    amp = np.where(G["kap"] > 0, 1.0 / np.maximum(G["kap"], 1.0), 0.0)
    x = cnormal(P, AT.NC, AT.HALF + 1) * amp
    d = cnormal(P, AT.NC, AT.HALF + 1) * amp * 0.5
    e = cnormal(P, AT.NC, AT.HALF + 1) * amp * sigma_e * 1e-3
    cl = np.repeat(np.arange(ncl), P // ncl)
    return G, x, d, e, cl


def test_null_and_planted():
    P, ncl = 64, 16
    G, x, d, e, cl = synth(P, ncl)
    allv = np.ones_like(G["unique"])
    l1 = AT.level1(G, cl, ncl, x, d, e, allv)
    assert all(abs(l1[n]["z_x"]) < 4 and abs(l1[n]["z_d"]) < 4 for n in AT.BNAME if l1[n]), l1
    l2, det = AT.level2(G, cl, ncl, x, d, e, allv)
    p = det["p"][np.isfinite(det["p"])]
    frac = float(np.mean(p < AT.P_SHELL))
    assert frac <= 0.02, frac
    # MDE is 2.8 SE in delta units
    n = "16-32"
    assert abs(l1[n]["MDE_x"] - 2.8 * l1[n]["se_alpha"] * l1[n]["rx_over_rd"]) < 1e-15

    # band-uniform bias at 8 SE in band 16-32
    b = AT.BNAME.index("16-32")
    a_true = 8 * l1[n]["se_alpha"]
    mb = (G["band"] == b)
    e1 = e + np.where(mb, a_true, 0.0) * x
    l1p = AT.level1(G, cl, ncl, x, d, e1, allv)
    assert abs(l1p[n]["alpha"] / a_true - 1) < 0.25 and l1p[n]["z_x"] >= 5, l1p[n]
    assert all(abs(l1p[m]["z_x"]) < 4 for m in AT.BNAME if m != n and l1p[m]), l1p

    # mode-specific bias: random sign per shell in band 4-8, size 20 x the per-shell noise
    b2 = AT.BNAME.index("4-8")
    gid2, shells, bos = AT.shell_ids(G, allv)
    sign = RNG.choice([-1.0, 1.0], size=len(shells))
    sel = bos == b2
    fit0 = det["fit"]
    se_shell = np.sqrt(np.median(fit0["cov"][sel, 0, 0]))
    coef = np.zeros(len(shells))
    coef[sel] = 20 * se_shell * sign[sel]
    amap = np.where(gid2 >= 0, coef[np.maximum(gid2, 0)], 0.0)
    e2 = e + amap * x
    l1m = AT.level1(G, cl, ncl, x, d, e2, allv)
    l2m, _ = AT.level2(G, cl, ncl, x, d, e2, allv)
    m4 = "4-8"
    assert l2m[m4]["frac_p"] >= 0.5, l2m[m4]
    mask = (gid2 >= 0) & (G["band"] == b2)
    true_lb = np.sqrt(np.sum(np.abs(amap * x)[:, mask] ** 2) / np.sum(np.abs(d[:, mask]) ** 2))
    assert abs(l2m[m4]["learnable_bias"] / true_lb - 1) < 0.3, (l2m[m4], true_lb)
    assert abs(l1m[m4]["B_x"]) < 0.5 * true_lb, (l1m[m4], true_lb)   # the pooled fit sees only the signs' residual
    print(f"stats: null shell frac {frac:.4f}; uniform bias recovered {l1p[n]['alpha'] / a_true:.3f} "
          f"(z {l1p[n]['z_x']:.1f}); shell bias: pooled z {l1m[m4]['z_x']:.1f}, shells flagged "
          f"{l2m[m4]['frac_p']:.2f}, learnable {l2m[m4]['learnable_bias'] / true_lb:.3f} of truth")


def frame(spec):
    return np.fft.irfft2(spec, s=(AT.NC, AT.NC)).astype(np.float32)


def fake_job(root):
    G = AT.geometry()
    KX, KY, kap = G["KX"], G["KY"], G["kap"]
    nu, mu = 2e-4, 1.0
    rate = np.where(kap > 0, nu * kap ** 2 + mu / np.where(kap > 0, kap, 1.0) ** 2, 0.0)
    d64 = np.exp(-rate * AT.DT)
    d32 = d64.astype(np.float32).astype(np.float64)
    drate = -np.log1p((d32 - d64) / d64) / AT.DT
    apred = -drate * AT.LAG_T
    afm = G["af_valid"]
    inj_unit = np.array([2e-3, 1.5e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    bidx = np.maximum(G["band"], 0)
    for w in (0, 1):
        scale = 1.0 + w                                               # worker 1 plants twice the shift
        states = []
        for s in range(2):
            states.append({"member": 1 + w, "t": 100.0 * (s + 1), "spe": 1.0, "cfl": 0.4})
            th = np.fft.rfft2(RNG.standard_normal((AT.NC, AT.NC)))
            for j in range(2):
                D = {}
                th_L = th + 0.3 * np.fft.rfft2(RNG.standard_normal((AT.NC, AT.NC)))
                ref = {0: th}
                for k in AT.LAGS:
                    ref[k] = th + (th_L - th) * k / AT.LAGS[-1]
                for k, v in ref.items():
                    D[f"lean_u1__sub__{k}"] = frame(v)
                    D[f"lean_u1__af__{k}"] = np.fft.rfft2(D[f"lean_u1__sub__{k}"].astype(np.float64)) * afm
                x = np.fft.rfft2(D["lean_u1__sub__0"].astype(np.float64))
                arms = ["lean32ek", "inj"] + (["dquant", "inj64", "lean32", "f64ic32", "lean32sr"] if j == 0 else []) \
                    + (["prod", "lean32ek_fast"] if (s == 0 and j == 0) else [])
                for arm in arms:
                    for k in AT.LAGS:
                        rk = np.fft.rfft2(D[f"lean_u1__sub__{k}"].astype(np.float64))
                        dk = rk - x
                        noise = 1e-4 * np.fft.rfft2(RNG.standard_normal((AT.NC, AT.NC)))
                        if arm == "dquant":
                            err = apred * (x + dk / 2)
                        elif arm == "inj64":
                            err = scale * inj_unit[bidx] * (G["band"] >= 0) * x * k / AT.LAGS[-1]
                        elif arm == "inj":
                            err = scale * inj_unit[bidx] * (G["band"] >= 0) * x * k / AT.LAGS[-1] + noise
                        elif arm == "prod":
                            err = 0.0
                        else:
                            err = noise
                        v = rk + err
                        D[f"{arm}__sub__{k}"] = frame(v) if arm != "prod" else D[f"lean_u1__sub__{k}"]
                        D[f"{arm}__af__{k}"] = (rk + err) * afm if arm == "dquant" else \
                            np.fft.rfft2(D[f"{arm}__sub__{k}"].astype(np.float64)) * afm
                if j == 0 and s == 0:
                    rk = np.fft.rfft2(D["lean_u1__sub__100"].astype(np.float64))
                    dk = rk - x
                    for n, c in ((40, 0.0), (30, 1e-4), (32, 0.8e-4), (80, -0.64e-4)):
                        v = rk + c * dk
                        D[f"dt{n}__sub__100"] = frame(v) if n != 40 else D["lean_u1__sub__100"]
                        D[f"dt{n}__af__100"] = np.fft.rfft2(D[f"dt{n}__sub__100"].astype(np.float64)) * afm
                np.savez(os.path.join(root, f"pair_w{w}_s{s}_p{j}.npz"), **D)
                th = th_L
        np.savez(os.path.join(root, f"tables_w{w}.npz"), decay64=d64, decay32=d32, rate=rate, kx=KX, ky=KY,
                 drate_inj=np.zeros_like(rate))
        lag_keys = [str(k * AT.STRIDE) for k in AT.LAGS]
        js = {"worker": w, "states": states, "injection": {"drate_b": (scale * np.ones(len(AT.BANDS))).tolist()},
              "bitwise": {"prod_vs_lean_u1_frames": {k: True for k in lag_keys},
                          "fast_vs_lean32ek_state": {"40": True, "400": True, "4000": True},
                          "dt40_vs_lean_u1": [{"state": 0, "frame": True, "state_hash": True}]},
              "ms_per_step": {"lean_u1": 6.4, "lean32ek": 3.35, "lean32ek_fast": 2.7, "lean32sr": 4.5, "lean32": 3.2}}
        with open(os.path.join(root, f"worker_{w}.json"), "w") as fh:
            json.dump(js, fh)


def test_end_to_end():
    tmp = tempfile.mkdtemp(prefix="taxo_test_", dir=os.environ.get("TMPDIR"))
    try:
        jobdir, outdir, sealdir = (os.path.join(tmp, n) for n in ("job", "out", "sealed"))
        os.makedirs(jobdir)
        fake_job(jobdir)
        argv = sys.argv
        sys.argv = ["analyze_taxonomy.py", "--in", jobdir, "--out", outdir, "--sealed-dir", sealdir]
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                AT.main()
        finally:
            sys.argv = argv
        log = buf.getvalue()
        assert "TAXO_ANALYSIS_DONE" in log and "ERROR" not in log, log[-2000:]
        with open(os.path.join(outdir, "taxonomy_summary.json")) as fh:
            summ = json.load(fh)
        assert summ["n_pairs"] == 8 and summ["n_states"] == 4, summ["n_pairs"]
        assert summ["control_a"]["pass"], summ["control_a"]
        assert summ["control_b"]["pass"], summ["control_b"]
        assert summ["bitwise"]["prod_eq_lean_u1"] and summ["bitwise"]["dt40_eq_lean_u1"]
        assert summ["fast_form_pass"]
        assert "unsealed" not in summ and "candidates" not in summ
        with open(summ["sealed"]["path"], "rb") as fh:
            blob = fh.read()
        assert hashlib.sha256(blob).hexdigest() == summ["sealed"]["sha256"]
        sealed = json.loads(blob)
        assert {"lean32ek", "dt30", "dt32", "dt_order_ratio"} <= set(sealed["candidates"]), sealed["candidates"].keys()
        r = sealed["candidates"]["dt_order_ratio"]["sub"]["4-8"]
        assert abs(r - (-0.64)) < 0.05, r                            # D80 / D30 as planted
        for line in log.splitlines():
            if line.startswith(("TAXO_CONTROL", "TAXO_ANALYSIS_DONE")):
                print("  " + line[:160])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_null_and_planted()
    test_end_to_end()
    print("TAXO_ANALYSIS_TEST_OK")


if __name__ == "__main__":
    main()
