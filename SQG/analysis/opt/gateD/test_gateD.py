"""test_gateD.py -- CPU tests of gate D's harness. numpy; the torch checks run only if torch imports, i.e. in the job.

  * The injected pattern is the taxonomy regression's own quantity: B_hat built from bias_pattern_lean32ek.json and
    fitted back by analyze_taxonomy.level1 returns the same per-band alpha / beta.
  * band_energy_np sums to the field's mean square (zero-mean field).
  * pair_index / val_pairs / rollout_ics on fake segments.
  * analyze_dose on a synthetic run, with a known drift and absorption:
    - PASS_D when the low-dose effect / low dose is far below the seed spread;
    - FAIL_D when it is not;
    - INCONCLUSIVE_NO_POWER when the positive control shows nothing.
    Run for the default doses 1000 / 10000 and, when the job sets them, env DOSE_LO / DOSE_HI (job 7841338 died
    here: the synthetic arms were named a1000 / a10000 while analyze_dose looked for a860 / a8600).
  * With torch: bias_hat and band_energy on torch equal the numpy versions to float32 accuracy.

  python3 SQG/analysis/opt/gateD/test_gateD.py          # prints GATED_TEST_OK
"""
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dose_common as DC                                             # noqa: E402
import analyze_dose as AD                                            # noqa: E402

AT = DC.AT


def test_pattern_roundtrip(G, cx, cd):
    rng = np.random.RandomState(1)
    P = 6
    x = rng.randn(P, DC.NC, DC.NC)
    y = x + 0.1 * rng.randn(P, DC.NC, DC.NC)
    e = DC.bias_hat_np(x, y, cx, cd)
    X, Y, E = np.fft.rfft2(x), np.fft.rfft2(y), np.fft.rfft2(e)
    l1 = AT.level1(G, np.arange(P), P, X, Y - X, E, np.ones_like(G["unique"]))
    with open(DC.PATTERN) as fh:
        p = json.load(fh)
    worst = 0.0
    for b, nm in enumerate(DC.BNAME):
        ra, rb = l1[nm]["alpha"], l1[nm]["beta"]
        worst = max(worst, abs(ra - p["alpha"][b]) / abs(p["alpha"][b]), abs(rb - p["beta"][b]) / abs(p["beta"][b]))
    assert worst < 1e-6, f"pattern round trip off by {worst:.2e}"
    return worst


def test_energy(G):
    x = np.random.RandomState(2).randn(DC.NC, DC.NC)
    x -= x.mean()
    e = DC.band_energy_np(x, G)
    corner = (G["band"] < 0) & (G["kap"] > 0)                       # modes outside every band but k = 0 (none)
    assert not corner.any(), "a non-zero mode lies outside every band"
    assert abs(e.sum() - np.mean(x ** 2)) < 1e-9 * np.mean(x ** 2), (e.sum(), np.mean(x ** 2))


def fake_frames(d, members=(1, 2, 3, 4), times=(100.0, 200.0), n_frames=251, seed=3):
    rng = np.random.RandomState(seed)
    for m in members:
        for t in times:
            name = f"m{m:02d}_t{t:07.2f}"
            a = (rng.randn(n_frames, DC.NC, DC.NC) * 0.1).astype(np.float32)
            np.save(os.path.join(d, name + ".npy"), a)
            with open(os.path.join(d, name + ".json"), "w") as fh:
                json.dump({"complete": True, "n_frames": n_frames}, fh)


def test_indexing(d):
    segs, arrs = DC.open_segments(d, [4])
    assert [s[1] for s in segs] == [4, 4] and all(a.shape == (251, DC.NC, DC.NC) for a in arrs)
    seg, k = DC.pair_index(arrs)
    assert len(seg) == 2 * (251 - DC.LAG) and k.max() == 251 - DC.LAG - 1
    x, y, pick = DC.val_pairs(arrs, 20)
    assert np.array_equal(x[3], arrs[seg[pick[3]]][k[pick[3]]]) and np.array_equal(y[3], arrs[seg[pick[3]]][k[pick[3]] + DC.LAG])
    ics, tags = DC.rollout_ics(arrs, 4)
    assert ics.shape == (4, DC.NC, DC.NC) and tags == [(0, 0), (0, 125), (1, 0), (1, 125)]


def synthetic_run(run, frames, G, cx, cd, delta_lo, delta_hi, doses, kap_hi=1.0, seed_noise=0.02, T=1000, n_ic=8):
    """Arm dirs with history.json + rollout_best.npz: R_0A = 0, R_0B = seed_noise, R_lo = delta_lo,
    R_hi = delta_hi (relative to the climatology), and one-step predictions f_a = f_0A + dose kappa B_hat.
    doses = (lo, hi) as strings; the arms are a<lo>_sA / a<hi>_sA, as run_tr_gateD_polaris.pbs names them."""
    lo, hi = doses
    clim, _, _ = AD.climatology(frames, G)
    _, va = DC.open_segments(frames, [4])
    xv, yv, pick = DC.val_pairs(va, 16)
    B = DC.bias_hat_np(xv.astype(np.float64), yv.astype(np.float64), cx, cd)
    rng = np.random.RandomState(5)
    f0 = rng.randn(*xv.shape)
    lev = {"a0_sA": (0.0, 0.0), "a0_sB": (seed_noise, 0.0), f"a{lo}_sA": (delta_lo, float(lo)),
           f"a{hi}_sA": (delta_hi, float(hi) * kap_hi)}
    for arm, (r, shift) in lev.items():
        d = os.path.join(run, arm)
        os.makedirs(d, exist_ok=True)
        hist = [{"step": s, "train": 0.01 / (1 + s / 1000), "val_clean": 0.02 + 1e-6 * (s - 3000) ** 2 / 1e3,
                 "val_own": 0.02} for s in range(1000, 7000, 1000)]
        with open(os.path.join(d, "history.json"), "w") as fh:
            json.dump({"hist": hist}, fh)
        E = np.broadcast_to(clim * (1 + r), (T + 1, n_ic, len(clim))).copy()
        E *= 1 + 0.001 * rng.randn(T + 1, n_ic, 1)
        np.savez(os.path.join(d, "rollout_best.npz"), E=E, xmax=np.ones((T + 1, n_ic)), blow=np.full(n_ic, -1),
                 ic_tags=np.zeros((n_ic, 2)), preds=(f0 + shift * B).astype(np.float32), val_pick=pick)


def run_analysis(run, frames, out, doses):
    env = dict(os.environ, DOSE_LO=doses[0], DOSE_HI=doses[1])      # analyze_dose reads its arm names from these
    p = subprocess.run([sys.executable, os.path.join(HERE, "analyze_dose.py"), "--run", run, "--frames", frames,
                        "--out", out, "--n-val", "16"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, env=env)
    last = [l for l in p.stdout.splitlines() if l.startswith("DOSE_ANALYSIS_DONE")]
    assert p.returncode == 0 and last, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    return last[-1].split()[1].split("=")[1], p.stdout


def dose_pairs():
    """The defaults (job 7835786) and, if the job sets them, the env doses."""
    pairs = [("1000", "10000")]
    env = (os.environ.get("DOSE_LO"), os.environ.get("DOSE_HI"))
    if all(env) and env not in pairs:
        pairs.append(env)
    return pairs


def test_analysis(G, cx, cd):
    verdicts = {}
    for doses in dose_pairs():
        got = {}
        with tempfile.TemporaryDirectory() as d:
            frames = os.path.join(d, "frames")
            os.makedirs(frames)
            fake_frames(frames)
            for case, (lo, hi, kh) in {"pass": (0.001, 0.05, 1.0), "fail": (5.0, 50.0, 1.0),
                                       "nopower": (0.0, 0.0, 0.0)}.items():
                run = os.path.join(d, case)
                synthetic_run(run, frames, G, cx, cd, lo, hi, doses, kap_hi=kh)
                v, out = run_analysis(run, frames, os.path.join(d, case + "_out"), doses)
                got[case] = v
                if case == "pass":
                    assert "kappa: a0_sB(noise ref)" in out or f"a{doses[0]}_sA=1.000" in out, out
        assert got == {"pass": "PASS_D", "fail": "FAIL_D", "nopower": "INCONCLUSIVE_NO_POWER"}, (doses, got)
        verdicts["/".join(doses)] = got
    return verdicts


def test_torch(G, cx, cd):
    try:
        import torch
    except ImportError:
        return "skipped (no torch)"
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    bias_hat, band_energy = DC.torch_tools(torch, G, dev)
    rng = np.random.RandomState(7)
    x, y = rng.randn(2, DC.NC, DC.NC).astype(np.float32), rng.randn(2, DC.NC, DC.NC).astype(np.float32)
    bt = bias_hat(torch.from_numpy(x)[:, None].to(dev), torch.from_numpy(y)[:, None].to(dev)).cpu().numpy()[:, 0]
    bn = DC.bias_hat_np(x.astype(np.float64), y.astype(np.float64), cx, cd)
    rel = np.abs(bt - bn).max() / np.abs(bn).max()
    et = band_energy(torch.from_numpy(x)[:, None].to(dev)).cpu().numpy()
    en = DC.band_energy_np(x, G)
    rel_e = np.abs(et - en).max() / np.abs(en).max()
    assert rel < 1e-4 and rel_e < 1e-4, (rel, rel_e)
    return f"torch ok on {dev} (bias_hat rel {rel:.1e}, energy rel {rel_e:.1e})"


def main():
    G = DC.geometry()
    cx, cd = DC.bias_maps(G)
    worst = test_pattern_roundtrip(G, cx, cd)
    test_energy(G)
    with tempfile.TemporaryDirectory() as d:
        fake_frames(d, members=(4,))
        test_indexing(d)
    v = test_analysis(G, cx, cd)
    t = test_torch(G, cx, cd)
    print(f"GATED_TEST_OK pattern round trip {worst:.1e}; Parseval; indexing; analysis verdicts {v}; {t} "
          f"(python {sys.version.split()[0]}, numpy {np.__version__})")


if __name__ == "__main__":
    main()
