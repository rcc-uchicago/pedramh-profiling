"""test_prec_outcome.py -- tests of the bf16-vs-ieee training-outcome harness (PREDICTIONS_prec_outcome.md).

Login part (numpy; python 3.6 ok): analyze_prec on synthetic arms.
  * the stratified splits: seeds (1, 2) x 3 ieee + (1, 1, 2, 2) candidates = 10 x 10 = 100, the observed one included;
    perm_p on a 3-arm example = 1/3;
  * every verdict on a constructed case:
    - PASS_P (no shift);
    - WORSE / BETTER (val shifted 5 sd up / down);
    - ROLLOUT_DIFFERS (one band-window cell shifted);
    - UNSTABLE (a candidate initial condition blows up);
    - INCONCLUSIVE (a 1.1 sd shift inside a wide candidate spread);
    - INVALID (wrong prec, short run, no bf16 in the forward, a seed without candidates, different val pairs);
  * train_dose.py's census hook returns None (ast).
GPU part (--gpu --frames <dir> --run <scratch dir>; in the job, under sqg-env):
  * set_precision: tf32 makes o1.2's check fail, and a conv under bf16 autocast returns bfloat16;
  * train_dose.py runs 20 steps on the real frames as ieee and as bf16:
    - PREC_FORWARD counts bf16 outputs under bf16 and none under ieee;
    - bf16's peak memory is below ieee's;
    - a rerun resumes (DOSE_RESUME step=20) and keeps forward_census in config.json;
    - resuming the bf16 arm as ieee is refused (DOSE_PREC_MISMATCH);
  * rollout_dose.py loads the bf16-trained best.pt in strict fp32 (5 steps).

  python3 SQG/analysis/opt/mixprec/test_prec_outcome.py                                  # login: PREC_TEST_OK
  python  SQG/analysis/opt/mixprec/test_prec_outcome.py --gpu --frames <dir> --run <dir>   # job stage 0
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GATED = os.path.join(os.path.dirname(HERE), "gateD")
sys.path.insert(0, HERE)
import analyze_prec as AP                                            # noqa: E402

STEPS, T, NIC = 100, 500, 4                                          # 2 rollout windows of 250 steps
VAL_REF = [0.005540, 0.005592, 0.005534, 0.005620, 0.005612, 0.005688]   # gate D's six ieee arms (best val_clean)
SEEDS_REF, SEEDS_CAND = [1, 2, 1, 2, 1, 2], [1, 1, 2, 2]
CLIM = np.ones(8)


def arm(name, seed, prec, val, r_off, kind):
    """A load_arm()-shaped synthetic arm: flat history at `val`, rollout band energies = CLIM * (1 + r_off)."""
    hist = [{"step": s, "train": 0.0016, "val_clean": val + 1e-6 * (s == STEPS), "val_own": val, "lr": 1e-4,
             "s_step": 0.42 if prec == "ieee" else 0.33, "peak_GiB": None if kind == "ref" else 12.0}
            for s in (STEPS // 2, STEPS)]
    E = np.ones((T + 1, NIC, 8)) * CLIM * (1.0 + np.asarray(r_off))
    cfg = {"alpha": 0.0, "seed": seed, "steps": STEPS, "batch": 16, "train_segments": ["m01"], "val_segments": ["m04"],
           "n_train_pairs": 10, "n_val": 8, "persistence_val": 0.08}
    if prec != "ieee":
        cfg.update({"prec": prec, "prec_record": {"prec": prec}, "forward_census": {"bf16": 30, "conv_linear_modules": 30}})
    ro = {"E": E, "blow": np.full(NIC, -1), "ic_tags": np.zeros((NIC, 2), int), "val_pick": np.arange(8),
          "preds": np.full((8, 4, 4), val, np.float32)}
    return {"dir": f"/x/{name}", "name": name, "cfg": cfg, "hist": hist, "ro": ro, "seed": seed, "prec": prec}


def build(val_cand, r_ref=None, r_cand=None):
    r_ref = r_ref if r_ref is not None else [np.full(8, v) for v in (-0.02, -0.01, 0.0, 0.01, 0.02, 0.0)]
    r_cand = r_cand if r_cand is not None else [np.full(8, v) for v in (-0.015, 0.015, -0.005, 0.005)]
    ref = [arm(f"ieee{i}", s, "ieee", v, r, "ref") for i, (s, v, r) in enumerate(zip(SEEDS_REF, VAL_REF, r_ref))]
    cand = [arm(f"bf16_{i}", s, "bf16", v, r, "cand") for i, (s, v, r) in enumerate(zip(SEEDS_CAND, val_cand, r_cand))]
    return ref, cand


def run(ref, cand, prec="bf16"):
    return AP.analyze(ref, cand, prec, CLIM, 2.0, "best", steps=STEPS)


def login_tests():
    sp = AP.splits(SEEDS_REF, SEEDS_CAND)
    obs = tuple(range(6, 10))
    assert len(sp) == 100 and len(set(sp)) == 100 and obs in sp, (len(sp), obs in sp)
    p, o, n = AP.perm_p([0.0, 0.0, 1.0], 2, AP.splits([1, 1], [1]), lambda r, c: abs(c.mean() - r.mean()))
    assert n == 3 and abs(p - 1 / 3) < 1e-12 and o == 1.0, (p, o, n)

    m, sd = np.mean(VAL_REF), np.std(VAL_REF, ddof=1)
    cases = {}
    ref, cand = build([m - 0.3 * sd, m + 0.3 * sd, m + 0.1 * sd, m - 0.1 * sd])
    cases["PASS_P"] = run(ref, cand)
    ref, cand = build([max(VAL_REF) + 5 * sd + 1e-7 * i for i in range(4)])
    cases["WORSE"] = run(ref, cand)
    ref, cand = build([min(VAL_REF) - 5 * sd - 1e-7 * i for i in range(4)])
    cases["BETTER"] = run(ref, cand)
    shifted = [np.full(8, v) for v in (-0.015, 0.015, -0.005, 0.005)]
    for r in shifted:
        r[4] += 0.5                                                  # one band, both windows
    ref, cand = build([m - 0.3 * sd, m + 0.3 * sd, m + 0.1 * sd, m - 0.1 * sd], r_cand=shifted)
    cases["ROLLOUT_DIFFERS"] = run(ref, cand)
    ref, cand = build([m - 0.3 * sd, m + 0.3 * sd, m + 0.1 * sd, m - 0.1 * sd])
    cand[2]["ro"]["blow"][1] = 300
    cand[2]["ro"]["E"][300:, 1] = np.nan
    cases["UNSTABLE"] = run(ref, cand)
    ref, cand = build([m + 1.1 * sd - 3 * sd, m + 1.1 * sd + 3 * sd, m + 1.1 * sd - 3 * sd, m + 1.1 * sd + 3 * sd])
    cases["INCONCLUSIVE"] = run(ref, cand)
    for want, res in cases.items():
        assert res["verdict"] == want, (want, res["verdict"], res.get("validity"), res.get("val", {}).get(
            "M1_val_best", {}).get("p"), res.get("rollout", {}).get("p_fw"))
    assert cases["WORSE"]["val"]["M1_val_best"]["p"] == 0.01, cases["WORSE"]["val"]["M1_val_best"]["p"]
    assert cases["ROLLOUT_DIFFERS"]["rollout"]["p_fw"] == 0.01 and cases["ROLLOUT_DIFFERS"]["rollout"]["windows"] == 2
    assert cases["UNSTABLE"]["rollout"]["windows"] == 2                      # 3 of 4 alive >= half: still usable
    assert cases["PASS_P"]["rollout"]["T"] < 1e-9 and cases["PASS_P"]["rollout"]["p_fw"] == 1.0

    def invalid(mutate, why):
        ref, cand = build([m] * 4)
        mutate(ref, cand)
        res = run(ref, cand)
        assert res["verdict"] == "INVALID" and any(why in b for b in res["validity"]), (why, res["validity"])

    invalid(lambda r, c: c[0].update(prec="ieee"), "candidate arm has prec")
    invalid(lambda r, c: c[1]["hist"][-1].update(step=STEPS - 1), "of 100 steps")
    invalid(lambda r, c: c[2]["cfg"].update(forward_census={"bf16": 0}), "no bf16")
    invalid(lambda r, c: [x.update(seed=1) for x in c], "seed 2 lacks")
    invalid(lambda r, c: c[3]["ro"].update(val_pick=np.arange(1, 9)), "val_pick differs")
    invalid(lambda r, c: r[0].update(prec="bf16"), "ref arm has prec")
    return {k: (v["verdict"], v["val"]["M1_val_best"]["p"], v["rollout"]["p_fw"]) for k, v in cases.items()}


def test_hook_returns_none():
    """train_dose.py's forward hooks must return None (a returned value replaces the module's output; the first
    draft's lambda returned census.setdefault(...), a str). ast only, login-runnable."""
    import ast
    with open(os.path.join(GATED, "train_dose.py")) as fh:
        tree = ast.parse(fh.read())
    hooks = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "register_forward_hook":
            assert not isinstance(node.args[0], ast.Lambda), "a forward hook is a lambda (its value is returned)"
            hooks += 1
        if isinstance(node, ast.FunctionDef) and node.name == "hook":
            assert not any(isinstance(r, ast.Return) and r.value is not None for r in ast.walk(node)), \
                "the census hook returns a value"
    assert hooks == 1, hooks
    return hooks


def sh(cmd, env):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    return p.returncode, p.stdout


def gpu_tests(frames, run_dir):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, GATED)
    import train_dose as TD
    dev = torch.device("cuda:0")
    TD.strict_fp32(torch, F, dev)
    ctx, rec = TD.set_precision(torch, F, dev, "tf32")
    assert rec["fp32_errors_after"]["matmul"] > 2e-5 or rec["fp32_errors_after"]["conv"] > 2e-5, rec
    TD.strict_fp32(torch, F, dev)                                    # back to o1.2's flags
    ctx, rec = TD.set_precision(torch, F, dev, "bf16")
    with ctx():
        d = F.conv2d(torch.ones(1, 1, 8, 8, device=dev), torch.ones(1, 1, 3, 3, device=dev)).dtype
    assert d == torch.bfloat16, d
    ctx, rec = TD.set_precision(torch, F, dev, "ieee")
    with ctx():
        d = F.conv2d(torch.ones(1, 1, 8, 8, device=dev), torch.ones(1, 1, 3, 3, device=dev)).dtype
    assert d == torch.float32, d

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    env = dict(os.environ)
    train = os.path.join(GATED, "train_dose.py")
    base = [sys.executable, train, "--frames", frames, "--alpha", "0", "--seed", "1", "--steps", "20", "--val-every",
            "10", "--ckpt-every", "10", "--n-val", "32"]
    out, peak = {}, {}
    for prec in ("ieee", "bf16"):
        d = os.path.join(run_dir, prec)
        rc, log = sh(base + ["--out", d, "--prec", prec], env)
        with open(d + ".log", "w") as fh:
            fh.write(log)
        assert rc == 0 and "DOSE_ARM_DONE" in log, (prec, rc, log[-2000:])
        line = [ln for ln in log.splitlines() if ln.startswith("PREC_FORWARD")]
        assert len(line) == 1, (prec, line)
        n16 = int(line[0].split("bf16=")[1].split()[0])
        assert (n16 > 0) == (prec == "bf16"), line
        with open(os.path.join(d, "history.json")) as fh:
            h = json.load(fh)
        assert h["config"]["prec"] == prec and h["config"]["forward_census"]["bf16"] == n16, h["config"]
        peak[prec] = max(r["peak_GiB"] for r in h["hist"])
        out[prec] = line[0]
    assert peak["bf16"] < peak["ieee"], peak
    rc, log = sh(base + ["--out", os.path.join(run_dir, "bf16"), "--prec", "bf16"], env)
    assert rc == 0 and "DOSE_RESUME step=20" in log and "DOSE_ARM_DONE" in log, log[-2000:]
    with open(os.path.join(run_dir, "bf16", "config.json")) as fh:                 # a finished arm's rerun (preemption)
        assert json.load(fh).get("forward_census", {}).get("bf16", 0) > 0, "forward_census lost on rerun"
    rc, log = sh(base + ["--out", os.path.join(run_dir, "bf16"), "--prec", "ieee"], env)
    assert rc != 0 and "ERROR DOSE_PREC_MISMATCH" in log, (rc, log[-2000:])
    rc, log = sh([sys.executable, os.path.join(GATED, "rollout_dose.py"), "--frames", frames, "--arm",
                  os.path.join(run_dir, "bf16"), "--which", "best", "--steps", "5", "--n-ic", "4", "--n-pred", "8",
                  "--n-val", "32"], env)
    assert rc == 0 and "ROLLOUT_DONE" in log, log[-2000:]
    return {"forward": out, "peak_GiB": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--frames")
    ap.add_argument("--run")
    a = ap.parse_args()
    print(f"PREC_TEST login {json.dumps(login_tests())} hooks={test_hook_returns_none()}", flush=True)
    if a.gpu:
        print(f"PREC_TEST gpu {json.dumps(gpu_tests(a.frames, a.run))}", flush=True)
    print("PREC_TEST_OK", flush=True)


if __name__ == "__main__":
    main()
