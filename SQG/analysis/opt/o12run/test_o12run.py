"""test_o12run.py -- CPU tests of the o1.2-on-Polaris harness (no jax, no torch, no GPU). Python >= 3.6 + numpy:
login node (system python3) or the job (sqg-env).

  * every o12_polaris.PATCHES entry matches o1.2's source exactly once and keeps the line count; the patched source
    has no Weidong path, no fp32_precision, no train_m16 read, and compiles (on Python >= 3.8; o1.2 targets 3.10);
  * build_val_store.py + the zarr stand-in round-trip the frames, as o1.2's make_pairs() / PairDataset read them; a
    broken pair chain is refused;
  * Helper.ref_frame returns theta = sub x 8 exactly; check_first_segment's verdicts (PASS, FAIL raises, INFO, none);
  * o12_extrapolate parses a synthetic o1.2 log (tqdm \\r noise included) and its projection equals the formula.
  python3 SQG/analysis/opt/o12run/test_o12run.py        # prints O12_TEST_OK
"""
import io
import json
import math
import os
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_val_store as BVS                                     # noqa: E402
import o12_extrapolate as EX                                      # noqa: E402
import o12_polaris as OP                                          # noqa: E402


def test_patches():
    with open(OP.O12) as fh:
        src = fh.read()
    out = OP.apply_patches(src)
    assert out.count("\n") == src.count("\n"), "line count changed"
    for bad in ("/data/weidong", "/home/nvidia", "fp32_precision", "zarr.open_group(TRAIN_STORE", "GD.DT_SAVE",
                "RUN_CHECKPOINT_DIR", "range(1, 8)", "plt.close(fig)"):
        assert bad not in out, f"patched source still contains {bad!r}"
    missing = [name for name, _, new in OP.PATCHES if new not in out]
    assert not missing, f"patched text absent: {missing}"
    for name, old, new in OP.PATCHES:                             # a patch that matches twice must be refused
        try:
            OP.apply_patches(src + "\n" + old, [(name, old, new)])
            raise AssertionError(f"{name}: a double match was accepted")
        except RuntimeError:
            pass
    try:
        OP.apply_patches(src, [("bad", "N_EPOCH = 250\n", "N_EPOCH = 250\n\n")])
        raise AssertionError("a line-count change was accepted")
    except RuntimeError:
        pass
    compiled = sys.version_info >= (3, 8)
    if compiled:
        compile(out, OP.O12, "exec", dont_inherit=True)
    return len(OP.PATCHES), compiled


def fake_taxonomy(d, states=2, pairs=4, seed=0):
    rng = np.random.RandomState(seed)
    st = []
    for s in range(states):
        prev = (rng.randn(256, 256).astype(np.float32) / np.float32(8.0))
        for j in range(pairs):
            nxt = (rng.randn(256, 256).astype(np.float32) / np.float32(8.0))
            np.savez(os.path.join(d, f"pair_w0_s{s}_p{j}.npz"), lean_u1__sub__0=prev, lean_u1__sub__100=nxt)
            prev = nxt
        st.append({"member": s + 1, "t": 100.0 * (s + 1), "offset_calls": 0, "pairs": [{"j": j} for j in range(pairs)]})
    with open(os.path.join(d, "worker_0.json"), "w") as fh:
        json.dump({"states": st}, fh)


def test_val_store_and_refs():
    with tempfile.TemporaryDirectory() as d:
        ref, store = os.path.join(d, "ref"), os.path.join(d, "store")
        os.makedirs(ref)
        fake_taxonomy(ref)
        with redirect_stdout(io.StringIO()):
            theta, sources = BVS.build(ref, store)
        assert theta.shape == (10, 256, 256) and theta.dtype == np.float32 and len(sources) == 2
        mods = OP.stand_ins(d, False)
        g = mods["zarr"].open_group(store, mode="r")
        rows = [np.arange(s["offset"], s["offset"] + s["n_records"] - 1) for s in g.attrs["sources"]]   # make_pairs
        assert sum(len(r) for r in rows) == 8
        with np.load(os.path.join(ref, "pair_w0_s1_p2.npz")) as z:
            want_x, want_y = z["lean_u1__sub__0"] * np.float32(8), z["lean_u1__sub__100"] * np.float32(8)
        row = g.attrs["sources"][1]["offset"] + 2
        assert np.array_equal(g["theta"][row], want_x) and np.array_equal(g["theta"][row + 1], want_y)
        assert np.array_equal(g["theta"][row] / 8.0, want_x / np.float32(8))          # PairDataset's x == sub
        # Helper.ref_frame: o1.2 ring frame k = 100 (j + 1) of the state's first segment
        env = {"O12_OUT": d, "O12_VAL_STORE": store, "O12_REF_DIR": ref, "SQG_ASQG_PY": "x", "RANK": "1"}
        h = OP.Helper(env)
        assert np.array_equal(h.ref_frame(2, 200.0, 300), want_y) and h.ref_frame(2, 200.0, 500) is None
        assert h.ref_frame(9, 100.0, 100) is None
        # a broken chain is refused
        with np.load(os.path.join(ref, "pair_w0_s0_p1.npz")) as z:
            a0, a1 = z["lean_u1__sub__0"].copy(), z["lean_u1__sub__100"]
        a0[0, 0] += np.float32(1e-3)
        np.savez(os.path.join(ref, "pair_w0_s0_p1.npz"), lean_u1__sub__0=a0, lean_u1__sub__100=a1)
        try:
            with redirect_stdout(io.StringIO()):
                BVS.build(ref, os.path.join(d, "store2"))
            raise AssertionError("a broken pair chain was accepted")
        except SystemExit as e:
            assert "VAL_STORE_CHAIN" in str(e)
        # check_first_segment verdicts
        online = type("O", (), {"n_pairs": 600, "k": 712, "load_time": 9.5})()
        for solver, diff, n, verdict in (("ref", 0.0, 4, "PASS"), ("lean_u1", 0.0, 4, "PASS"),
                                         ("lean32ek_fast", 3e-6, 4, "INFO"), ("ref", -1.0, 0, "NO_REFERENCE")):
            h = OP.Helper(dict(env, RANK="0", O12_SOLVER=solver))
            buf = io.StringIO()
            with redirect_stdout(buf):
                h.check_first_segment(diff, n, 0.0, online)
            assert f"O12_BITWISE solver={solver}" in buf.getvalue() and buf.getvalue().rstrip().endswith(verdict)
        h = OP.Helper(dict(env, RANK="0", O12_SOLVER="lean_u1"))
        try:
            with redirect_stdout(io.StringIO()):
                h.check_first_segment(1e-7, 4, 0.0, online)
            raise AssertionError("a bitwise mismatch of lean_u1 passed")
        except AssertionError as e:
            assert "differ from production" in str(e)


LOG = """strict fp32 check passed: conv 1.0e-07, matmul 2.0e-07, attention 1.0e-07
UViT_m16_dt0.02_tendency_online3: 4 GPU(s), NVIDIA A100-SXM4-40GB
O12_KNOB BATCH=16 (o1.2: 32)
O12_KNOB N_VAL=64 (o1.2: 200)
O12_SOLVER ref
O12_FILL pairs=600 frames=712 seconds=280.0 load_s=10.0 s_per_frame_steady=0.38000 solver=ref
O12_BITWISE solver=ref frames_per_rank=4 max_abs_diff=0.000e+00 PASS
buffer filled: 600 pairs from 1 segments in 280 s; first segment checked
ConvNextUViT 25.00M params, tendency (big skip), circular padding
epochs:   0%|          | 0/4\r                    \rpeak GPU memory (PyTorch) 20.3 GiB
epoch 0: train 0.0100  val 0.0200 (persistence 0.0300)  lr 2.00e-05  5.000 s/step  (155 s)
O12_EPOCH epoch=0 steps=30 step_median=5.0000 step_p10=4.9 step_p90=5.2 train_s=150.0 val_s=3.00 save_last_s=1.00 solver=ref
O12_SAVE best seconds=1.50
epoch 1: train 0.0090  val 0.0190 (persistence 0.0300)  lr 4.00e-05  4.900 s/step  (150 s)
O12_EPOCH epoch=1 steps=30 step_median=4.9000 step_p10=4.8 step_p90=5.0 train_s=147.0 val_s=2.00 save_last_s=1.00 solver=ref
O12_SAVE best seconds=1.50
epoch 2: train 0.0080  val 0.0180 (persistence 0.0300)  lr 6.00e-05  4.900 s/step  (150 s)
O12_EPOCH epoch=2 steps=30 step_median=4.8000 step_p10=4.8 step_p90=5.0 train_s=147.0 val_s=2.00 save_last_s=1.00 solver=ref
done: best val 0.0180
O12_PHASES solver=ref startup_to_fill_done_s=340.0 total_s=820.0
O12_RUN_DONE solver=ref
"""


def test_extrapolate():
    r = EX.parse(LOG)
    assert r["world_size"] == 4 and r["peak"] == 20.3 and r["params"] == 25.0 and r["run_done"]
    assert len(r["epochs"]) == 3 and r["bitwise"]["verdict"] == "PASS" and r["filled"]["seconds"] == 280
    p = EX.pieces(r)
    assert p["s_step"] == 4.85 and p["t_init"] == 60.0 and p["t_val"] == 2.0 and p["t_save_best"] == 1.5
    assert p["val_batches_run"] == 1 and p["s_frame"] == 0.38 and p["t_load"] == 10.0
    pr = EX.project(p, 14, 16, 4)
    steps = 250 * ((174993 // 14) // 16)
    assert steps == 195250 == pr["steps"]                                     # o1.2's own 195,250 steps
    want = (60.0 + 10.0 + (100 + 4 * 4901) * 0.38 + steps * (4.85 + 0.005)
            + 250 * (2.0 * math.ceil(math.ceil(200 / 14) / 16) + 1.0 + 1.5)
            + math.ceil(steps * 12 / 5000) * 10.0)
    assert abs(pr["hours"] * 3600 - want) < 1e-6, (pr["hours"] * 3600, want)
    assert abs(pr["gpu_hours"] - pr["hours"] * 14) < 1e-9
    one = EX.project(p, 4, 16, 1)
    assert one["steps"] == 250 * ((174993 // 4) // 16) and one["terms_s"]["steps"] == one["steps"] * 4.85
    p2 = dict(p, s_frame=None)
    assert EX.project(p2, 4, 16, 1) is None
    return pr["hours"]


def main():
    n, compiled = test_patches()
    test_val_store_and_refs()
    h = test_extrapolate()
    print(f"O12_TEST_OK patches={n} line-count kept, {'compiled' if compiled else 'compile skipped (python < 3.8)'}; "
          f"val store + zarr stand-in round-trip; ref frames exact; bitwise verdicts; parser + formula "
          f"(synthetic 14-GPU projection {h:.1f} h) (python {sys.version.split()[0]})")


if __name__ == "__main__":
    main()
