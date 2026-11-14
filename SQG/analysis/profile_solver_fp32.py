"""profile_solver_fp32.py -- Job E: what does the A-SQG solver step cost in float32 instead of float64?

Timing only. A float32 solver computes something different (DESIGN §4: a science decision, Weidong / jesswan), so
nothing here is an equivalence check against the archive. This measures what the change would buy.

The float32 step is the production step with every array in single precision:
  - model and stepper from sqg.aux(GD.p._replace(f64=False)) (pyqg_jax Precision.SINGLE), same N, L, k_f, dt;
  - GD.F0_hat as complex64 and GD.decay as float32;
  - the same four lines as A-SQG_data.inner (make_inner below), and the same 40-step lax.scan as o1.2.
A-SQG_data.py is imported unmodified (hash-pinned), through profile_solver.import_asqg.

Checks, before any number is reported:
  - builder: make_inner(GD.sm, GD.F0_hat, GD.decay) is bitwise equal to GD.inner after 40 steps (the f32 variant
    differs from production only in dtype, not in code);
  - no silent promotion: after one call every array leaf of the f32 state is complex64/float32, and the optimized
    HLO of the f32 call (fused bodies included) has no full-size c128/f64 [1,]4096x2049 or 4096x4096 array (small
    1-D f64 wavenumber vectors are allowed). JAX runs with x64 enabled
    (A-SQG_data sets it), so one float64 constant would quietly turn the whole step back into float64.
--mode bench: ms/step (median of --repeats x --calls jitted 40-step calls, as profile_solver.py), with and without
coarse(), HLO op counts, XLA cost model of one step, JAX memory. --mode nsys: as profile_solver.py --mode nsys.

The IC is SQG_INIT=random (the production spin-up IC stepped to AB3 level 2, profile_solver.py): every step runs the
same FFTs whatever the field, so the timing does not depend on it. The f32 state is the f64 one cast to f32.
Writes <out>/profile_solver_fp32.json; prints FP32_OK.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import json
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import profile_solver as PS                                     # noqa: E402  (import_asqg, STRIDE, CALLS, M)


def make_inner(sm, F0_hat, decay):
    """A-SQG_data.inner with its module globals passed in: AB3 on advection + fixed forcing, then exact damping."""
    def inner(state, _):
        updates = sm.model.get_updates(state.state)
        updates = updates.update(qh=updates.qh + F0_hat)
        state = sm.stepper.apply_updates(state, updates)
        qh = sm.model.postprocess_state(state.state).qh * decay
        return state.update(state=state.state.update(qh=qh)), None
    return inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "nsys"], default="bench")
    ap.add_argument("--variant", default="f32_ref", help="nsys mode: f64_ref or f32_ref")
    ap.add_argument("--calls", type=int, default=60, help="bench: calls per throughput measurement (x40 steps)")
    ap.add_argument("--repeats", type=int, default=3, help="bench: throughput repeats (median reported)")
    ap.add_argument("--reps", type=int, default=5, help="nsys: optimizer-equivalents (12 calls + coarse) captured")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t_import = time.time()
    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    t_import = time.time() - t_import

    N, STRIDE, CALLS, M = GD.N, PS.STRIDE, PS.CALLS, PS.M
    if os.environ.get("SQG_INIT", "checkpoint") != "random":
        raise SystemExit("ERROR SQG_FP32_NEEDS_RANDOM_INIT: this harness only runs from SQG_INIT=random")
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    st64 = GD.advance(template, 2)
    assert int(st64._ablevel) == 2

    # ---- the float32 model: the same parameters, Precision.SINGLE ----------------------------------------------
    p32 = GD.p._replace(f64=False)
    sm32, _, _, _, _ = GD.sqg.aux(p32)
    F0_32 = GD.F0_hat.astype(jnp.complex64)
    decay32 = GD.decay.astype(jnp.float32)
    down = {np.dtype("complex128"): jnp.complex64, np.dtype("float64"): jnp.float32}
    st32 = jax.tree_util.tree_map(lambda x: x.astype(down[x.dtype]) if x.dtype in down else x, st64)

    inners = {"f64_ref": GD.inner, "f64_built": make_inner(GD.sm, GD.F0_hat, GD.decay),
              "f32_ref": make_inner(sm32, F0_32, decay32)}
    calls = {k: jax.jit(lambda st, f=f: lax.scan(f, st, None, length=STRIDE)[0]) for k, f in inners.items()}
    st0 = {"f64_ref": st64, "f64_built": st64, "f32_ref": st32}
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    copy = lambda st: jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), st)
    block = jax.block_until_ready

    def opt_step(fn, st):
        frames = []
        for _ in range(CALLS):
            st = fn(st)
            frames.append(coarse(st))
        return st, frames

    # ---- checks ---------------------------------------------------------------------------------------------------
    a = [np.asarray(x) for x in jax.tree_util.tree_leaves(block(calls["f64_ref"](copy(st64))))]
    b = [np.asarray(x) for x in jax.tree_util.tree_leaves(block(calls["f64_built"](copy(st64))))]
    builder_ok = all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))
    out32 = block(calls["f32_ref"](copy(st32)))
    leaf_dtypes = sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(out32) if x.ndim > 0})
    hlo32 = calls["f32_ref"].lower(st32).compile().as_text()
    wide = len(re.findall(r"(c128|f64)\[(1,)?4096,(2049|4096)\]", hlo32))   # full-size fields, incl. fused bodies
    if not builder_ok:
        raise SystemExit("ERROR SQG_FP32_BUILDER_NOT_BITWISE: make_inner differs from GD.inner in float64")
    if wide or any(d in ("complex128", "float64") for d in leaf_dtypes):
        raise SystemExit(f"ERROR SQG_FP32_PROMOTED: {wide} full-size f64/c128 HLO arrays, state dtypes {leaf_dtypes}")
    print(f"checks: builder bitwise == GD.inner (40 steps); f32 state {leaf_dtypes}, 0 full-size f64/c128 in HLO")

    if args.mode == "nsys":
        import torch                                    # only for cudaProfilerStart/Stop + NVTX ranges
        fn, st = calls[args.variant], copy(st0[args.variant])
        for _ in range(2):
            st, fr = opt_step(fn, st)
            block((st, fr))
        torch.cuda.profiler.start()
        for r in range(args.reps):
            torch.cuda.nvtx.range_push(f"{args.variant}_optstep_{r}")
            st, fr = opt_step(fn, st)
            block((st, fr))
            torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        print(f"PROFILE_OK nsys {args.variant}: {args.reps} x {CALLS} x {STRIDE} steps captured")
        return

    dev = jax.devices()[0]
    res = {"device": str(dev), "device_kind": dev.device_kind, "jax": jax.__version__, "init": "random",
           "import_s": round(t_import, 2), "stride": STRIDE, "calls_per_optstep": CALLS,
           "checks": {"builder_bitwise_vs_GD_inner_40_steps": builder_ok, "f32_state_dtypes": leaf_dtypes,
                      "f32_hlo_full_size_f64_or_c128_arrays": wide}}
    pats = {"fft": r"\bfft\(", "conditional": r"\bconditional\(", "while": r"\bwhile\(", "copy": r"\bcopy\(",
            "fusion": r"\bfusion\("}
    res["hlo"], res["cost_one_step"] = {}, {}
    for name in ("f64_ref", "f32_ref"):
        txt = calls[name].lower(st0[name]).compile().as_text()
        with open(os.path.join(args.out, f"hlo_{name}.txt"), "w") as fh:
            fh.write(txt)
        res["hlo"][name] = {k: len(re.findall(p, txt)) for k, p in pats.items()}
        ca = jax.jit(lambda st, f=inners[name]: f(st, None)[0]).lower(st0[name]).compile().cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        res["cost_one_step"][name] = {k: float(ca[k]) for k in ("flops", "bytes accessed") if k in ca}

    def bench(fn, st, with_coarse):
        st = copy(st)
        for _ in range(3):
            st = fn(st)
        block(st)
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            frames = []
            for _ in range(args.calls):
                st = fn(st)
                if with_coarse:
                    frames.append(coarse(st))
            block((st, frames))
            times.append(time.perf_counter() - t0)
        return 1e3 * statistics.median(times) / (args.calls * STRIDE)

    res["ms_per_step"] = {n: round(bench(calls[n], st0[n], False), 4) for n in ("f64_ref", "f32_ref")}
    res["ms_per_step_with_coarse"] = {n: round(bench(calls[n], st0[n], True), 4) for n in ("f64_ref", "f32_ref")}
    res["optstep_solver_s"] = {n: round(v * CALLS * STRIDE / 1e3, 3) for n, v in res["ms_per_step_with_coarse"].items()}
    res["f64_over_f32"] = round(res["ms_per_step"]["f64_ref"] / res["ms_per_step"]["f32_ref"], 3)
    for name, ca in res["cost_one_step"].items():
        if "bytes accessed" in ca:
            ca["effective_GBps"] = round(ca["bytes accessed"] / (res["ms_per_step"][name] * 1e-3) / 1e9, 1)
    ms = dev.memory_stats() or {}
    res["jax_memory_GiB"] = {k: round(ms[k] / 2**30, 2) for k in ("bytes_in_use", "peak_bytes_in_use") if k in ms}

    with open(os.path.join(args.out, "profile_solver_fp32.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"device {dev.device_kind}; HLO per 40-step call {res['hlo']}")
    print(f"one-step XLA cost {res['cost_one_step']}")
    print(f"ms/step {res['ms_per_step']}  with coarse {res['ms_per_step_with_coarse']}  f64/f32 = {res['f64_over_f32']}")
    print(f"solver per optimizer step (480 steps + 12 coarse): {res['optstep_solver_s']} s; JAX memory {res['jax_memory_GiB']}")
    print(f"FP32_OK bench -> {args.out}/profile_solver_fp32.json")


if __name__ == "__main__":
    main()
