"""profile_solver.py -- Job A: standalone profile of the A-SQG solver step (JAX + pyqg_jax, 4096^2 float64 AB3).

Measures what one online-training optimizer step costs on the solver side, without the training loop:
the production pattern is 12 x [jitted lax.scan of STRIDE=40 `inner` steps, then `coarse()` = irfft2 + [::16, ::16]]
= S = 480 solver steps (o1.2-TrainUViT_m16_tendency.py, OnlinePairs.advance).

The solver is the real one: /project/pedramh/weidong/SQG/fixed_ring/A-SQG_data.py imported as a module exactly as
o1.2 does (--member 1, stdout silenced, its signal handlers restored; its import-time mkdirs target directories that
already exist), started from a real archived AB3 checkpoint of member 1 (steady state, AB level 2).

Variants of the 40-step jitted call (all must be bitwise identical to `ref`; checked here):
  ref          : jax.jit(lambda st: lax.scan(GD.inner, st, None, length=40)[0])        -- as o1.2
  donate       : same, donate_argnums=0 (no defensive copy of the ~400 MB AB3 carry)
  fixed        : GD.inner with the AB3 level fixed at 2 (pyqg_jax's runtime lax.switch on _ablevel removed;
                 same coefficients 23/12, -16/12, 5/12 x dt), valid once the state is at level 2
  fixed_donate : both
Tests (--mode bench): compile time; optimized-HLO op counts (fft / conditional / while / copy / fusion) and XLA cost
analysis of ONE step; bitwise equality vs ref after 480 steps; whether a call blocks the Python thread (dispatch time
vs completion time); steady ms/step per variant, alone and in the production pattern (with coarse); coarse() alone;
JAX device memory; which CUDA libraries were actually loaded (module cuda/12.9 prepends LD_LIBRARY_PATH).
--mode nsys: warm up, then cudaProfilerStart, REPS optimizer-equivalents of one variant (one NVTX range each),
cudaProfilerStop -- run under `nsys profile --capture-range=cudaProfilerApi`.

Writes <out>/profile_solver_bench.json (+ hlo_<variant>.txt); prints PROFILE_OK.

Polaris overrides (profile_solver_polaris.pbs; each one unset = the Midway path above, unchanged):
  SQG_ASQG_PY        path of the vendored solver copy (SQG/fixed_ring/A-SQG_data.py, byte-identical to Weidong's)
  SQG_TRAINING_ROOT  run the solver with only its hard-coded TRAINING_ROOT line replaced, in memory (patch_training_root)
  SQG_INIT=random    start from the production spin-up IC stepped to AB3 level 2 instead of the archived checkpoint
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # as production (JAX shares the GPU with torch)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import contextlib
import importlib.util
import io
import json
import re
import signal
import statistics
import sys
import time

ASQG_PY = os.environ.get("SQG_ASQG_PY", "/project/pedramh/weidong/SQG/fixed_ring/A-SQG_data.py")
CKPT_NAME = "ckpt_t0100.00.npz"
STRIDE = 40                 # solver steps per jitted call (o1.2 STRIDE)
CALLS = 12                  # calls per optimizer step (S = 480 = 12 x 40)
M = 16                      # coarse subsampling


def patch_training_root(src, root):
    """`src` with its one `TRAINING_ROOT = ...` line replaced by `TRAINING_ROOT = Path(r"<root>")`.

    A-SQG_data.py mkdirs its run dirs under the hard-coded Midway TRAINING_ROOT at import, and hashes its own bytes
    into CONFIG, so the file itself must not be edited (PROFILING_PROMPT.md §3.1). One line in, one line out: the
    line numbers of every traceback still match the file on disk.
    """
    lines = src.split("\n")
    hits = [i for i, line in enumerate(lines) if line.startswith("TRAINING_ROOT = ")]
    if len(hits) != 1:
        raise RuntimeError(f"expected one TRAINING_ROOT line in {ASQG_PY}, found {len(hits)}")
    lines[hits[0]] = f'TRAINING_ROOT = Path(r"{root}")'
    return "\n".join(lines)


def import_asqg():
    """A-SQG_data.py as a module, as o1.2's import_asqg (argv --member 1, silenced, signal handlers restored).

    With SQG_TRAINING_ROOT set, the source is compiled with patch_training_root applied, under the unmodified file's
    name and __file__, so CONFIG["solver_sha256"] is still the hash of the file on disk.
    """
    argv, handlers = sys.argv, {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM)}
    sys.argv = [ASQG_PY, "--member", "1"]
    try:
        spec = importlib.util.spec_from_file_location("asqg_data", ASQG_PY)
        mod = importlib.util.module_from_spec(spec)
        root = os.environ.get("SQG_TRAINING_ROOT")
        with contextlib.redirect_stdout(io.StringIO()):
            if root:
                with open(ASQG_PY, "rb") as fh:
                    src = patch_training_root(importlib.util.decode_source(fh.read()), root)
                exec(compile(src, ASQG_PY, "exec", dont_inherit=True), mod.__dict__)
            else:
                spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
        for s, h in handlers.items():
            signal.signal(s, h)
    return mod


def loaded_libs():
    """CUDA libraries mapped into this process (which cuFFT/cudart JAX really uses)."""
    libs = set()
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.split()[-1] if len(line.split()) >= 6 else ""
            if re.search(r"lib(cufft|cudart|cublas|cudnn|nvToolsExt|cupti)[^/]*\.so", path):
                libs.add(path)
    return sorted(libs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "nsys"], default="bench")
    ap.add_argument("--variant", default="ref", help="nsys mode: which variant to capture")
    ap.add_argument("--calls", type=int, default=60, help="bench: calls per throughput measurement (x40 steps)")
    ap.add_argument("--repeats", type=int, default=3, help="bench: throughput repeats (median reported)")
    ap.add_argument("--reps", type=int, default=5, help="nsys: optimizer-equivalents (12 calls + coarse) captured")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t_import = time.time()
    GD = import_asqg()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from pyqg_jax import steppers as pst
    t_import = time.time() - t_import

    N = GD.N
    dev = jax.devices()[0]
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))   # stepper-state template, as GD.main
    init = os.environ.get("SQG_INIT", "checkpoint")
    if init == "random":
        # DT is fixed and the CFL checks are host-side, so every step runs the same 4096^2 FFTs whatever the field;
        # the only state that changes which code runs is the AB3 level, 2 after two steps (PROFILING_PROMPT.md §2b).
        ckpt, completed = None, None
        st0 = GD.advance(template, 2)
    elif init == "checkpoint":
        ckpt = GD.CHECKPOINT_DIR / CKPT_NAME
        st0, completed = GD.load_checkpoint(template, ckpt)
    else:
        raise ValueError(f"SQG_INIT={init!r}: expected random or checkpoint")
    assert int(st0._ablevel) == 2, f"checkpoint AB level {int(st0._ablevel)} != 2: the fixed variant is not valid"

    # ---- variants -------------------------------------------------------------------------------------------
    dt = GD.sm.stepper.dt
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt      # pyqg_jax AB3Stepper level-2 branch, verbatim

    def ab3_level2(ss, updates):
        """pyqg_jax AB3Stepper.apply_updates with the lax.switch replaced by its level-2 branch."""
        up, upp = ss._updates
        new_state = pst._nostep_tree_map(lambda v, u, u_p, u_pp: v + (c1 * u) + (c2 * u_p) + (c3 * u_pp),
                                         ss.state, updates, up, upp)
        return pst.AB3State(state=new_state, t=ss.t + jnp.float32(dt), tc=ss.tc + 1, _ablevel=jnp.uint8(2),
                            _updates=(pst._map_state_remove_nostep(updates), up))

    def inner_fixed(state, _):
        """GD.inner with ab3_level2 instead of sm.stepper.apply_updates."""
        updates = GD.sm.model.get_updates(state.state)
        updates = updates.update(qh=updates.qh + GD.F0_hat)
        state = ab3_level2(state, updates)
        qh = GD.sm.model.postprocess_state(state.state).qh * GD.decay
        return state.update(state=state.state.update(qh=qh)), None

    def make(inner, donate):
        return jax.jit(lambda st: lax.scan(inner, st, None, length=STRIDE)[0], donate_argnums=(0,) if donate else ())

    inners = {"ref": GD.inner, "fixed": inner_fixed}
    variants = {"ref": make(GD.inner, False), "donate": make(GD.inner, True),
                "fixed": make(inner_fixed, False), "fixed_donate": make(inner_fixed, True)}
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))

    copy = lambda st: jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), st)
    block = jax.block_until_ready

    def opt_step(adv, st):
        """Production pattern of one optimizer step: 12 x (40 steps, coarse frame)."""
        frames = []
        for _ in range(CALLS):
            st = adv(st)
            frames.append(coarse(st))
        return st, frames

    if args.mode == "nsys":
        import torch                                    # only for cudaProfilerStart/Stop + NVTX ranges
        adv = variants[args.variant]
        st = copy(st0)
        for _ in range(2):                              # compile + warm (outside the capture range)
            st, fr = opt_step(adv, st)
            block((st, fr))
        torch.cuda.profiler.start()
        for r in range(args.reps):
            torch.cuda.nvtx.range_push(f"{args.variant}_optstep_{r}")
            st, fr = opt_step(adv, st)
            block((st, fr))
            torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        print(f"PROFILE_OK nsys {args.variant}: {args.reps} x {CALLS} x {STRIDE} steps captured")
        return

    res = {"device": str(dev), "device_kind": dev.device_kind, "jax": jax.__version__, "init": init,
           "checkpoint": None if ckpt is None else str(ckpt), "completed_blocks": completed, "stride": STRIDE,
           "calls_per_optstep": CALLS,
           "import_s": round(t_import, 2), "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")}

    # ---- compile + optimized HLO + cost analysis of one step --------------------------------------------------
    res["compile_s"], res["hlo"], res["cost_one_step"] = {}, {}, {}
    pats = {"fft": r"\bfft\(", "conditional": r"\bconditional\(", "while": r"\bwhile\(", "copy": r"\bcopy\(",
            "fusion": r"\bfusion\(", "custom_call": r"custom-call\(", "send_recv_host": r"(send|recv|outfeed|infeed)\("}
    for name, fn in variants.items():
        t0 = time.time()
        comp = fn.lower(st0).compile()
        res["compile_s"][name] = round(time.time() - t0, 2)
        txt = comp.as_text()
        with open(os.path.join(args.out, f"hlo_{name}.txt"), "w") as fh:
            fh.write(txt)
        res["hlo"][name] = {k: len(re.findall(p, txt)) for k, p in pats.items()}
    for name, inner in inners.items():
        one = jax.jit(lambda st, inner=inner: inner(st, None)[0]).lower(st0).compile()
        ca = one.cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        res["cost_one_step"][name] = {k: float(ca[k]) for k in ("flops", "bytes accessed") if k in ca}

    # ---- bitwise equivalence after 12 calls (480 steps) --------------------------------------------------------
    ref_out = block(opt_step(variants["ref"], copy(st0))[0])
    ref_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(ref_out)]
    res["bitwise_vs_ref_480_steps"] = {}
    for name in ("donate", "fixed", "fixed_donate"):
        out = block(opt_step(variants[name], copy(st0))[0])
        leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]
        eq = all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(ref_leaves, leaves))
        dq = float(np.max(np.abs(np.asarray(out.state.qh) - np.asarray(ref_out.state.qh))))
        res["bitwise_vs_ref_480_steps"][name] = {"equal": bool(eq), "max_abs_dqh": dq}

    # ---- does a call block the Python thread? (dispatch vs completion of one optimizer step) -------------------
    res["dispatch"] = {}
    for name in ("ref", "fixed"):
        adv, st = variants[name], copy(st0)
        st, fr = opt_step(adv, st)                       # warm
        block((st, fr))
        per_call = []
        t0 = time.perf_counter()
        for _ in range(CALLS):
            c0 = time.perf_counter()
            st = adv(st)
            fr = coarse(st)
            per_call.append(time.perf_counter() - c0)
        t_disp = time.perf_counter() - t0
        block((st, fr))
        t_done = time.perf_counter() - t0
        res["dispatch"][name] = {"dispatch_all_12_s": round(t_disp, 4), "complete_12_s": round(t_done, 4),
                                 "dispatch_fraction": round(t_disp / t_done, 3),
                                 "per_call_dispatch_ms": [round(1e3 * x, 2) for x in per_call]}

    # ---- steady throughput ------------------------------------------------------------------------------------
    def bench(fn, with_coarse):
        st = copy(st0)
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

    res["ms_per_step"] = {name: round(bench(fn, False), 4) for name, fn in variants.items()}
    res["ms_per_step_with_coarse"] = {name: round(bench(variants[name], True), 4) for name in ("ref", "fixed")}
    st = copy(st0)
    block(coarse(st))
    t0 = time.perf_counter()
    for _ in range(50):
        f = coarse(st)
    block(f)
    res["coarse_ms"] = round(1e3 * (time.perf_counter() - t0) / 50, 3)
    res["optstep_solver_s"] = {name: round(res["ms_per_step_with_coarse"][name] * CALLS * STRIDE / 1e3, 3)
                               for name in res["ms_per_step_with_coarse"]}
    for name, ca in res["cost_one_step"].items():
        if "bytes accessed" in ca:
            ms = res["ms_per_step"][name]
            ca["effective_GBps"] = round(ca["bytes accessed"] / (ms * 1e-3) / 1e9, 1)
    ms = dev.memory_stats() or {}
    res["jax_memory_GiB"] = {k: round(ms[k] / 2**30, 2) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
                             if k in ms}
    res["loaded_cuda_libs"] = loaded_libs()

    with open(os.path.join(args.out, "profile_solver_bench.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"device {dev.device_kind}, jax {jax.__version__}, compile {res['compile_s']}")
    print(f"HLO per 40-step call: {res['hlo']}")
    print(f"one-step cost: {res['cost_one_step']}")
    print(f"bitwise vs ref (480 steps): {res['bitwise_vs_ref_480_steps']}")
    print(f"dispatch: " + ", ".join(f"{k}: {v['dispatch_all_12_s']}s of {v['complete_12_s']}s"
                                   for k, v in res["dispatch"].items()))
    print(f"ms/step {res['ms_per_step']}  with coarse {res['ms_per_step_with_coarse']}  coarse {res['coarse_ms']} ms")
    print(f"solver per optimizer step (480 steps + 12 coarse): {res['optstep_solver_s']} s")
    print(f"JAX memory {res['jax_memory_GiB']}; cuda libs: {[os.path.basename(x) for x in res['loaded_cuda_libs']]}")
    print(f"PROFILE_OK bench -> {args.out}/profile_solver_bench.json")


if __name__ == "__main__":
    main()
