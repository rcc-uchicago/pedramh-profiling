"""profile_lean_nsight.py -- the program run_nsight_lean_polaris.pbs runs: checks and timing of the solver variants,
and the steady-state window that nsys / Nsight Compute capture. One A100, from the repo root, after the SQG env bootstrap.

  "$PY" SQG/analysis/opt/nsight/profile_lean_nsight.py --mode bench --out <dir> [--variants ref,lean32,...]
  "$PY" SQG/analysis/opt/nsight/profile_lean_nsight.py --mode capture --variant lean32 --reps 2 --calls 12 --out <dir>

Variants: each a jitted 40-step call AB3State -> AB3State, all from the random IC at AB3 level 2 (the production
spin-up IC stepped twice, as SQG_INIT=random in jobs A/E/F and seed2_opt.py):
  ref        o1.2's call, jax.jit(lambda st: lax.scan(GD.inner, st, None, length=40)[0])
  lean_u1    seed 1's float64 lean step + the cos trigger (lean_solver.make_advance(unroll1=True)); bitwise = ref (7815637)
  lean32     seed 2's float32 step (seed2_opt.make_variant("lean32")); passed round 1b's gate (7815096)
  lean32_r2, lean32_u1, lean32_ring3, lean32_u1_ring3   exact rewrites of lean32 (lean32_probes.py)
--mode bench
  CHECK, fixed before the job and never loosened: each probe vs seed 2's lean32 after 1 call (40 steps, the timed
  code) from the random IC. PASS = every field finite, tc exact, |t - t_ref| <= 1e-6 |t_ref|, relative L2 of each
  AB3 field (theta_hat, both tendencies) <= 1e-6, and finite after 480 steps. Bitwise equality after 40 and 480 steps
  is information. Prints NSIGHT_CHECK_OK <probe> ... or ERROR NSIGHT_CHECK_FAIL <probe> <reasons>.
  BENCH: ms/step of every variant, profile_solver.py's method (median of 3 x 60 calls after 3 warm-up calls), and
  each variant's optimized HLO (hlo_<v>.txt) with its op counts (seed2_opt.hlo_stats). Writes nsight_bench.json;
  prints NSIGHT_BENCH_OK.
  Options for the cuFFT-version arm (the job reruns this mode with LD_PRELOAD of a newer libcufft.so.11):
    --fft-micro          standalone 4096^2 FFT costs (seed2_opt's table: C2C / C2R / R2C in f64 and f32), ms per call
    --save-frames DIR    o1.2's coarse() frame (256^2 float32) after 40 and 480 steps of ref / lean_u1 / lean32
    --compare-frames DIR relative L2 / bitwise of those frames against the ones another run saved in DIR
  The libcufft / libcudart files actually mapped into the process are recorded (loaded_cuda_libs).
--mode capture
  compile + 2 optimizer-equivalents (12 calls + coarse) of warm-up outside the profiled range, then cudaProfilerStart,
  --reps NVTX ranges "<variant>_rep_<r>" of --calls calls each (no coarse; each range ends with block_until_ready),
  cudaProfilerStop. Prints NSIGHT_CAPTURE_OK. Under a profiler the wall times printed are not timings.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")      # as production (JAX shares the GPU with torch)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import math                                                          # noqa: E402
import statistics                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
import traceback                                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(os.path.dirname(HERE))                    # SQG/analysis
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)
import profile_solver as PS                                          # noqa: E402  (import_asqg, STRIDE, CALLS, M)
import lean32_probes as LP                                           # noqa: E402  (imports seed2_opt and lean_solver)

S2, LS = LP.S2, LP.LS
CHECK_REL, T_REL_TOL = 1e-6, 1e-6                                    # pre-set; see the docstring
ALL = ("ref", "lean_u1", "lean32") + tuple(LP.PROBES)
LEAVES = ("theta_hat", "T_-1", "T_-2")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    return x


def build(GD, pst, name, c32):
    """(adv, hlo): adv(st) -> st 40 steps later (jitted); hlo(st) -> the optimized HLO text of that call."""
    import jax
    from jax import lax
    N, NK, DT, STRIDE = GD.N, GD.N // 2 + 1, GD.DT, PS.STRIDE
    if name == "ref":
        f = jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0])
        return f, (lambda st: f.lower(st).compile().as_text())
    if name == "lean_u1":
        _, ncol = LS.preconditions(GD)
        consts = LS.build_consts(GD, ncol)
        f = LS.make_advance(GD, STRIDE, ncol=ncol, unroll1=True)
    elif name == "lean32":
        f, consts = S2.make_variant("lean32", pst, N, NK, DT, STRIDE)[0], c32
    elif name in LP.PROBES:
        f, consts = LP.make_call(pst, N, NK, DT, STRIDE, **LP.PROBES[name]), c32
    else:
        raise SystemExit(f"ERROR NSIGHT_BAD_VARIANT {name} (one of {', '.join(ALL)})")
    return (lambda st: f(st, consts)), (lambda st: f.lower(st, consts).compile().as_text())


def main():
    t_wall = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "capture"], required=True)
    ap.add_argument("--variants", default=",".join(ALL), help="bench: variants to time (probes are also checked)")
    ap.add_argument("--variant", default="lean32", help="capture: the one variant")
    ap.add_argument("--calls", type=int, default=60, help="bench: calls per measurement; capture: calls per range")
    ap.add_argument("--repeats", type=int, default=3, help="bench: measurements (median reported)")
    ap.add_argument("--reps", type=int, default=2, help="capture: NVTX ranges")
    ap.add_argument("--fft-micro", action="store_true", help="bench: standalone FFT cost table")
    ap.add_argument("--save-frames", default="", help="bench: write coarse frames of ref/lean_u1/lean32 here")
    ap.add_argument("--compare-frames", default="", help="bench: compare coarse frames with those saved here")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from pyqg_jax import steppers as pst
    t_import = time.time() - t0
    block = jax.block_until_ready
    copy = lambda st: jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), st)       # noqa: E731
    N, STRIDE, CALLS, M = GD.N, PS.STRIDE, PS.CALLS, PS.M
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    st0 = GD.advance(template, 2)
    if int(st0._ablevel) != 2:
        raise SystemExit("ERROR NSIGHT_IC_NOT_LEVEL2")
    c32 = LP.tables32(GD)
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    dev = jax.devices()[0]

    if args.mode == "capture":
        import torch                                                 # only for cudaProfilerStart/Stop + NVTX ranges
        adv, _ = build(GD, pst, args.variant, c32)
        st = copy(st0)
        t0 = time.time()
        for _ in range(2):                                           # 2 optimizer-equivalents, outside the range
            frames = []
            for _ in range(CALLS):
                st = adv(st)
                frames.append(coarse(st))
            block((st, frames))
        t_warm = time.time() - t0
        torch.cuda.profiler.start()
        t0 = time.time()
        for r in range(args.reps):
            torch.cuda.nvtx.range_push(f"{args.variant}_rep_{r}")
            for _ in range(args.calls):
                st = adv(st)
            block(st)
            torch.cuda.nvtx.range_pop()
        t_prof = time.time() - t0
        torch.cuda.profiler.stop()
        info = {"device_kind": dev.device_kind, "jax": jax.__version__, "variant": args.variant, "reps": args.reps,
                "calls_per_rep": args.calls, "steps_per_rep": args.calls * STRIDE, "import_s": round(t_import, 1),
                "warm_s": round(t_warm, 1), "profiled_s_under_profiler": round(t_prof, 2)}
        with open(os.path.join(args.out, f"capture_{args.variant}_{os.getpid()}.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        print(f"NSIGHT_CAPTURE_OK {args.variant} reps={args.reps} calls={args.calls} "
              f"steps_per_rep={args.calls * STRIDE} warm={t_warm:.0f}s")
        return

    # ---- bench: build, compile, HLO ----------------------------------------------------------------------------
    names = [v for v in args.variants.split(",") if v]
    bad = [v for v in names if v not in ALL]
    if bad:
        raise SystemExit(f"ERROR NSIGHT_BAD_VARIANT {bad}")
    probes = [v for v in names if v in LP.PROBES]
    if probes and "lean32" not in names:
        names.insert(0, "lean32")                                    # the probes' reference
    res = {"device_kind": dev.device_kind, "jax": jax.__version__, "solver_sha256": GD.CONFIG["solver_sha256"],
           "import_s": round(t_import, 1), "stride": STRIDE, "check_rule": {"rel_l2_max_40": CHECK_REL,
           "t_rel_tol": T_REL_TOL, "reference": "seed2_opt lean32", "start": "random IC, AB3 level 2"},
           "variants": {}}
    advs = {}
    for name in names:
        vr = res["variants"][name] = {}
        try:
            adv, hlo = build(GD, pst, name, c32)
            t0 = time.time()
            txt = hlo(st0)
            vr["compile_s"] = round(time.time() - t0, 1)
            with open(os.path.join(args.out, f"hlo_{name}.txt"), "w") as fh:
                fh.write(txt)
            vr["hlo"] = S2.hlo_stats(txt)
            advs[name] = adv
        except Exception as exc:
            vr["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            vr["traceback"] = traceback.format_exc()[-4000:]
            print(f"ERROR NSIGHT_BUILD_FAIL {name} {type(exc).__name__}", flush=True)
    print(f"nsight bench: {dev.device_kind}, jax {jax.__version__}, import {t_import:.0f} s; built "
          + ", ".join(f"{n} (compile {res['variants'][n].get('compile_s')} s, full-size copies "
                      f"{res['variants'][n].get('hlo', {}).get('full_size_copies')})" for n in advs), flush=True)

    # ---- check: probes vs seed 2's lean32 ------------------------------------------------------------------------
    def fields(st):
        return (st.state.qh, st._updates[0].qh, st._updates[1].qh)

    @jax.jit
    def compare_dev(a, b):
        rows = []
        for x, y in zip(fields(a), fields(b)):
            d = x - y
            bits = lambda z: (lax.bitcast_convert_type(jnp.real(z), jnp.uint64),            # noqa: E731
                              lax.bitcast_convert_type(jnp.imag(z), jnp.uint64))
            (xr, xi), (yr, yi) = bits(x), bits(y)
            rows.append(jnp.stack([jnp.sum(jnp.abs(d) ** 2), jnp.sum(jnp.abs(y) ** 2),
                                   jnp.all(jnp.isfinite(x)).astype(jnp.float64),
                                   (jnp.all(xr == yr) & jnp.all(xi == yi)).astype(jnp.float64)]))
        meta = jnp.stack([a.tc == b.tc, a._ablevel == b._ablevel, a.t == b.t]).astype(jnp.float64)
        return jnp.stack(rows), meta, jnp.stack([a.t, b.t]).astype(jnp.float64)

    def compare(a, b):
        rows, meta, tt = (np.asarray(x, dtype=np.float64) for x in jax.device_get(compare_dev(a, b)))
        rel = [float(np.sqrt(r[0] / r[1])) if r[1] > 0 else (0.0 if r[0] == 0 else math.inf) for r in rows]
        t_rel = abs(tt[0] - tt[1]) / abs(tt[1]) if tt[1] != 0 else abs(tt[0])
        return {"rel_l2": dict(zip(LEAVES, rel)), "rel_l2_max": max(rel), "finite": bool(rows[:, 2].all()),
                "bitwise": bool(rows[:, 3].all()), "tc_equal": bool(meta[0]), "ablevel_equal": bool(meta[1]),
                "t_equal": bool(meta[2]), "t_rel_diff": float(t_rel)}

    def run(adv, n_calls):
        st = copy(st0)
        for _ in range(n_calls):
            st = adv(st)
        return block(st)

    if probes and "lean32" in advs:
        r40, r480 = run(advs["lean32"], 1), run(advs["lean32"], CALLS)
        for name in probes:
            vr = res["variants"][name]
            if name not in advs:
                print(f"ERROR NSIGHT_CHECK_FAIL {name} not_built", flush=True)
                vr["check_pass"] = False
                continue
            try:
                c40, c480 = compare(run(advs[name], 1), r40), compare(run(advs[name], CALLS), r480)
                reasons = []
                if not c40["finite"] or not c480["finite"]:
                    reasons.append("nonfinite")
                if not (c40["tc_equal"] and c40["ablevel_equal"]):
                    reasons.append("tc_or_ablevel")
                if not c40["t_rel_diff"] <= T_REL_TOL:
                    reasons.append(f"t_rel={c40['t_rel_diff']:.1e}")
                if not c40["rel_l2_max"] <= CHECK_REL:
                    reasons.append(f"rel40={c40['rel_l2_max']:.3e}>{CHECK_REL:g}")
                vr.update({"check_40": c40, "check_480": c480, "check_pass": not reasons, "check_reasons": reasons})
                detail = (f"rel40={c40['rel_l2_max']:.2e} rel480={c480['rel_l2_max']:.2e} "
                          f"bitwise40={c40['bitwise']} bitwise480={c480['bitwise']} t_exact={c40['t_equal']}")
                print(f"{'NSIGHT_CHECK_OK ' + name if not reasons else 'ERROR NSIGHT_CHECK_FAIL ' + name + ' ' + ';'.join(reasons)}"
                      f" | {detail}", flush=True)
            except Exception as exc:
                vr["check_pass"], vr["check_error"] = False, f"{type(exc).__name__}: {exc}"[:2000]
                print(f"ERROR NSIGHT_CHECK_FAIL {name} exception:{type(exc).__name__}", flush=True)
        del r40, r480

    # ---- bench -------------------------------------------------------------------------------------------------
    def bench(adv):
        st = copy(st0)
        for _ in range(3):
            st = adv(st)
        block(st)
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            for _ in range(args.calls):
                st = adv(st)
            block(st)
            times.append(time.perf_counter() - t0)
        return 1e3 * statistics.median(times) / (args.calls * STRIDE), [1e3 * t / (args.calls * STRIDE) for t in times]

    for name in names:
        if name not in advs:
            continue
        vr = res["variants"][name]
        try:
            ms, all_ms = bench(advs[name])
            vr["ms_per_step"], vr["ms_per_step_repeats"] = round(ms, 4), [round(x, 4) for x in all_ms]
        except Exception as exc:
            vr["bench_error"] = f"{type(exc).__name__}: {exc}"[:2000]
            print(f"ERROR NSIGHT_BENCH_FAIL {name} {type(exc).__name__}", flush=True)
    base = res["variants"].get("lean32", {}).get("ms_per_step")
    ref = res["variants"].get("ref", {}).get("ms_per_step")
    for name in names:
        vr = res["variants"][name]
        ms = vr.get("ms_per_step")
        if ms:
            vr["speedup_vs_ref"] = round(ref / ms, 3) if ref else None
            vr["speedup_vs_lean32"] = round(base / ms, 3) if base else None
            print(f"NSIGHT_BENCH {name} ms={ms:.4f} vs_ref={vr['speedup_vs_ref']} vs_lean32={vr['speedup_vs_lean32']} "
                  f"check={vr.get('check_pass', '-')}", flush=True)

    # ---- frames: the same physics under another library build? (o1.2's coarse frame, 40 and 480 steps) --------------
    if args.save_frames or args.compare_frames:
        res["frames"] = {}
        for name in ("ref", "lean_u1", "lean32"):
            if name not in advs:
                continue
            try:
                st = copy(st0)
                fr = {}
                for i in range(1, CALLS + 1):
                    st = advs[name](st)
                    if i in (1, CALLS):
                        fr[f"f{i * STRIDE}"] = np.asarray(coarse(st))
                if args.save_frames:
                    os.makedirs(args.save_frames, exist_ok=True)
                    np.savez(os.path.join(args.save_frames, f"frames_{name}.npz"), **fr)
                if args.compare_frames:
                    other = np.load(os.path.join(args.compare_frames, f"frames_{name}.npz"))
                    cmp = {}
                    for k, a in fr.items():
                        b = other[k].astype(np.float64)
                        cmp[k] = {"rel_l2": float(np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(b)),
                                  "bitwise": bool(np.array_equal(a.view(np.uint32), other[k].view(np.uint32)))}
                    res["frames"][name] = cmp
                    print(f"NSIGHT_FRAMES {name} vs {args.compare_frames}: "
                          + " ".join(f"{k} rel={v['rel_l2']:.2e} bitwise={v['bitwise']}" for k, v in cmp.items()),
                          flush=True)
            except Exception as exc:
                res["frames"][name] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                print(f"ERROR NSIGHT_FRAMES_FAIL {name} {type(exc).__name__}", flush=True)

    # ---- standalone FFT costs (seed2_opt's micro-table) -------------------------------------------------------------
    if args.fft_micro:
        try:
            NK = N // 2 + 1
            r64 = jax.random.normal(jax.random.key(11), (N, N), dtype=jnp.float64)
            c128 = lax.complex(r64, r64[::-1])
            cases = {"fft2_c2c_c128": (jnp.fft.fft2, c128),
                     "irfft2_c2r_c128": (lambda a: jnp.fft.irfft2(a, s=(N, N)), c128[:, :NK]),
                     "rfft2_r2c_f64": (jnp.fft.rfft2, r64),
                     "fft2_c2c_c64": (jnp.fft.fft2, c128.astype(jnp.complex64)),
                     "irfft2_c2r_c64": (lambda a: jnp.fft.irfft2(a, s=(N, N)), c128[:, :NK].astype(jnp.complex64)),
                     "rfft2_r2c_f32": (jnp.fft.rfft2, r64.astype(jnp.float32))}
            micro = {}
            for key, (f, a) in cases.items():
                f = jax.jit(f)
                block(f(a))
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    for _ in range(20):
                        r = f(a)
                    block(r)
                    ts.append(time.perf_counter() - t0)
                micro[key] = round(1e3 * statistics.median(ts) / 20, 4)
            res["fft_ms_per_call_4096"] = micro
            print("NSIGHT_FFT_MICRO " + " ".join(f"{k}={v}" for k, v in micro.items()), flush=True)
            del r64, c128, cases, r
        except Exception as exc:
            res["fft_ms_per_call_4096"] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            print(f"ERROR NSIGHT_FFT_MICRO_FAIL {type(exc).__name__}", flush=True)
    try:
        with open("/proc/self/maps") as fh:
            libs = {ln.split()[-1] for ln in fh if any(x in ln for x in ("libcufft", "libcudart", "libcublas."))}
        res["loaded_cuda_libs"] = sorted(libs)
        print("NSIGHT_LIBS " + " ".join(sorted(os.path.basename(x) + "@" + os.path.dirname(x) for x in libs)),
              flush=True)
    except OSError as exc:
        res["loaded_cuda_libs"] = f"{type(exc).__name__}: {exc}"
    ms_stats = dev.memory_stats() or {}
    res["jax_memory_GiB"] = {k: round(ms_stats[k] / 2**30, 2) for k in ("bytes_in_use", "peak_bytes_in_use")
                             if k in ms_stats}
    res["wall_s"] = round(time.time() - t_wall, 1)
    path = os.path.join(args.out, "nsight_bench.json")
    with open(path, "w") as fh:
        json.dump(jsonable(res), fh, indent=1)
    n_fail = sum(1 for p in probes if not res["variants"][p].get("check_pass"))
    print(f"NSIGHT_BENCH_OK {path} check_failures={n_fail} wall={res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:                                     # one greppable line, then the traceback
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR NSIGHT_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
