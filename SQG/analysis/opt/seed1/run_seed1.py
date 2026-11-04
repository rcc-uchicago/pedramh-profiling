"""run_seed1.py -- seed 1 (float64, JAX level): gate, then time, the lean A-SQG solver step against production.

Inside a compute-node job, from the repo root, after the SQG env bootstrap (SQG_ASQG_PY, SQG_TRAINING_ROOT set):
    "$PY" SQG/analysis/opt/seed1/run_seed1.py --out <dir>
One A100, ~10 min (--budget-min 14: optional phases shrink to fit). Tokens: SEED1_GATE_OK <v> or
ERROR SEED1_GATE_FAIL <v> <reason>; SEED1_BENCH; SEED1_OPTSTEP; SEED1_LONG; SEED1_DONE. Writes <dir>/seed1.json
(rewritten after every phase) and <dir>/hlo_seed1_<v>.txt (optimized HLO of each 40-step call).

Reference: o1.2's call, jax.jit(lambda st: lax.scan(GD.inner, st, None, length=40)[0]), with A-SQG_data.py imported
unmodified, as profile_solver.import_asqg does. Variants (lean_solver.make_advance, float64, same mathematics):
  lean        dead work removed (filter x1, beta term +0, dense F0 read, complex decay, per-step |k|^-1), AB3 fixed at
              level 2, history in a 3-slot ring (no rotation copies)
  lean_u1     lean + an exact x*cos(+-0) factor in both complex128 loop fusions (XLA: 1 element per thread)
  lean_split  lean + an optimization barrier between the tendency and the AB3 update

Gate, fixed before any run (float64-only changes):
  G1  one solver step from each test state: every value finite, and the relative L2 error of each AB3 field
      (theta_hat and both stored tendencies) vs the production step <= 1e-12.
  G2  one 40-step call, the code that is timed, from each test state: finite, relative L2 of each field <= 1e-9, t and
      tc identical, AB3 level 2. It catches a wrong ring rotation, which a 1-step test cannot see; rounding-only
      differences stay near 1e-15 over 40 steps.
  G3  finite after 480 steps (12 calls) from each test state, where the reference is finite.
Information only: bitwise equality after 480 steps; after 4,000 steps from the synthetic state, the relative L2 of the
256^2 coarse field (o1.2's coarse()) and the relative difference of the energy 0.5<theta^2> vs the reference.

Test states: `random` = the production spin-up IC stepped to AB3 level 2 (theta ~ 1e-3, as jobs A/E time; there the
advection term is tiny next to the forcing); `synthetic` = a broadband random-phase theta, |theta_k| ~ k^(-4/3)
exp(-(k/(KMAX/2))^2) on 1 <= |k| <= KMAX, scaled to a spectral CFL max|u| KMAX dt of 0.3 (spin-up reached <= 0.42) and
stepped to level 2 by the production step, so every term of the step is O(1). Timing uses `random`, as jobs A/E did.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # as production (JAX shares the GPU with torch)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # SQG/analysis: profile_solver
sys.path.insert(0, HERE)
import profile_solver as PS  # noqa: E402  (import_asqg, STRIDE, CALLS, M)
import lean_solver as LS  # noqa: E402  (standard library only at import)

TOL_1STEP = 1e-12          # G1, the policy's float64 tolerance
TOL_40STEP = 1e-9          # G2, structural check of the timed call
VARIANTS = {"lean": {}, "lean_u1": {"unroll1": True}, "lean_split": {"split": True}}
FIELDS = ("qh", "T0", "T1")  # theta_hat, _updates[0] (newest tendency), _updates[1]
BITWISE_CALLS = 12         # 480 steps = one optimizer step
LONG_CALLS = 100           # 4,000 steps = one 0.02 training interval
SYN_CFL = 0.3
SYN_SEED = 20261104


def elapsed():
    return time.time() - T_START


def jsonable(x):
    """Plain JSON (no NaN/inf literals, no numpy scalars)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    if hasattr(x, "item"):
        return jsonable(x.item())
    return str(x)


def fields(np, st):
    """Host copy of an AB3State: the three full fields and the bookkeeping scalars."""
    return {"qh": np.asarray(st.state.qh), "T0": np.asarray(st._updates[0].qh), "T1": np.asarray(st._updates[1].qh),
            "t": float(np.asarray(st.t)), "tc": int(np.asarray(st.tc)), "ablevel": int(np.asarray(st._ablevel))}


def rel_l2(np, a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / nb if nb > 0 else float("nan")


def compare(np, cand, ref, forced=None):
    """Per-field relative L2 error, max |error| and where, finiteness, value and bit equality (signed zeros)."""
    out, sq_d, sq_r = {}, 0.0, 0.0
    for k in FIELDS:
        a, b = cand[k], ref[k]
        d = a - b
        nd = float(np.sqrt(np.vdot(d, d).real))
        nr = float(np.sqrt(np.vdot(b, b).real))
        absd = np.abs(d)
        i = int(np.argmax(np.where(np.isnan(absd), np.inf, absd)))
        out[k] = {"rel_l2": nd / nr if nr > 0 else (0.0 if nd == 0 else float("inf")),
                  "max_abs": float(absd.flat[i]), "where": [int(x) for x in np.unravel_index(i, d.shape)],
                  "ref_abs_there": float(np.abs(b.flat[i])), "finite": bool(np.isfinite(a).all()),
                  "values_equal": bool(np.array_equal(a, b)),
                  "bits_equal": bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))}
        sq_d, sq_r = sq_d + nd * nd, sq_r + nr * nr
        del d, absd
    rels = [out[k]["rel_l2"] for k in FIELDS]
    worst = max(range(len(FIELDS)), key=lambda j: math.inf if math.isnan(rels[j]) else rels[j])
    out["max_rel_l2"] = rels[worst]
    out["worst_field"] = FIELDS[worst]
    out["combined_rel_l2"] = math.sqrt(sq_d / sq_r) if sq_r > 0 else float("nan")
    for flag in ("finite", "values_equal", "bits_equal"):
        out[flag] = all(out[k][flag] for k in FIELDS)
    out["t_equal"] = cand["t"] == ref["t"]
    out["tc_equal"] = cand["tc"] == ref["tc"]
    out["ablevel"] = cand["ablevel"]
    if forced is not None:                       # the 24 forced modes: error relative to |F0| there
        idx, fabs = forced
        out["T0_forced_modes_max_err_over_F"] = float((np.abs(cand["T0"][idx] - ref["T0"][idx]) / fabs).max())
    return out


def gate_reason(g):
    """None if the variant passes G1-G3, else a one-word reason."""
    for ic, c in g["1step"].items():
        if not c["finite"]:
            return f"nonfinite_after_1_step_{ic}"
        if not c["max_rel_l2"] <= TOL_1STEP:
            return f"1step_rel_l2={c['max_rel_l2']:.3e}>{TOL_1STEP:g}_in_{c['worst_field']}_{ic}"
    for ic, c in g["40step"].items():
        if not c["finite"]:
            return f"nonfinite_after_40_steps_{ic}"
        if not c["max_rel_l2"] <= TOL_40STEP:
            return f"40step_rel_l2={c['max_rel_l2']:.3e}>{TOL_40STEP:g}_in_{c['worst_field']}_{ic}"
        if not (c["t_equal"] and c["tc_equal"] and c["ablevel"] == 2):
            return f"40step_t_tc_or_ablevel_differ_{ic}"
    for ic, c in g["480step"].items():
        if c["ref_finite"] and not c["finite"]:
            return f"nonfinite_after_480_steps_{ic}"
    if not g["1step"] or not g["40step"] or not g["480step"]:
        return "gate_not_run"
    return None


def synthetic_state(GD, template, cfl=SYN_CFL, seed=SYN_SEED):
    """Broadband random-phase theta at a realistic amplitude (spectral CFL `cfl`), stepped to AB3 level 2 by the
    production step. Not an attractor state: it exercises every term of the step at O(1), nothing more."""
    import jax
    import jax.numpy as jnp
    N, KMAX, DT, kap = GD.N, GD.KMAX, GD.DT, GD.kap
    phase = 2.0 * jnp.pi * jax.random.uniform(jax.random.key(seed), kap.shape, dtype=jnp.float64)
    amp = jnp.where((kap >= 1.0) & (kap <= KMAX),
                    jnp.where(kap > 0, kap, 1.0) ** (-4.0 / 3.0) * jnp.exp(-(kap / (0.5 * KMAX)) ** 2), 0.0)
    th = jnp.fft.irfft2(amp * jnp.exp(1j * phase), s=(N, N))           # real field: Hermitian by construction
    qh = jnp.fft.rfft2(th) * ((kap > 0) & (kap <= KMAX))                # zero mean, inside the 2/3 disc
    u = jnp.fft.irfft2(-1j * GD.ky * qh / GD.kapi, s=(N, N))            # velocities as A-SQG's diagnostics()
    v = jnp.fft.irfft2(1j * GD.kx * qh / GD.kapi, s=(N, N))
    umax = float(jnp.max(jnp.sqrt(u ** 2 + v ** 2)))
    scale = cfl / (umax * KMAX * DT)
    qh = (qh * scale).astype(jnp.complex128)
    st = template.update(state=template.state.update(qh=qh[None]))
    st = GD.advance(st, 2)                                              # levels 0 -> 1 -> 2 via pyqg's lax.switch
    th2 = jnp.fft.irfft2(st.state.qh[0], s=(N, N))
    return st, {"seed": seed, "spectrum": "|theta_k| ~ k^(-4/3) exp(-(k/(KMAX/2))^2), 1 <= k <= KMAX, random phases",
                "cfl": cfl, "umax": umax * scale, "theta_rms": float(jnp.sqrt(jnp.mean(th2 ** 2))),
                "theta_max": float(jnp.max(jnp.abs(th2))), "ablevel": int(st._ablevel)}


def main():
    ap = argparse.ArgumentParser(description="seed 1: gate and time the lean float64 A-SQG solver step")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=60, help="bench: calls per throughput measurement (x40 steps)")
    ap.add_argument("--repeats", type=int, default=3, help="bench: throughput repeats (median reported)")
    ap.add_argument("--optstep-reps", type=int, default=5, help="o1.2 optimizer-step pattern repeats (median)")
    ap.add_argument("--budget-min", type=float, default=14.0, help="wall budget; optional phases shrink to fit")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "seed1.json")
    res = {"seed": 1, "angle": "float64, JAX level, same mathematics",
           "thresholds": {"G1_1step_rel_l2_per_field": TOL_1STEP, "G2_40step_rel_l2_per_field": TOL_40STEP,
                          "G3": "finite after 480 steps where the reference is finite"},
           "phases_s": {}}

    def save():
        res["elapsed_s"] = round(elapsed(), 1)
        with open(out_json, "w") as fh:
            json.dump(jsonable(res), fh, indent=1)

    try:
        run(args, res, save)
    except SystemExit:
        save()
        raise
    except Exception as exc:  # noqa: BLE001  (one greppable line, full traceback in the JSON)
        res["error"] = traceback.format_exc()
        save()
        msg = str(exc).splitlines()[0][:200] if str(exc) else ""
        print(f"ERROR SEED1_HARNESS {type(exc).__name__}: {msg}", flush=True)
        sys.exit(1)
    save()
    print(f"SEED1_DONE -> {out_json} ({elapsed() / 60:.1f} min)", flush=True)


def run(args, res, save):
    budget = 60.0 * args.budget_min
    names = [v for v in args.variants.split(",") if v]
    if [v for v in names if v not in VARIANTS] or not names:
        print(f"ERROR SEED1_UNKNOWN_VARIANT {args.variants} (known: {','.join(VARIANTS)})")
        raise SystemExit(2)
    if not os.path.isfile(PS.ASQG_PY):
        print(f"ERROR SEED1_NO_SOLVER {PS.ASQG_PY} (set SQG_ASQG_PY to SQG/fixed_ring/A-SQG_data.py)")
        raise SystemExit(2)

    # ---- the production solver, imported unmodified ------------------------------------------------------------
    t = time.time()
    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    res["phases_s"]["import"] = round(time.time() - t, 1)
    dev = jax.devices()[0]
    N, M, STRIDE, CALLS = GD.N, PS.M, PS.STRIDE, PS.CALLS
    block = jax.block_until_ready

    def copy(st):
        return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), st)

    res.update(device=dev.device_kind, jax=jax.__version__, N=N, stride=STRIDE, calls_per_optstep=CALLS,
               asqg_py=PS.ASQG_PY, variants={n: VARIANTS[n] for n in names})

    # ---- preconditions: every removed term is exactly 0 or 1 in this configuration ------------------------------
    try:
        pre, ncol = LS.preconditions(GD)
    except LS.PreconditionError as exc:
        res["preconditions"] = {"failed": str(exc)}
        print(f"ERROR SEED1_PRECONDITION {exc}: the lean step does not apply to this configuration")
        raise SystemExit(3)
    res["preconditions"] = pre
    consts = LS.build_consts(GD, ncol)
    F0 = np.asarray(GD.F0_hat)
    fidx = np.nonzero(F0 != 0)
    forced = (fidx, np.abs(F0[fidx]))

    # ---- test states ------------------------------------------------------------------------------------------------
    t = time.time()
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    st_rand = GD.advance(template, 2)
    st_syn, syn = synthetic_state(GD, template)
    states = {"random": st_rand, "synthetic": st_syn}
    for ic, st in states.items():
        if int(st._ablevel) != 2:
            raise RuntimeError(f"test state {ic} is at AB3 level {int(st._ablevel)}, not 2")
    energy = jax.jit(lambda st: 0.5 * jnp.sum(GD.w * jnp.abs(st.state.qh[0] / N ** 2) ** 2))
    syn["energy"] = float(energy(st_syn))
    res["synthetic_ic"] = syn
    res["random_ic_energy"] = float(energy(st_rand))
    res["phases_s"]["states"] = round(time.time() - t, 1)
    print(f"seed1: {dev.device_kind}, jax {jax.__version__}; preconditions OK (forcing in k_x columns < {ncol}); "
          f"synthetic IC: CFL {syn['cfl']:.2f}, theta_rms {syn['theta_rms']:.3g}", flush=True)
    save()

    # ---- the functions -------------------------------------------------------------------------------------------
    ref40 = jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0])     # o1.2's advance
    ref1 = jax.jit(lambda st: GD.inner(st, None)[0])
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))  # o1.2
    finite = jax.jit(lambda st: jnp.isfinite(st.state.qh).all() & jnp.isfinite(st._updates[0].qh).all()
                     & jnp.isfinite(st._updates[1].qh).all())
    calls, ones, jits = {"ref": ref40}, {"ref": ref1}, {}
    for name in names:
        f40 = LS.make_advance(GD, STRIDE, ncol=ncol, **VARIANTS[name])
        f1 = LS.make_advance(GD, 1, ncol=ncol, **VARIANTS[name])
        jits[name] = (f40, f1)
        calls[name] = (lambda st, f=f40: f(st, consts))
        ones[name] = (lambda st, f=f1: f(st, consts))
    failed = {}                                    # variant -> reason (gate failure or exception)

    # ---- compile: optimized HLO of each 40-step call, XLA's cost model of one step -----------------------------
    nk = N // 2 + 1
    pats = {"fft": r"\bfft\(", "copy": r"\bcopy\(", "full_field_copy": rf"c128\[(?:1,)?{N},{nk}\]\{{[0-9,]*\}} copy\(",
            "fusion": r"\bfusion\(", "while": r"\bwhile\(", "conditional": r"\bconditional\(",
            "cosine": r"\bcosine\(", "pad": r"\bpad\(", "custom_call": r"custom-call\("}
    res["compile"] = {}
    for name in ["ref"] + names:
        t = time.time()
        try:
            if name == "ref":
                c40, c1 = ref40.lower(st_rand).compile(), ref1.lower(st_rand).compile()
            else:
                c40 = jits[name][0].lower(st_rand, consts).compile()
                c1 = jits[name][1].lower(st_rand, consts).compile()
        except Exception as exc:  # noqa: BLE001
            if name == "ref":
                raise
            failed[name] = f"compile_error_{type(exc).__name__}"
            res["compile"][name] = {"error": traceback.format_exc()}
            continue
        txt = c40.as_text()
        with open(os.path.join(args.out, f"hlo_seed1_{name}.txt"), "w") as fh:
            fh.write(txt)
        ca = c1.cost_analysis()
        ca = (ca[0] if ca else None) if isinstance(ca, (list, tuple)) else ca
        res["compile"][name] = {"aot_compile_s": round(time.time() - t, 2),
                                "hlo_40step_call": {k: len(re.findall(p, txt)) for k, p in pats.items()},
                                "xla_cost_one_step": {k: float(ca[k]) for k in ("flops", "bytes accessed")
                                                      if ca and k in ca}}
    res["phases_s"]["compile"] = round(sum(v.get("aot_compile_s", 0) for v in res["compile"].values()), 1)
    save()

    # ---- G1 (1 step) and G2 (one timed 40-step call), from both test states ------------------------------------
    t = time.time()
    gates = {name: {"1step": {}, "40step": {}, "480step": {}} for name in names}
    ms_est = {}                                    # ms/step from one 40-step call on the second state (warm)
    for ic, st0 in states.items():
        r1 = fields(np, block(ref1(st0)))
        t0 = time.perf_counter()
        r40s = block(ref40(st0))
        ms_est["ref"] = 1e3 * (time.perf_counter() - t0) / STRIDE
        r40 = fields(np, r40s)
        del r40s
        for name in names:
            if name in failed:
                continue
            try:
                c1 = compare(np, fields(np, block(ones[name](st0))), r1, forced)
                t0 = time.perf_counter()
                v40s = block(calls[name](st0))
                ms_est[name] = 1e3 * (time.perf_counter() - t0) / STRIDE
                c40 = compare(np, fields(np, v40s), r40)
                del v40s
            except Exception as exc:  # noqa: BLE001
                failed[name] = f"exception_{type(exc).__name__}"
                gates[name]["exception"] = traceback.format_exc()
                continue
            gates[name]["1step"][ic], gates[name]["40step"][ic] = c1, c40
        del r1, r40
    res["ms_per_step_estimate_one_call"] = {k: round(v, 3) for k, v in ms_est.items()}

    # ---- G3 + bitwise information: 480 steps (12 calls) from both test states ----------------------------------
    for ic, st0 in states.items():
        rs = st0
        for _ in range(BITWISE_CALLS):
            rs = ref40(rs)
        rf = fields(np, block(rs))
        ref_finite = bool(all(np.isfinite(rf[k]).all() for k in FIELDS))
        rc, rE = np.asarray(coarse(rs)), float(energy(rs))
        del rs
        for name in names:
            if name in failed:
                continue
            try:
                vs = st0
                for _ in range(BITWISE_CALLS):
                    vs = calls[name](vs)
                c = compare(np, fields(np, block(vs)), rf)
                c.update(ref_finite=ref_finite, coarse_rel_l2=rel_l2(np, coarse(vs), rc),
                         energy_rel_diff=float(energy(vs)) / rE - 1.0)
                del vs
            except Exception as exc:  # noqa: BLE001
                failed[name] = f"exception_{type(exc).__name__}"
                gates[name]["exception"] = traceback.format_exc()
                continue
            gates[name]["480step"][ic] = c
        del rf
    res["phases_s"]["gate"] = round(time.time() - t, 1)

    passed = []
    for name in names:
        reason = failed.get(name) or gate_reason(gates[name])
        if reason:
            failed[name] = reason
            print(f"ERROR SEED1_GATE_FAIL {name} {reason}", flush=True)
            continue
        passed.append(name)
        g = gates[name]
        print(f"SEED1_GATE_OK {name} 1step_max_rel_l2={max(c['max_rel_l2'] for c in g['1step'].values()):.2e} "
              f"40step_max_rel_l2={max(c['max_rel_l2'] for c in g['40step'].values()):.2e} "
              + " ".join(f"bitwise480_{ic}={c['bits_equal']}" for ic, c in g["480step"].items()), flush=True)
    res["gate"] = {name: {"pass": name in passed, "reason": failed.get(name), **gates[name]} for name in names}
    save()

    # ---- throughput: profile_solver.py's method (median of repeats x calls 40-step calls, from `random`) --------
    def bench(fn, st0, ncalls, repeats, with_coarse=False):
        st = copy(st0)
        for _ in range(3):
            st = fn(st)
        block(st)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            frames = []
            for _ in range(ncalls):
                st = fn(st)
                if with_coarse:
                    frames.append(coarse(st))
            block((st, frames))
            times.append(time.perf_counter() - t0)
        return 1e3 * statistics.median(times) / (ncalls * STRIDE), [1e3 * x / (ncalls * STRIDE) for x in times]

    def opt_step(fn, st):
        """o1.2's pattern for one optimizer step: 12 x (40 solver steps, coarse frame)."""
        frames = []
        for _ in range(CALLS):
            st = fn(st)
            frames.append(coarse(st))
        return st, frames

    bench_names = [n for n in names if not str(failed.get(n, "")).startswith(("compile_error", "exception"))]
    timed = ["ref"] + bench_names
    est = {n: ms_est.get(n, ms_est.get("ref", 10.0)) for n in timed}
    best_guess = min([est[n] for n in bench_names if n in passed] or [est["ref"]])
    optstep_need = (args.optstep_reps + 1) * CALLS * STRIDE * (est["ref"] + best_guess) / 1e3
    per_call = STRIDE * sum(est.values()) / 1e3
    room = 0.80 * budget - elapsed() - optstep_need - 30.0
    ncalls = max(10, min(args.calls, int((room / per_call - 3) / args.repeats)))
    t = time.time()
    res["bench"] = {"calls": ncalls, "repeats": args.repeats, "init": "random", "ms_per_step": {}, "repeats_ms": {}}
    if ncalls < args.calls:
        print(f"WARN SEED1_BENCH_SHORTENED calls {args.calls} -> {ncalls} to fit --budget-min {args.budget_min}")
    for name in timed:
        ms, reps = bench(calls[name], st_rand, ncalls, args.repeats)
        res["bench"]["ms_per_step"][name], res["bench"]["repeats_ms"][name] = round(ms, 4), [round(x, 4) for x in reps]
        if name != "ref":
            ms_ref = res["bench"]["ms_per_step"]["ref"]
            res["bench"].setdefault("speedup", {})[name] = round(ms_ref / ms, 4)
            print(f"SEED1_BENCH {name} ms_ref={ms_ref:.4f} ms_cand={ms:.4f} speedup={ms_ref / ms:.3f} "
                  f"gate={'OK' if name in passed else 'FAIL'}", flush=True)
        save()
    res["phases_s"]["bench"] = round(time.time() - t, 1)

    # ---- one o1.2 optimizer step (12 calls + coarse) for the best passing variant ------------------------------
    ms_b = res["bench"]["ms_per_step"]
    best = min(passed, key=lambda n: ms_b[n]) if passed else None
    res["best_passing_variant"] = best
    if best is not None:
        t = time.time()
        reps = args.optstep_reps
        per_rep = CALLS * STRIDE * (ms_b["ref"] + ms_b[best]) / 1e3
        if elapsed() + (reps + 1) * per_rep > 0.90 * budget:
            reps = max(2, int((0.90 * budget - elapsed()) / per_rep) - 1)
        out = {}
        for name in ("ref", best):
            st = copy(st_rand)
            st, fr = opt_step(calls[name], st)              # warm
            block((st, fr))
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                st, fr = opt_step(calls[name], st)
                block((st, fr))
                times.append(time.perf_counter() - t0)
            out[name] = {"s_median": round(statistics.median(times), 4), "s_all": [round(x, 4) for x in times]}
        out["speedup"] = round(out["ref"]["s_median"] / out[best]["s_median"], 4)
        out.update(variant=best, reps=reps)
        res["optstep"] = out
        res["phases_s"]["optstep"] = round(time.time() - t, 1)
        print(f"SEED1_OPTSTEP {best} s_ref={out['ref']['s_median']:.3f} s_cand={out[best]['s_median']:.3f} "
              f"speedup={out['speedup']:.3f} (12 x 40 steps + coarse, median of {reps})", flush=True)
        save()
    else:
        print("WARN SEED1_NO_PASSING_VARIANT: optimizer-step timing skipped")

    # ---- information: 4,000 steps from the synthetic state (chaotic: judged statistically, not pointwise) -------
    t = time.time()
    long_names = sorted(bench_names, key=lambda n: (n not in passed, ms_b.get(n, 1e9)))
    res["long"] = {"steps": LONG_CALLS * STRIDE, "init": "synthetic", "energy_t0": syn["energy"], "variants": {}}
    need_ref = LONG_CALLS * STRIDE * ms_b["ref"] / 1e3
    if not long_names or elapsed() + need_ref + LONG_CALLS * STRIDE * ms_b[long_names[0]] / 1e3 > 0.97 * budget:
        print("WARN SEED1_LONG_SKIPPED: no time left in --budget-min")
        res["long"]["skipped"] = "budget"
    else:
        rs = st_syn
        for _ in range(LONG_CALLS):
            rs = ref40(rs)
        rs = block(rs)
        rc, rE, rq, rfin = np.asarray(coarse(rs)), float(energy(rs)), np.asarray(rs.state.qh), bool(finite(rs))
        del rs
        res["long"].update(ref_energy=rE, ref_energy_change_from_t0=rE / syn["energy"] - 1.0, ref_finite=rfin)
        for name in long_names:
            if elapsed() + LONG_CALLS * STRIDE * ms_b[name] / 1e3 > 0.97 * budget:
                res["long"]["variants"][name] = {"skipped": "budget"}
                continue
            vs = st_syn
            for _ in range(LONG_CALLS):
                vs = calls[name](vs)
            vs = block(vs)
            vfin = bool(finite(vs))
            q = np.asarray(vs.state.qh)
            d = q - rq
            info = {"finite": vfin, "coarse_rel_l2": rel_l2(np, coarse(vs), rc),
                    "energy": float(energy(vs)), "qh_rel_l2": float(np.sqrt(np.vdot(d, d).real / np.vdot(rq, rq).real))}
            info["energy_rel_diff"] = info["energy"] / rE - 1.0
            del vs, q, d
            res["long"]["variants"][name] = info
            print(f"SEED1_LONG {name} steps={LONG_CALLS * STRIDE} coarse_rel_l2={info['coarse_rel_l2']:.3e} "
                  f"energy_rel_diff={info['energy_rel_diff']:+.3e} qh_rel_l2={info['qh_rel_l2']:.3e} "
                  f"finite={vfin}", flush=True)
            if rfin and not vfin:
                print(f"ERROR SEED1_LONG_NONFINITE {name}: non-finite after {LONG_CALLS * STRIDE} steps, "
                      "the reference is finite", flush=True)
            save()
    res["phases_s"]["long"] = round(time.time() - t, 1)

    # ---- drift check: the reference timed again at the end, if there is time -----------------------------------
    need = (3 + ncalls * args.repeats) * STRIDE * ms_b["ref"] / 1e3
    if elapsed() + need < 0.97 * budget:
        ms, reps = bench(calls["ref"], st_rand, ncalls, args.repeats)
        res["bench"]["ms_per_step_ref_again"] = round(ms, 4)
        res["bench"]["ref_drift_pct"] = round(100.0 * (ms / ms_b["ref"] - 1.0), 2)
    else:
        res["bench"]["ms_per_step_ref_again"] = "skipped (budget)"
    mem = dev.memory_stats() or {}
    res["jax_memory_GiB"] = {k: round(mem[k] / 2 ** 30, 2) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
                             if k in mem}


if __name__ == "__main__":
    main()
