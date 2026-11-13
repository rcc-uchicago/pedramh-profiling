"""energy_budget.py -- the MD-style energy-budget test (team_2026-11-06/ANALOGUES.md §2.6): does a solver arm close the
SPE and VIE budgets as float64 does? One A100 per process; run_energy_polaris.pbs starts 4 (one per GPU), then
analyze_energy.py. Predictions and the decision rule: PREDICTIONS_energy.md (written before the first run).

  "$PY" SQG/analysis/opt/energy/energy_budget.py --out <dir> --worker <w> --ckpt <member>:<ckpt_t*.npz>
        [--tau 0.5] [--frames-dir <gate D frames>] [--deadline <epoch s>] [--arms lean_u1,...]

From one archived checkpoint (loaded as o1.2 does, L192-204) each arm runs `tau` time units in 40-step calls, and after
every call `budget()` evaluates, in float64 on the arm's complex128 state (FIELDS, in this order):
  SPE, VIE                 the invariants, as GD.diagnostics (VIE = SPE's sum weighted by 1/|k|)
  P_S, nu_S, mu_S          SPE injection and the two dissipation terms (= GD.diagnostics' P, eps_nu, eps_mu)
  P_V, nu_V, mu_V          the same for VIE
  Q_S, Q_V                 sum lam w Re(conj(theta) T)/N^4 with T the stored newest tendency: the integrating-factor
                           term (the scheme damps the AB3 increment by exp(-lam dt), not by ~1 - lam dt / 2)
  B_S, B_V                 sum lam^2 a: Q's steady-state-balance form
  A_S, A_V                 sum (lam_eff - lam) a for float32(decay) (dquant's and lean32's table; mechanism A)
  E_S, E_V                 the same for lean32ek's float32 e = -expm1(-lam dt) table
lam = NU k^2 + MU / k^2 inside |k| <= KMAX (decay = 0 outside, where theta is 0). Arms (PREDICTIONS_energy.md):
lean_u1 (reference), lean32ek_fast (candidate), f64ic32 (null), dquant (calibration), lean32 (positive control).
Checks: lean_u1's o1.2 frames at CHECK_CALLS against gate D's stored production frames (read only), and budget() against
GD.diagnostics + the CFL number every DIAG_EVERY calls. --deadline stops an arm early (and skips the rest) so the
analysis still runs inside the walltime. Writes budget_w<w>_<arm>.npz (vals: (calls + 1, len(FIELDS)) float64) and
worker_<w>.json; prints ENERGY_WORKER_DONE last.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse                                                      # noqa: E402
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import math                                                          # noqa: E402
import re                                                            # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
for p in (os.path.join(OPT, "seed1"), os.path.join(OPT, "delta"), os.path.join(OPT, "taxonomy"), ANALYSIS):
    sys.path.insert(0, p)
import profile_solver as PS                                          # noqa: E402  (import_asqg, STRIDE, M)
import lean32_precise as LP                                          # noqa: E402  (tables, S2.make_variant, LP.tables32)
import lean_solver as LS                                             # noqa: E402  (lean_u1)
import lean32_fast as LF                                             # noqa: E402  (lean32ek_fast)

ARMS = ("lean_u1", "lean32ek_fast", "f64ic32", "dquant", "lean32")   # run order: reference and candidate first
# 2026-11-13 (PREDICTIONS_gates_tilerows.md): IF-AB3 arms at 25 steps per 40-step call (h = 8e-6), through
# o12run/tilerows_advance.py; not in the default --arms. CFL and ms/step use each arm's own step.
ARMS_EXTRA = ("tilerows_if25", "if32_25")
ARM_STEPS = {"tilerows_if25": 25, "if32_25": 25}                     # solver steps per call (else STRIDE)
FIELDS = ("SPE", "VIE", "P_S", "nu_S", "mu_S", "P_V", "nu_V", "mu_V",
          "Q_S", "Q_V", "B_S", "B_V", "A_S", "A_V", "E_S", "E_V")
CHECK_CALLS = (0, 1, 10, 100, 1000, 2500)
DIAG_EVERY = 50
THETA_SCALE = 8.0


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    return x


def main():
    t_wall = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--ckpt", required=True, help="<member>:<path to ckpt_t*.npz>")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--frames-dir", default="")
    ap.add_argument("--deadline", type=float, default=0.0, help="epoch seconds; 0 = none")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--warp-path", default="", help="tilerows_if25 only")
    ap.add_argument("--nvmath-path", default="", help="tilerows_if25 only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W = args.worker
    arms = [a for a in args.arms.split(",") if a]
    unknown = sorted(set(arms) - set(ARMS) - set(ARMS_EXTRA))
    if unknown:
        raise SystemExit(f"ERROR ENERGY_UNKNOWN_ARM {unknown}")

    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pyqg_jax import steppers as pst
    block = jax.block_until_ready
    N, NK, DT, STRIDE, M = GD.N, GD.N // 2 + 1, GD.DT, PS.STRIDE, PS.M
    h = STRIDE * DT
    n_calls = int(round(args.tau / h))
    assert abs(n_calls * h - args.tau) < 1e-12 and n_calls % 2 == 0, (n_calls, args.tau)   # Simpson needs even n
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))

    # ---- loading, as o1.2 L192-204 (and taxonomy.py) ---------------------------------------------------------------
    def config_hash(member):
        cfg = dict(GD.CONFIG, member=member, seed_initial_condition=member)
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def load(path, member):
        own, GD.CONFIG_HASH = GD.CONFIG_HASH, config_hash(member)
        try:
            return GD.load_checkpoint(template, path)
        finally:
            GD.CONFIG_HASH = own

    m_s, path = args.ckpt.split(":", 1)
    member = int(m_s)
    if not os.path.exists(path):
        raise SystemExit(f"ERROR ENERGY_CKPT_MISSING {path}")
    t_c = float(re.search(r"ckpt_t([0-9.]+)\.npz", path).group(1))
    st0, completed = load(path, member)
    st0 = block(st0)
    if int(st0._ablevel) != 2:
        raise SystemExit(f"ERROR ENERGY_NOT_LEVEL2 {path}")

    # ---- arms (as taxonomy.py builds them) --------------------------------------------------------------------------
    _, ncol = LS.preconditions(GD)
    cu1 = LS.build_consts(GD, ncol)
    u1_call = LS.make_advance(GD, STRIDE, ncol=ncol, unroll1=True)
    c32 = LP.tables(GD)
    fast_call = LF.make_call(pst, N, NK, DT, STRIDE, mode="ek", u1=True, ring=3)
    l32_call = LP.S2.make_variant("lean32", pst, N, NK, DT, STRIDE)[0]
    tab32 = LP.LP.tables32(GD)
    c_dq = dict(cu1, decay=cu1["decay"].astype(jnp.float32).astype(jnp.float64))
    adv = {"lean_u1": lambda st: u1_call(st, cu1), "lean32ek_fast": lambda st: fast_call(st, c32),
           "f64ic32": lambda st: u1_call(st, cu1), "dquant": lambda st: u1_call(st, c_dq),
           "lean32": lambda st: l32_call(st, tab32)}
    if set(arms) & set(ARMS_EXTRA):
        for p in ("timestep", "tilerows", "o12run"):
            sys.path.insert(0, os.path.join(OPT, p))
        if "tilerows_if25" in arms:
            import site
            sys.path.insert(0, args.warp_path)
            site.addsitedir(args.nvmath_path)
            import warp as wp
            wp.config.enable_backward = False
            wp.init()
        import tilerows_advance as TA
        for a in ARMS_EXTRA:
            if a in arms:
                adv[a] = TA.make(GD, STRIDE, a, rank=W, log=print, gate=False, history=True)

    def round32(st):
        r = lambda a: a.astype(jnp.complex64).astype(jnp.complex128)                       # noqa: E731
        u0, u1 = st._updates
        return pst.AB3State(state=st.state.update(qh=r(st.state.qh)), t=st.t, tc=st.tc, _ablevel=jnp.uint8(2),
                            _updates=(u0.update(qh=r(u0.qh)), u1.update(qh=r(u1.qh))))

    # ---- the budget terms (run-time operands, not closure constants) -----------------------------------------------
    full = lambda x: jnp.broadcast_to(jnp.asarray(x, jnp.float64), (N, NK))                # noqa: E731
    kap, kapi, pos = full(GD.kap), full(GD.kapi), full(GD.pos)
    inside = (kap <= GD.KMAX).astype(jnp.float64)
    lam = (GD.NU * kap ** 2 + GD.MU * pos / kapi ** 2) * inside
    dec32 = jnp.where(inside > 0, c_dq["decay"], 1.0)
    dr32 = jnp.where(inside > 0, -jnp.log(dec32) / DT - lam, 0.0)
    e32 = c32["edec"].astype(jnp.float64)
    drek = jnp.where(inside > 0, -jnp.log1p(-jnp.where(inside > 0, e32, 0.0)) / DT - lam, 0.0)
    bc = {"w": full(GD.w), "kap": kap, "kapi": kapi, "pos": pos, "lam": lam, "dr32": dr32, "drek": drek,
          "F0": jnp.asarray(GD.F0_hat[0])}

    @jax.jit
    def budget(st, b):
        qh = st.state.qh[0]
        T = st._updates[0].qh[0]
        a = b["w"] * jnp.abs(qh / N ** 2) ** 2                                  # as GD.diagnostics
        v = b["pos"] / b["kapi"]                                                # VIE weight 1/|k|, 0 at k = 0
        fr = b["w"] * jnp.real(jnp.conj(qh) * b["F0"]) / N ** 4
        q = b["lam"] * b["w"] * jnp.real(jnp.conj(qh) * T) / N ** 4
        l2a = b["lam"] ** 2 * a
        nu = GD.NU * b["kap"] ** 2 * a
        mu = GD.MU * b["pos"] * a / b["kapi"] ** 2
        return jnp.stack([
            0.5 * jnp.sum(a), 0.5 * jnp.sum(v * a),
            jnp.sum(fr), jnp.sum(nu), jnp.sum(mu),
            jnp.sum(v * fr), jnp.sum(v * nu), jnp.sum(v * mu),
            jnp.sum(q), jnp.sum(v * q), jnp.sum(l2a), jnp.sum(v * l2a),
            jnp.sum(b["dr32"] * a), jnp.sum(v * b["dr32"] * a), jnp.sum(b["drek"] * a), jnp.sum(v * b["drek"] * a)])

    diag = jax.jit(GD.diagnostics)
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    iS = {k: FIELDS.index(k) for k in FIELDS}

    def diag_check(st, row, dt_arm=DT):
        """max relative difference of budget() vs GD.diagnostics (SPE, VIE, eps_nu, eps_mu; P relative to eps), and
        the CFL number at the arm's own step dt_arm."""
        values, _, umax = jax.device_get(diag(st))
        mine = [row[iS["SPE"]], row[iS["VIE"]], row[iS["nu_S"]], row[iS["mu_S"]], row[iS["P_S"]]]
        scale = [abs(values[0]), abs(values[1]), abs(values[2]), abs(values[3]), abs(values[2]) + abs(values[3])]
        rel = max(abs(float(m) - float(v)) / max(s, 1e-300) for m, v, s in zip(mine, values, scale))
        return rel, float(umax) * GD.KMAX * dt_arm

    frames = None
    frames_fn = os.path.join(args.frames_dir, f"m{member:02d}_t{t_c:07.2f}.npy") if args.frames_dir else ""
    if frames_fn and os.path.exists(frames_fn):
        frames = np.load(frames_fn, mmap_mode="r")

    res = {"worker": W, "member": member, "t": t_c, "ckpt": path, "completed_blocks": int(completed),
           "device_kind": jax.devices()[0].device_kind, "jax": jax.__version__,
           "solver_sha256": GD.CONFIG["solver_sha256"], "fields": FIELDS, "h": h, "dt": DT, "stride": STRIDE,
           "n_calls_target": n_calls, "tau_target": args.tau, "frames_file": frames_fn if frames is not None else "",
           "arms": {}, "bitwise_vs_gateD": {}, "deadline": args.deadline}

    row0 = np.asarray(jax.device_get(budget(st0, bc)))
    rel0, cfl0 = diag_check(st0, row0)
    P_S, P_V = row0[iS["P_S"]], row0[iS["P_V"]]
    static = {"P_S": P_S, "P_V": P_V, "eps_S": row0[iS["nu_S"]] + row0[iS["mu_S"]],
              "eps_V": row0[iS["nu_V"]] + row0[iS["mu_V"]], "SPE": row0[iS["SPE"]], "VIE": row0[iS["VIE"]],
              "rIF_S": -0.5 * DT * row0[iS["Q_S"]] / P_S, "rIF_V": -0.5 * DT * row0[iS["Q_V"]] / P_V,
              "rbal_S": -0.5 * DT * row0[iS["B_S"]] / P_S, "rbal_V": -0.5 * DT * row0[iS["B_V"]] / P_V,
              "sA_S": -row0[iS["A_S"]] / P_S, "sA_V": -row0[iS["A_V"]] / P_V,
              "sE_S": -row0[iS["E_S"]] / P_S, "sE_V": -row0[iS["E_V"]] / P_V, "diag_rel": rel0, "cfl": cfl0}
    res["static_at_ic"] = {k: float(v) for k, v in static.items()}
    print(f"ENERGY_IC w={W} member={member} t={t_c:.2f} SPE={static['SPE']:.6g} VIE={static['VIE']:.6g} "
          f"P_S={P_S:.4g} eps_S={static['eps_S']:.4g} P_V={P_V:.4g} eps_V={static['eps_V']:.4g} CFL={cfl0:.3f} "
          f"diag_rel={rel0:.1e}", flush=True)
    print(f"ENERGY_STATIC w={W} rIF S/V {static['rIF_S']:.3e}/{static['rIF_V']:.3e} rbal {static['rbal_S']:.3e}/"
          f"{static['rbal_V']:.3e} sA {static['sA_S']:.3e}/{static['sA_V']:.3e} sE {static['sE_S']:.3e}/"
          f"{static['sE_V']:.3e} (instantaneous, at the IC)", flush=True)

    failures = 0
    stopped = False
    for arm in arms:
        info = {"calls": 0, "seconds": None, "ms_per_step": None, "finite": None, "cfl_max": None,
                "diag_max_rel": None, "error": None, "stopped_by_deadline": False}
        res["arms"][arm] = info
        if stopped:
            info["error"] = "skipped: deadline"
            print(f"ENERGY_ARM w={W} arm={arm} SKIPPED (deadline)", flush=True)
            continue
        try:
            f = adv[arm]
            steps = ARM_STEPS.get(arm, STRIDE)
            dt_arm = STRIDE * DT / steps
            info.update({"steps_per_call": steps, "dt": dt_arm})
            s = round32(st0) if arm == "f64ic32" else st0
            rows = [budget(s, bc)]
            rel_max, cfl_max = 0.0, 0.0
            t0 = time.time()
            s = block(f(s))                                                     # call 1 (includes the compile)
            rows.append(budget(s, bc))
            t1 = time.time()
            done = 1
            if arm == "lean_u1" and frames is not None:
                for k in (0, 1):
                    fr = np.asarray(coarse(st0 if k == 0 else s), dtype=np.float32) / np.float32(THETA_SCALE)
                    res["bitwise_vs_gateD"][str(k)] = bool(np.array_equal(fr.view(np.uint32),
                                                                          frames[k].view(np.uint32)))
            for i in range(2, n_calls + 1):
                s = f(s)
                rows.append(budget(s, bc))
                done = i
                if arm == "lean_u1" and frames is not None and i in CHECK_CALLS:
                    fr = np.asarray(coarse(s), dtype=np.float32) / np.float32(THETA_SCALE)
                    res["bitwise_vs_gateD"][str(i)] = bool(np.array_equal(fr.view(np.uint32),
                                                                          frames[i].view(np.uint32)))
                if i % DIAG_EVERY == 0:
                    rel, cfl = diag_check(s, np.asarray(jax.device_get(rows[-1])), dt_arm)
                    rel_max, cfl_max = max(rel_max, rel), max(cfl_max, cfl)
                    if args.deadline and time.time() > args.deadline and i < n_calls:
                        info["stopped_by_deadline"] = stopped = True
                        print(f"ENERGY_DEADLINE w={W} arm={arm} stopped at call {i} of {n_calls}", flush=True)
                        break
            vals = np.array(jax.device_get(rows), dtype=np.float64)          # no 2,501-operand XLA concat
            t2 = time.time()
            if done % 2:                                                        # keep an even number of intervals
                vals, done = vals[:-1], done - 1
            np.savez(os.path.join(args.out, f"budget_w{W}_{arm}.npz"), vals=vals, fields=np.array(FIELDS))
            info.update({"calls": done, "seconds": round(t2 - t0, 1), "compile_call_s": round(t1 - t0, 1),
                         "ms_per_step": round(1e3 * (t2 - t1) / max((len(rows) - 2) * steps, 1), 4),
                         "finite": bool(np.all(np.isfinite(vals))), "cfl_max": cfl_max, "diag_max_rel": rel_max})
            print(f"ENERGY_ARM w={W} arm={arm} calls={done} seconds={info['seconds']:.0f} "
                  f"ms/step={info['ms_per_step']:.3f} finite={info['finite']} CFLmax={cfl_max:.3f} "
                  f"diag_rel={rel_max:.1e}", flush=True)
        except Exception as exc:                                                # one arm's crash keeps the others
            failures += 1
            first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
            info["error"] = f"{type(exc).__name__}: {first}"
            print(f"ERROR ENERGY_ARM_CRASHED w={W} arm={arm} {info['error']}", flush=True)

    b = res["bitwise_vs_gateD"]
    print(f"ENERGY_BITWISE w={W} lean_u1 vs gate D frames {b if b else 'NOT CHECKED'}", flush=True)
    res["wall_s"] = round(time.time() - t_wall, 1)
    with open(os.path.join(args.out, f"worker_{W}.json"), "w") as fh:
        json.dump(jsonable(res), fh, indent=1)
    ok = sum(1 for a in res["arms"].values() if a["error"] is None)
    print(f"ENERGY_WORKER_DONE w={W} arms={ok}/{len(arms)} failures={failures} wall={res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR ENERGY_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
