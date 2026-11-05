"""delta_fidelity.py -- does a float32 solver damage o1.2's training delta theta(t + 0.02) - theta(t)?

  "$PY" SQG/analysis/opt/delta/delta_fidelity.py --out <dir> [--state-in <npz>] [--state-out <npz>]
One A100, from the repo root, after the SQG env bootstrap (run_delta_polaris.pbs). Gates and predictions:
PREDICTIONS_delta.md, written before the first run.

Variants (40-step calls AB3State -> AB3State, complex128 outside):
  ref        o1.2's call (production float64); used for one bitwise check of lean_u1
  lean_u1    seed 1's float64 lean step, bitwise = ref (7815637): the float64 REFERENCE here, 1.45x faster
  mixed      seed 2: float32 FFTs and products, float64 state / AB3 / damping
  lean32     seed 2: all float32 (round 1b PASS)
  lean32e    lean32 + expm1 damping + summed AB3 terms (lean32_precise.py)
  lean32ek   lean32e + compensated (hi + lo) state (lean32_precise.py)
  f64ic32    CONTROL: lean_u1 (float64) started from the state rounded once to complex64. Its error is what one
             float32 rounding of the state does through the float64 dynamics: the floor any float32 description of
             the state implies.
Phases:
  A  start state. --state-in: an npz written by --state-out. Otherwise seed 2's synthetic broadband field (E(k) ~
     k^-5/3, CFL 0.3), stepped to AB3 level 2, then integrated by lean_u1 for --spinup-t time units (no archived
     checkpoint is staged on Polaris). SPE and CFL are printed every 0.05; abort if CFL > 0.6 or non-finite.
  B  increments (report only): from the start state rounded to complex64 (identical for every variant; lo = 0), the
     1-step and 40-step increments theta_v(n) - theta(0) vs lean_u1's, relative L2 per |k| octave of the 4096^2
     spectrum.
  C  pairs (the gate): J start states S_0..S_{J-1} along lean_u1's trajectory, 0.02 apart (S_{j+1} = lean_u1's end
     state of pair j). From each S_j every variant runs 4,000 steps (o1.2's lag); o1.2's coarse frame (irfft2,
     [::16, ::16], float32, / THETA_SCALE) is kept at lags 40, 400, 1000, 2000, 4000 steps. Delta_v(lag) = F_v(lag) -
     F(0). Metrics vs lean_u1: e_delta = |Delta_v - Delta_ref| / |Delta_ref| (relative L2), overall and per |k| octave
     of the 256^2 frame; e_frame; rms(Delta) in o1.2's units, next to the UViT's error (val MSE ~0.0043, o1.1).
     Pair 0 also runs ref and checks lean_u1's frames against it bitwise.
  D  ms/step of mixed, lean32, lean32e, lean32ek (median of 3 x --bench-calls calls).
GATE (pre-set): at lag 4,000 steps, median over pairs, e_delta(v) <= max(2 x e_delta(f64ic32), 1e-6), overall AND in
every coarse band carrying >= 1% of Delta_ref's energy. Prints DELTA_GATE <v> PASS|FAIL ...; DELTA_DONE last.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import math                                                          # noqa: E402
import statistics                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OPT, "seed1"))
import profile_solver as PS                                          # noqa: E402
import lean32_precise as LP                                          # noqa: E402  (imports seed2_opt, lean32_probes)
import lean_solver as LS                                             # noqa: E402

S2 = LP.S2
THETA_SCALE = 8.0                                                    # o1.2: x = theta / THETA_SCALE
MODEL_VAL_MSE = 0.0043                                               # o1.1 val loss (o1.2 docstring), o1.2's units
LAGS = (1, 10, 25, 50, 100)                                          # ring frames (x 40 steps); o1.2's pair = 100
GATE_FACTOR, GATE_ABS, GATE_BAND_SHARE = 2.0, 1e-6, 0.01             # pre-set; see the docstring
SYN_SEED, SYN_KD, SYN_CFL = 20261104, 300.0, 0.30                    # seed 2's synthetic field
CANDIDATES = ("mixed", "lean32", "lean32e", "lean32ek")


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
    ap.add_argument("--state-in", default="")
    ap.add_argument("--state-out", default="")
    ap.add_argument("--spinup-t", type=float, default=0.5, help="time units of float64 spin-up (no --state-in)")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--bench-calls", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from pyqg_jax import steppers as pst
    block = jax.block_until_ready
    N, NK, DT, STRIDE, M = GD.N, GD.N // 2 + 1, GD.DT, PS.STRIDE, PS.M
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    res = {"device_kind": jax.devices()[0].device_kind, "jax": jax.__version__,
           "solver_sha256": GD.CONFIG["solver_sha256"], "gate": {"factor": GATE_FACTOR, "abs": GATE_ABS,
           "band_share": GATE_BAND_SHARE, "lag_steps": LAGS[-1] * STRIDE}, "theta_scale": THETA_SCALE,
           "model_val_mse": MODEL_VAL_MSE}

    # ---- variants -------------------------------------------------------------------------------------------------
    c32 = LP.tables(GD)
    res["edec_vs_decay_max_abs"] = LP.edec_check(GD, c32)
    _, ncol = LS.preconditions(GD)
    cu1 = LS.build_consts(GD, ncol)
    u1_call = LS.make_advance(GD, STRIDE, ncol=ncol, unroll1=True)
    u1_one = LS.make_advance(GD, 1, ncol=ncol, unroll1=True)
    s2 = {k: S2.make_variant(k, pst, N, NK, DT, STRIDE) for k in ("mixed", "lean32")}
    F0, P, mdl = np.asarray(GD.F0_hat[0]), S2.PATCH, GD.sm.model
    t64 = {"kinv": (mdl.f_0 / mdl.Nb) * jnp.sqrt(mdl.wv2i), "kx": mdl.kk, "ky": mdl.ll, "decay": GD.decay,
           "f0top": jnp.asarray(F0[:P, :P]), "f0bot": jnp.asarray(F0[N - P:, :P]), "tdt": jnp.float32(DT)}
    tab = {"mixed": t64, "lean32": LP.LP.tables32(GD)}               # seed2_opt.main()'s tables for each
    pc = {k: (LP.make_call(pst, N, NK, DT, STRIDE, **LP.VARIANTS[k]), LP.make_call(pst, N, NK, DT, 1, **LP.VARIANTS[k]))
          for k in LP.VARIANTS}
    ref_call = jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0])
    adv = {"lean_u1": lambda st: u1_call(st, cu1), "ref": ref_call}
    one = {"lean_u1": lambda st: u1_one(st, cu1)}
    for k in ("mixed", "lean32"):
        adv[k] = lambda st, k=k: s2[k][0](st, tab[k])
        one[k] = lambda st, k=k: s2[k][1](st, tab[k])
    for k in LP.VARIANTS:
        adv[k] = lambda st, k=k: pc[k][0](st, c32)
        one[k] = lambda st, k=k: pc[k][1](st, c32)
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    diag = jax.jit(GD.diagnostics)

    def with_fields(st, qh, t0, t1, t=None, tc=None):
        u0, u1 = st._updates
        return pst.AB3State(state=st.state.update(qh=qh), t=st.t if t is None else t, tc=st.tc if tc is None else tc,
                            _ablevel=jnp.uint8(2), _updates=(u0.update(qh=t0), u1.update(qh=t1)))

    def round32(st):
        r = lambda a: a.astype(jnp.complex64).astype(jnp.complex128)                       # noqa: E731
        return with_fields(st, r(st.state.qh), r(st._updates[0].qh), r(st._updates[1].qh))

    def spe_cfl(st):
        values, _, umax = jax.device_get(diag(st))
        return float(values[0]), float(umax) * GD.KMAX * DT

    # ---- A: start state -------------------------------------------------------------------------------------------
    t0 = time.time()
    if args.state_in:
        z = np.load(args.state_in)
        st = GD.advance(template, 2)
        st = with_fields(st, jnp.asarray(z["qh"]), jnp.asarray(z["T0"]), jnp.asarray(z["T1"]),
                         jnp.float32(z["t"]), jnp.asarray(z["tc"], dtype=st.tc.dtype))
        res["start"] = {"from": args.state_in}
    else:
        kap = GD.kap
        white = jax.random.normal(jax.random.key(SYN_SEED), (N, N), dtype=jnp.float64)
        amp = jnp.where(kap >= GD.KF, (kap / GD.KF) ** (-4.0 / 3.0) * jnp.exp(-(kap - GD.KF) / SYN_KD), kap / GD.KF)
        amp = jnp.where((kap > 0) & (kap <= GD.KMAX), amp, 0.0)
        th = jnp.fft.rfft2(white) * amp
        u = jnp.fft.irfft2(-1j * GD.ky * th / GD.kapi, s=(N, N))
        v = jnp.fft.irfft2(1j * GD.kx * th / GD.kapi, s=(N, N))
        th = (th * (SYN_CFL / (float(jnp.max(jnp.sqrt(u ** 2 + v ** 2))) * GD.KMAX * DT))).astype(jnp.complex128)
        st = GD.advance(GD.sm.initialize_stepper_state(template.state.update(qh=th[None])), 2)
        del white, amp, th, u, v
        n_calls = int(round(args.spinup_t / (STRIDE * DT)))
        every = max(1, int(round(0.05 / (STRIDE * DT))))
        hist = []
        for i in range(1, n_calls + 1):
            st = adv["lean_u1"](st)
            if i % every == 0 or i == n_calls:
                spe, cfl = spe_cfl(st)
                hist.append({"t_spin": round(i * STRIDE * DT, 4), "spe": spe, "cfl": cfl})
                print(f"DELTA_SPINUP t={i * STRIDE * DT:.3f} SPE={spe:.6g} CFL={cfl:.3f}", flush=True)
                if not (math.isfinite(spe) and cfl <= 0.6):
                    raise SystemExit(f"ERROR DELTA_SPINUP_UNSTABLE t={i * STRIDE * DT:.3f} SPE={spe} CFL={cfl}")
        res["start"] = {"from": "synthetic + lean_u1 spin-up", "spinup_t": args.spinup_t, "history": hist}
    st = block(st)
    if int(st._ablevel) != 2:
        raise SystemExit("ERROR DELTA_NOT_LEVEL2")
    spe0, cfl0 = spe_cfl(st)
    res["start"].update({"spe": spe0, "cfl": cfl0, "seconds": round(time.time() - t0, 1)})
    if args.state_out:
        np.savez(args.state_out, qh=np.asarray(st.state.qh), T0=np.asarray(st._updates[0].qh),
                 T1=np.asarray(st._updates[1].qh), t=np.asarray(st.t), tc=np.asarray(st.tc))
    print(f"DELTA_SPINUP_OK SPE={spe0:.6g} CFL={cfl0:.3f} ({res['start']['seconds']:.0f} s)", flush=True)

    # ---- B: increments per |k| octave (report only) ------------------------------------------------------------------
    bidx = jnp.where(GD.kap >= 1, jnp.floor(jnp.log2(jnp.maximum(GD.kap, 1.0))), 12).astype(jnp.int32).ravel()

    @jax.jit
    def band_dev(x, y, idx):
        num = jax.ops.segment_sum((jnp.abs(x - y) ** 2).ravel(), idx, num_segments=13)
        den = jax.ops.segment_sum((jnp.abs(y) ** 2).ravel(), idx, num_segments=13)
        return num, den

    def band_ratio(num, den):
        out = {f"{2 ** b}-{2 ** (b + 1)}": (float(np.sqrt(num[b] / den[b])) if den[b] > 0 else None) for b in range(11)}
        out["all"] = float(np.sqrt(num[:12].sum() / den[:12].sum()))
        return out

    def bands_full(x, y):
        """relative L2 of x vs y per |k| octave (and over k >= 1)."""
        return band_ratio(*(np.asarray(a, dtype=np.float64) for a in jax.device_get(band_dev(x, y, bidx))))

    def bands_norm_ratio(x, y):
        """|x| / |y| per |k| octave."""
        z = jnp.zeros_like(x)
        nx = np.asarray(jax.device_get(band_dev(x, z, bidx))[0], dtype=np.float64)
        ny = np.asarray(jax.device_get(band_dev(y, z, bidx))[0], dtype=np.float64)
        return band_ratio(nx, ny)

    sr = round32(st)
    res["increments"] = {}
    th0 = sr.state.qh[0]
    ref1, ref40 = one["lean_u1"](sr).state.qh[0] - th0, adv["lean_u1"](sr).state.qh[0] - th0
    res["increments"]["size_1step_over_theta"] = bands_norm_ratio(ref1, th0)
    res["increments"]["size_40step_over_theta"] = bands_norm_ratio(ref40, th0)
    sz = res["increments"]["size_1step_over_theta"]
    print(f"DELTA_INC size of the 1-step increment / theta: all {sz['all']:.2e}, k1-2 {sz['1-2']:.2e}, "
          f"k2-4 {sz['2-4']:.2e}, k8-16 {sz['8-16']:.2e} (float32 rounding of theta: ~6e-8)", flush=True)
    for k in CANDIDATES:
        d1 = one[k](sr).state.qh[0] - th0
        d40 = adv[k](sr).state.qh[0] - th0
        res["increments"][k] = {"1step": bands_full(d1, ref1), "40step": bands_full(d40, ref40)}
        b = res["increments"][k]
        print(f"DELTA_INC {k}: 1-step increment rel err all {b['1step']['all']:.2e}, k1-2 {b['1step']['1-2']:.2e}, "
              f"k2-4 {b['1step']['2-4']:.2e}, k8-16 {b['1step']['8-16']:.2e}; 40-step all {b['40step']['all']:.2e}, "
              f"k1-2 {b['40step']['1-2']:.2e}", flush=True)
    del ref1, ref40, sr, th0

    # ---- C: pairs at o1.2's lag ----------------------------------------------------------------------------------------
    kc = np.fft.fftfreq(N // M, 1.0 / (N // M))
    kr = np.fft.rfftfreq(N // M, 1.0 / (N // M))
    kapc = np.sqrt(kc[:, None] ** 2 + kr[None, :] ** 2)
    cb = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 182)]
    cmask = {f"{a}-{b}": (kapc >= a) & (kapc < b) for a, b in cb}

    def frames(f, s0):
        """o1.2's frames of one 4,000-step run from s0 at LAGS (+ frame 0), as float32 / THETA_SCALE, and the end."""
        out = {0: np.asarray(coarse(s0), dtype=np.float32) / np.float32(THETA_SCALE)}
        s = s0
        for i in range(1, LAGS[-1] + 1):
            s = f(s)
            if i in LAGS:
                out[i] = np.asarray(coarse(s), dtype=np.float32) / np.float32(THETA_SCALE)
        return out, s

    def delta_metrics(fv, fr):
        out = {}
        for lag in LAGS:
            dv = fv[lag].astype(np.float64) - fr[0].astype(np.float64)
            dr = fr[lag].astype(np.float64) - fr[0].astype(np.float64)
            hv, hr = np.fft.rfft2(dv), np.fft.rfft2(dr)
            num, den = np.abs(hv - hr) ** 2, np.abs(hr) ** 2
            band = {n: (float(np.sqrt(num[m].sum() / den[m].sum())) if den[m].sum() > 0 else None)
                    for n, m in cmask.items()}
            share = {n: float(den[m].sum() / den.sum()) for n, m in cmask.items()}
            out[lag * STRIDE] = {
                "e_delta": float(np.linalg.norm(dv - dr) / np.linalg.norm(dr)),
                "e_frame": float(np.linalg.norm(fv[lag].astype(np.float64) - fr[lag]) / np.linalg.norm(fr[lag])),
                "delta_rms": float(np.sqrt(np.mean(dr ** 2))), "band": band, "band_energy_share": share,
                "bitwise": bool(np.array_equal(fv[lag].view(np.uint32), fr[lag].view(np.uint32)))}
        return out

    pairs = []
    s = st
    for j in range(args.pairs):
        tj = time.time()
        fr, s_next = frames(adv["lean_u1"], s)
        pj = {"j": j, "spe": spe_cfl(s)[0]}
        runs = {k: adv[k] for k in CANDIDATES}
        runs["f64ic32"] = adv["lean_u1"]
        if j == 0:
            runs["ref"] = adv["ref"]
        for k, f in runs.items():
            fv, _ = frames(f, round32(s) if k == "f64ic32" else s)
            pj[k] = delta_metrics(fv, fr)
        pairs.append(pj)
        L = LAGS[-1] * STRIDE
        print(f"DELTA_PAIR j={j} ({time.time() - tj:.0f} s) rms(Delta)={pj['lean32'][L]['delta_rms']:.4f}: "
              + " ".join(f"{k}={pj[k][L]['e_delta']:.2e}" for k in list(CANDIDATES) + ["f64ic32"])
              + (f" ref_bitwise={pj['ref'][L]['bitwise']}" if j == 0 else ""), flush=True)
        s = s_next
    res["pairs"] = pairs

    # ---- gate ---------------------------------------------------------------------------------------------------
    L = LAGS[-1] * STRIDE

    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(statistics.median(xs)) if xs else None

    floor = med([p["f64ic32"][L]["e_delta"] for p in pairs])
    share = {n: med([p["lean32"][L]["band_energy_share"][n] for p in pairs]) for n in cmask}
    bands_gated = [n for n, sh in share.items() if sh >= GATE_BAND_SHARE]
    bfloor = {n: med([p["f64ic32"][L]["band"][n] for p in pairs]) for n in bands_gated}
    drms = med([p["lean32"][L]["delta_rms"] for p in pairs])
    model_rel = math.sqrt(MODEL_VAL_MSE) / drms
    res["summary"] = {"lag_steps": L, "delta_rms": drms, "model_rel_error": model_rel, "floor_e_delta": floor,
                      "band_energy_share": share, "bands_gated": bands_gated, "band_floor": bfloor,
                      "ref_bitwise_pair0": pairs[0]["ref"][L]["bitwise"], "variants": {}}
    print(f"DELTA_SUMMARY lag={L} rms(Delta)={drms:.4f} (o1.2 units); UViT val rel error ~{model_rel:.3f}; "
          f"f64ic32 floor e_delta={floor:.2e}; gated bands {bands_gated}; lean_u1 == ref bitwise "
          f"{res['summary']['ref_bitwise_pair0']}", flush=True)
    for k in CANDIDATES:
        e = med([p[k][L]["e_delta"] for p in pairs])
        eb = {n: med([p[k][L]["band"][n] for p in pairs]) for n in cmask}
        fails = []
        if not e <= max(GATE_FACTOR * floor, GATE_ABS):
            fails.append(f"all {e:.2e}>{max(GATE_FACTOR * floor, GATE_ABS):.2e}")
        for n in bands_gated:
            lim = max(GATE_FACTOR * bfloor[n], GATE_ABS)
            if not eb[n] <= lim:
                fails.append(f"k{n} {eb[n]:.2e}>{lim:.2e}")
        res["summary"]["variants"][k] = {"e_delta": e, "band": eb, "pass": not fails, "fails": fails,
                                         "vs_model_error": e / model_rel}
        print(f"DELTA_GATE {k} {'PASS' if not fails else 'FAIL'} e_delta={e:.2e} ({e / model_rel:.1e} of the UViT's "
              f"error) bands " + " ".join(f"k{n}={eb[n]:.1e}" if eb[n] is not None else f"k{n}=-" for n in cmask)
              + (" | " + "; ".join(fails) if fails else ""), flush=True)

    # ---- D: timing -------------------------------------------------------------------------------------------------
    res["ms_per_step"] = {}
    for k in CANDIDATES:
        x = s
        for _ in range(3):
            x = adv[k](x)
        block(x)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(args.bench_calls):
                x = adv[k](x)
            block(x)
            ts.append(time.perf_counter() - t0)
        res["ms_per_step"][k] = round(1e3 * statistics.median(ts) / (args.bench_calls * STRIDE), 4)
    print("DELTA_BENCH " + " ".join(f"{k}={v}" for k, v in res["ms_per_step"].items()), flush=True)

    res["wall_s"] = round(time.time() - t_wall, 1)
    path = os.path.join(args.out, "delta_fidelity.json")
    with open(path, "w") as fh:
        json.dump(jsonable(res), fh, indent=1)
    n_pass = sum(v["pass"] for v in res["summary"]["variants"].values())
    print(f"DELTA_DONE {path} pass={n_pass}/{len(CANDIDATES)} wall={res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR DELTA_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
