"""taxonomy.py -- item 3 of the 2026-11-06 team plan: is each faster solver's error in o1.2's 0.02 delta NOISE or BIAS,
band by band, on ARCHIVED states? One A100 per process; run_taxonomy_polaris.pbs starts 4 (one per GPU), then
analyze_taxonomy.py. Predictions and gates: PREDICTIONS_taxonomy.md (written before the first run).

  "$PY" SQG/analysis/opt/taxonomy/taxonomy.py --out <dir> --worker <w> --ckpt <member>:<npz> [--ckpt ...]

States: archived A-SQG_data.py checkpoints (ckpt_t*.npz, full float64 AB3 state at level 2), loaded with o1.2's own
recipe (GD.load_checkpoint under the member's CONFIG_HASH; o1.2 L192-204). From each, PAIRS consecutive 0.02 pairs
along float64 lean_u1 (bitwise production, job 7815637): S_{j+1} = lean_u1's state 4,000 steps after S_j.

Arms (each a 40-step call AB3State -> AB3State, run 100 times from S_j; f64ic32 from S_j rounded once to complex64):
  every pair       lean_u1 (the reference), lean32ek, inj (lean32ek with the planted damping-rate shift below)
  first pair       dquant (control a: lean_u1 with only its decay table rounded to float32 = mechanism A alone, in
  of each state    float64), inj64 (lean_u1 with the same planted shift: the shift's float64 response, the truth for
                   control b), lean32 (all float32), f64ic32 (the floor), lean32sr (stochastic rounding, lean32_fast.py)
  first state      prod (o1.2's call: lean_u1 == production bitwise) and lean32ek_fast (lean32_fast.py: must equal
                   lean32ek bit for bit at 40 / 400 / 4,000 steps)
  first pair of    the dt ladder (dt_ladder.py): h = 2e-4/n for n = 40 (control: == lean_u1 bitwise), 30, 32, 80
  DT_STATES states
The planted shift (control b): the damping rate of every 4096^2 mode inside the alias-free square (|k_x|, |k_y| <= 127)
is raised by drate_b, set per coarse band b from this worker's first reference pair so that its static estimate
drate_b * 0.02 * rms(x_b) / rms(Delta_b) = INJ_FACTOR x MDE_PRED_b (MDE_PRED from the team's R4 power analysis).
Frames, as o1.2 builds them: `sub` = float32(irfft2(theta_hat)[::16, ::16]) / 8 (256^2); and `af`, the alias-free
coarse spectrum: theta_hat restricted to |k_x|, |k_y| <= 127, scaled to the 256^2 grid and o1.2's units (complex128,
256 x 129 rfft layout). Kept at lags 0 / 40 / 400 / 1,000 / 2,000 / 4,000 steps (dt arms: 4,000 only), one npz per
pair in --out; worker_<w>.json holds the run facts, bitwise checks and timings. Prints TAXO_WORKER_DONE last.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse                                                      # noqa: E402
import hashlib                                                       # noqa: E402
import json                                                          # noqa: E402
import math                                                          # noqa: E402
import statistics                                                    # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
ANALYSIS = os.path.dirname(OPT)
for p in (os.path.join(OPT, "seed1"), os.path.join(OPT, "delta"), HERE, ANALYSIS):
    sys.path.insert(0, p)
import profile_solver as PS                                          # noqa: E402
import lean32_precise as LP                                          # noqa: E402
import lean_solver as LS                                             # noqa: E402
import lean32_fast as LF                                             # noqa: E402
import dt_ladder as DL                                               # noqa: E402

THETA_SCALE = 8.0                                                    # o1.2: x = theta / THETA_SCALE
NC = 256                                                             # o1.2's coarse grid (N / 16)
HALF = NC // 2
LAGS = (1, 10, 25, 50, 100)                                          # frames (x 40 steps); o1.2's pair = 100
LAG_T = 0.02
BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 182))
# Pre-registered (PREDICTIONS_taxonomy.md). MDE_PRED: the team's R4 minimum detectable bias of the x-coefficient, in
# units of the band's Delta, for 64 pairs over 16 states (LEARNING's L3 revision); 8-16 and up use its bound 1.2e-7.
MDE_PRED = (8.0e-6, 8.8e-7, 2.6e-7, 1.2e-7, 1.2e-7, 1.2e-7, 1.2e-7, 1.2e-7)
INJ_FACTOR = 3.0
DT_NSUB = (40, 30, 32, 80)
SR_SEED = 20261106


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    return x


def band_of(kap):
    """Coarse band index of |k| (numpy array), -1 outside every band."""
    import numpy as np
    out = np.full(np.shape(kap), -1, dtype=np.int64)
    for i, (a, b) in enumerate(BANDS):
        out[(kap >= a) & (kap < b)] = i
    return out


def main():
    t_wall = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="<member>:<path to ckpt_t*.npz>[@<calls>]; member 0 = a delta_fidelity.py --state-out npz "
                         "(synthetic spin-up: the harness smoke only); @calls = first advance that many lean_u1 calls")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--dt-states", type=int, default=2)
    ap.add_argument("--bench-calls", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W = args.worker

    GD = PS.import_asqg()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from pyqg_jax import steppers as pst
    block = jax.block_until_ready
    N, NK, DT, STRIDE, M = GD.N, GD.N // 2 + 1, GD.DT, PS.STRIDE, PS.M
    assert N // M == NC and abs(STRIDE * LAGS[-1] * DT - LAG_T) < 1e-12, (N, M, STRIDE, DT)
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    res = {"worker": W, "device_kind": jax.devices()[0].device_kind, "jax": jax.__version__,
           "solver_sha256": GD.CONFIG["solver_sha256"], "pairs": args.pairs, "lags_steps": [l * STRIDE for l in LAGS],
           "inj_factor": INJ_FACTOR, "mde_pred": MDE_PRED, "bands": BANDS, "states": [], "bitwise": {}, "npz": []}

    # ---- loading, as o1.2 L192-204 --------------------------------------------------------------------------------
    def config_hash(member):
        cfg = dict(GD.CONFIG, member=member, seed_initial_condition=member)
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def load(path, member):
        own, GD.CONFIG_HASH = GD.CONFIG_HASH, config_hash(member)
        try:
            return GD.load_checkpoint(template, path)
        finally:
            GD.CONFIG_HASH = own

    ckpts = []
    for spec in args.ckpt:
        m, rest = spec.split(":", 1)
        path, _, off = rest.partition("@")
        ckpts.append((int(m), path, int(off) if off else 0))

    # ---- variants ------------------------------------------------------------------------------------------------
    _, ncol = LS.preconditions(GD)
    cu1 = LS.build_consts(GD, ncol)
    u1_call = LS.make_advance(GD, STRIDE, ncol=ncol, unroll1=True)
    ref_call = jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0])
    c32 = LP.tables(GD)
    ek_call = LP.make_call(pst, N, NK, DT, STRIDE, comp=True)
    fast_call = LF.make_call(pst, N, NK, DT, STRIDE, mode="ek", u1=True, ring=3)
    sr_call = LF.make_call(pst, N, NK, DT, STRIDE, mode="sr")
    l32_call = LP.S2.make_variant("lean32", pst, N, NK, DT, STRIDE)[0]
    tab32 = LP.LP.tables32(GD)
    c_dq = dict(cu1, decay=cu1["decay"].astype(jnp.float32).astype(jnp.float64))
    c_sr = dict(c32, key=jax.random.key(SR_SEED + W))
    adv = {"lean_u1": lambda st: u1_call(st, cu1), "prod": ref_call, "lean32ek": lambda st: ek_call(st, c32),
           "lean32ek_fast": lambda st: fast_call(st, c32), "lean32sr": lambda st: sr_call(st, c_sr),
           "lean32": lambda st: l32_call(st, tab32), "dquant": lambda st: u1_call(st, c_dq),
           "f64ic32": lambda st: u1_call(st, cu1)}
    ladder = {n: DL.make_ladder(GD, n, ncol) for n in DT_NSUB}

    # tables on the alias-free coarse layout (rows ky = 0..127, -128..-1; cols kx = 0..128), for the analysis
    def af_rows(a):
        return jnp.concatenate([a[:HALF, :HALF + 1], a[N - HALF:, :HALF + 1]], 0)

    rate = GD.NU * GD.kap ** 2 + GD.MU * GD.pos / GD.kapi ** 2                  # (N, NK) float64
    kxf = np.asarray(jnp.broadcast_to(GD.kx, (N, NK)))
    kyf = np.asarray(jnp.broadcast_to(GD.ky, (N, NK)))
    kapf = np.asarray(GD.kap)
    in_sq = (np.abs(kxf) <= HALF - 1) & (np.abs(kyf) <= HALF - 1) & (kapf >= 1)
    band_f = np.where(in_sq, band_of(kapf), -1)
    tables_af = {"decay64": np.asarray(af_rows(jnp.asarray(GD.decay, jnp.float64))),
                 "decay32": np.asarray(af_rows(c_dq["decay"])), "rate": np.asarray(af_rows(rate)),
                 "kx": np.asarray(af_rows(jnp.asarray(kxf))), "ky": np.asarray(af_rows(jnp.asarray(kyf)))}

    def with_fields(st, qh, t0, t1, t=None, tc=None):
        u0, u1 = st._updates
        return pst.AB3State(state=st.state.update(qh=qh), t=st.t if t is None else t, tc=st.tc if tc is None else tc,
                            _ablevel=jnp.uint8(2), _updates=(u0.update(qh=t0), u1.update(qh=t1)))

    def round32(st):
        r = lambda a: a.astype(jnp.complex64).astype(jnp.complex128)                       # noqa: E731
        return with_fields(st, r(st.state.qh), r(st._updates[0].qh), r(st._updates[1].qh))

    coarse_sub = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    af_scale = (NC * NC) / (N * N) / THETA_SCALE

    @jax.jit
    def coarse_af(st):
        c = af_rows(st.state.qh[0]) * af_scale
        return c.at[HALF, :].set(0).at[:, HALF].set(0)                # |k_x|, |k_y| <= 127: no Nyquist row / column

    diag = jax.jit(GD.diagnostics)

    def spe_cfl(st):
        values, _, umax = jax.device_get(diag(st))
        return float(values[0]), float(umax) * GD.KMAX * DT

    def state_hash(st):
        h = hashlib.sha256()
        for a in (st.state.qh, st._updates[0].qh, st._updates[1].qh):
            h.update(np.asarray(a).tobytes())
        return h.hexdigest()

    def run_frames(f, s0, hash_at=()):
        """o1.2's frames of one 4,000-step run of f from s0: sub (float32 / 8) and af at LAGS (+ lag 0), the end
        state, and state hashes after the calls in hash_at."""
        sub = {0: np.asarray(coarse_sub(s0), dtype=np.float32) / np.float32(THETA_SCALE)}
        afd = {0: np.asarray(coarse_af(s0))}
        hashes, s = {}, s0
        for i in range(1, LAGS[-1] + 1):
            s = f(s)
            if i in LAGS:
                sub[i] = np.asarray(coarse_sub(s), dtype=np.float32) / np.float32(THETA_SCALE)
                afd[i] = np.asarray(coarse_af(s))
            if i in hash_at:
                hashes[i] = state_hash(s)
        return sub, afd, s, hashes

    # ---- the planted shift (control b), from the first reference pair -------------------------------------------
    inj = {}

    def build_injection(x_af, d_af):
        kxa, kya = tables_af["kx"], tables_af["ky"]
        valid = (np.abs(kxa) <= HALF - 1) & (np.abs(kya) <= HALF - 1)
        bc = np.where(valid, band_of(np.sqrt(kxa ** 2 + kya ** 2)), -1)
        drate_b = []
        for b in range(len(BANDS)):
            m = bc == b
            rx = float(np.sqrt(np.mean(np.abs(x_af[m]) ** 2)))
            rd = float(np.sqrt(np.mean(np.abs(d_af[m]) ** 2)))
            drate_b.append(INJ_FACTOR * MDE_PRED[b] * rd / (LAG_T * rx) if rx > 0 else 0.0)
        drate_f = np.where(band_f >= 0, np.asarray(drate_b)[np.maximum(band_f, 0)], 0.0)
        r_inj = rate + jnp.asarray(drate_f)
        inside = GD.kap <= GD.KMAX
        edec_inj = jnp.where(inside, -jnp.expm1(-r_inj * DT), 1.0).astype(jnp.float32)
        dec_inj = inside.astype(jnp.float64) * jnp.exp(-r_inj * DT)
        c32_inj, c_inj64 = dict(c32, edec=edec_inj), dict(cu1, decay=dec_inj)
        adv["inj"] = lambda st: ek_call(st, c32_inj)
        adv["inj64"] = lambda st: u1_call(st, c_inj64)
        inj.update({"drate_b": drate_b, "edec_changed_modes": int(np.sum(np.asarray(edec_inj != c32["edec"])))})
        tables_af["drate_inj"] = np.asarray(af_rows(jnp.asarray(drate_f)))
        print(f"TAXO_INJECT w={W} drate per band: " + " ".join(f"{d:.2e}" for d in drate_b)
              + f"; float32 e changed on {inj['edec_changed_modes']} modes", flush=True)

    # ---- states x pairs --------------------------------------------------------------------------------------------
    for si, (member, path, offset) in enumerate(ckpts):
        t0 = time.time()
        if not os.path.exists(path):
            raise SystemExit(f"ERROR TAXO_CKPT_MISSING {path}")
        if member == 0:                                               # synthetic spin-up npz: harness smoke only
            z = np.load(path)
            st = GD.advance(template, 2)
            st = with_fields(st, jnp.asarray(z["qh"]), jnp.asarray(z["T0"]), jnp.asarray(z["T1"]),
                             jnp.float32(z["t"]), jnp.asarray(z["tc"], dtype=st.tc.dtype))
            completed, t_state = -1, float(z["t"])
        else:
            st, completed = load(path, member)
            t_state = float(GD.block_time(completed))
        for _ in range(offset):
            st = adv["lean_u1"](st)
        st = block(st)
        if int(st._ablevel) != 2:
            raise SystemExit(f"ERROR TAXO_NOT_LEVEL2 {path}")
        spe, cfl = spe_cfl(st)
        info = {"member": member, "path": path, "offset_calls": offset, "t": t_state + offset * STRIDE * DT,
                "completed_blocks": completed, "spe": spe, "cfl": cfl, "load_s": round(time.time() - t0, 1),
                "pairs": []}
        print(f"TAXO_STATE w={W} s={si} member={member} t={info['t']:.2f} SPE={spe:.6g} CFL={cfl:.3f}", flush=True)
        s = st
        for j in range(args.pairs):
            tj = time.time()
            first_state, first_pair = si == 0, j == 0
            spe_j, cfl_j = spe_cfl(s)
            sub_r, af_r, s_next, _ = run_frames(adv["lean_u1"], s)
            dump = {f"lean_u1__sub__{k}": v for k, v in sub_r.items()}
            dump.update({f"lean_u1__af__{k}": v for k, v in af_r.items()})
            if not inj:
                build_injection(af_r[0], af_r[LAGS[-1]] - af_r[0])
            arms = ["lean32ek", "inj"]
            if first_pair:
                arms += ["dquant", "inj64", "lean32", "f64ic32", "lean32sr"]
            if first_state and first_pair:
                arms += ["prod", "lean32ek_fast"]
            hashes = {}
            for arm in arms:
                hat = (1, 10, 100) if (first_state and first_pair and arm in ("lean32ek", "lean32ek_fast")) else ()
                start = round32(s) if arm == "f64ic32" else s
                sub_v, af_v, _, hashes[arm] = run_frames(adv[arm], start, hash_at=hat)
                for k in LAGS:
                    dump[f"{arm}__sub__{k}"] = sub_v[k]
                    dump[f"{arm}__af__{k}"] = af_v[k]
                if arm == "prod":
                    res["bitwise"]["prod_vs_lean_u1_frames"] = {
                        str(k * STRIDE): bool(np.array_equal(sub_v[k].view(np.uint32), sub_r[k].view(np.uint32)))
                        for k in LAGS}
            if first_state and first_pair:
                res["bitwise"]["fast_vs_lean32ek_state"] = {
                    str(k * STRIDE): hashes["lean32ek_fast"][k] == hashes["lean32ek"][k] for k in (1, 10, 100)}
            if first_pair and si < args.dt_states:
                for n in DT_NSUB:
                    end = block(ladder[n](s))
                    dump[f"dt{n}__sub__{LAGS[-1]}"] = np.asarray(coarse_sub(end), dtype=np.float32) / np.float32(
                        THETA_SCALE)
                    dump[f"dt{n}__af__{LAGS[-1]}"] = np.asarray(coarse_af(end))
                    if n == 40:
                        res["bitwise"].setdefault("dt40_vs_lean_u1", []).append({
                            "state": si, "frame": bool(np.array_equal(dump[f"dt40__sub__{LAGS[-1]}"].view(np.uint32),
                                                                      sub_r[LAGS[-1]].view(np.uint32))),
                            "state_hash": state_hash(end) == state_hash(s_next)})
            name = f"pair_w{W}_s{si}_p{j}.npz"
            np.savez(os.path.join(args.out, name), **dump)
            res["npz"].append(name)
            pj = {"j": j, "spe": spe_j, "cfl": cfl_j, "arms": arms, "seconds": round(time.time() - tj, 1),
                  "dt": bool(first_pair and si < args.dt_states)}
            info["pairs"].append(pj)
            print(f"TAXO_PAIR w={W} s={si} p={j} ({pj['seconds']:.0f} s) CFL={pj['cfl']:.3f} arms={','.join(arms)}"
                  + (" +dt" if pj["dt"] else ""), flush=True)
            s = s_next
        res["states"].append(info)
    res["injection"] = inj
    np.savez(os.path.join(args.out, f"tables_w{W}.npz"), **tables_af)
    b = res["bitwise"]
    print(f"TAXO_BITWISE w={W} prod==lean_u1 {all(b.get('prod_vs_lean_u1_frames', {}).values())} "
          f"fast==lean32ek {b.get('fast_vs_lean32ek_state')} dt40==lean_u1 {b.get('dt40_vs_lean_u1')}", flush=True)

    # ---- timing ------------------------------------------------------------------------------------------------------
    res["ms_per_step"] = {}
    for k in ("lean_u1", "lean32ek", "lean32ek_fast", "lean32sr", "lean32"):
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
    print(f"TAXO_BENCH w={W} " + " ".join(f"{k}={v}" for k, v in res["ms_per_step"].items()), flush=True)

    res["wall_s"] = round(time.time() - t_wall, 1)
    with open(os.path.join(args.out, f"worker_{W}.json"), "w") as fh:
        json.dump(jsonable(res), fh, indent=1)
    print(f"TAXO_WORKER_DONE w={W} pairs={len(res['npz'])} wall={res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR TAXO_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
