"""seed2_opt.py -- seed 2: the fastest A-SQG solver step that keeps the physics, gated against production.

Three drop-in replacements for o1.2's 40-step call `jax.jit(lambda st: lax.scan(GD.inner, st, None, length=40)[0])`.
Each takes and returns the production AB3State (same pytree, complex128 fields) and is valid at AB3 level 2:

  lean64  float64 throughout, the same terms with less data movement (README.md):
          - u and v come from ONE forward complex FFT of conj(u_hat + i v_hat) / N^2 built on the full plane. That
            avoids the C2R input copy and the cuBLAS 1/N^2 scal for u and v. theta keeps its C2R.
          - the AB3 history is a 2-slot ring (scan unrolled by 2), not the (updates, updates_p) rotation;
          - terms that are exactly x1 or +0 in this configuration are dropped: pyqg's filter (filterfac = 0) and the
            background-gradient term (beta = U = 0). The forcing comes from two 8x8 patches instead of a dense 134 MB
            array. Every table is a jit argument, so XLA cannot constant-fold it back into a dense complex128 array.
  mixed   lean64 with the nonlinear term (the 4 FFTs and the products) in float32. theta_hat, the AB3 history, the
          forcing, the damping and the AB3 update stay float64 (the gyaradax split).
  lean32  lean64 entirely in float32 inside the call. The state is cast at the entry and exit of each 40-step call.

GATE (fixed before any run; README.md). From two level-2 states, the solver's random IC and a synthetic broadband
field at a production-like CFL, all of the following must hold:
  - after 1 solver step, the max over (theta_hat, T_-1, T_-2) of the relative L2 error vs GD.inner is <= 1e-12
    (lean64) or <= 1e-5 (mixed, lean32);
  - t, tc and _ablevel equal the reference's;
  - every value is finite, here and after 40, 480 and 4,000 steps (wherever the reference is finite);
  - the FFTs run in the precision the variant claims, and the lean32 loop body holds no full-size f64/c128 array.
Information only: bitwise equality after 480 steps; after 4,000 steps (one 0.02 interval), the relative L2
difference of the 256^2 coarse() field and the 0.5<theta^2> (SPE) difference and drift. Added by the main session
before the first run (report-only, never gating): the 1-step relative error per |k| octave band (so errors in
low-energy high-k modes are not hidden by the energetic large scales), and the ORIGINAL step run in float32 (job E's
profile_solver_fp32.make_inner) through the same 1-step and 4,000-step comparisons, so mixed/lean32's errors can be
split into plain float32 rounding vs anything the restructuring adds (rel1_over_f32_original).
TIMING: profile_solver.py's method (median of 3 x 60 jitted 40-step calls after 3 warm-up calls, ms per step). Then
the o1.2 optimizer-step pattern (12 calls + coarse()) for every passing variant.

    "$PY" SQG/analysis/opt/seed2/seed2_opt.py --out <dir>        # from the repo root, inside a GPU job
Prints SEED2_GATE_OK <v> | ERROR SEED2_GATE_FAIL <v> <reason>, SEED2_BENCH <v> ..., SEED2_OPTSTEP <v> ..., and
SEED2_DONE last. Writes <dir>/seed2.json and <dir>/hlo_<name>.txt.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(os.path.dirname(HERE))                    # SQG/analysis
SQG_DIR = os.path.dirname(ANALYSIS)                                  # SQG
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")      # as production
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("SQG_ASQG_PY", os.path.join(SQG_DIR, "fixed_ring", "A-SQG_data.py"))   # the vendored copy
if "SQG_TRAINING_ROOT" not in os.environ and os.environ.get("MEMBER_ROOT"):
    os.environ["SQG_TRAINING_ROOT"] = os.path.join(os.environ["MEMBER_ROOT"], "sqg_stage", "AllTrainData")

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import re                                                            # noqa: E402
import statistics                                                    # noqa: E402
import time                                                          # noqa: E402
import traceback                                                     # noqa: E402

sys.path.insert(0, ANALYSIS)
import profile_solver as PS                                          # noqa: E402  (import_asqg, STRIDE, CALLS, M)

import jax                                                           # noqa: E402
import jax.numpy as jnp                                              # noqa: E402
import numpy as np                                                   # noqa: E402
from jax import lax                                                  # noqa: E402

# ---- pre-registered constants (README.md). Do not change them after a run. ------------------------------------------
GATE = {"lean64": 1e-12, "mixed": 1e-5, "lean32": 1e-5}             # max relative L2 over the 3 AB3 fields, 1 step
# ROUND 1b (written 2026-11-05 before the rerun, at the user's request, after round 1 = job 7814300). Round 1 showed
# (a) the 1e-5 float32 limit sits BELOW plain float32's own error: the ORIGINAL step in float32 scored 2.48e-5 on the
# synthetic state, so no float32 code could pass it; (b) every variant failed only on t, the float32 time counter, which
# the step never reads (fields bitwise / <= 3e-14). Round 1's verdicts stand as recorded; round 1b gates:
#   float32 variants: 1-step error <= F32_RATIO x the ORIGINAL step's float32 error from the same state, measured in the
#                     same run (anchored to float32 rounding, not to the variants), and never above F32_CEIL;
#   lean64: unchanged (<= 1e-12);
#   all: tc and _ablevel exact; |t - t_ref| <= T_REL_TOL |t_ref|; exact t equality reported as information.
F32_RATIO, F32_CEIL, T_REL_TOL = 2.0, 1e-4, 1e-6
FFT_PRECISION = {"lean64": "double", "mixed": "single", "lean32": "single"}
SYN_SEED = 20261104                                                  # synthetic field: jax.random seed
SYN_KD = 300.0                                                       # amplitude ~ k^(-4/3) exp(-(k - KF) / SYN_KD)
SYN_CFL = 0.30                                                       # umax * KMAX * DT at construction
PATCH = 8                                                            # forcing patch: rows [0,8) U [N-8,N), cols [0,8)
LEAVES = ("theta_hat", "T_-1", "T_-2")
FULL = r"\[(?:1,)?4096,(?:2049|4096)\]"                              # a full-size field in HLO text


# ---- the lean step ----------------------------------------------------------------------------------------------------
def build_w(ph, kx, ky, s):
    """conj(u_hat + i v_hat) * s on the full (N, N) plane, from psi_hat = theta_hat / |k| on the rfft half plane.

    u_hat + i v_hat = -(k_x + i k_y) psi_hat. Columns 0..N/2 are taken from the stored half plane. Columns N/2+1..N-1
    hold k_x' = j - N < 0 and come from the Hermitian mirror psi_hat(-k) = conj(psi_hat(k)), i.e. row (-r) mod N and
    column N - j. The forward FFT of the result is u - i v (s = 1/N^2, a power of two, so exact).
    """
    n, half = ph.shape[0], ph.shape[0] // 2
    shape_l = ph.shape
    m_l = lax.complex(jnp.broadcast_to(-s * kx[None, :], shape_l), jnp.broadcast_to(s * ky[:, None], shape_l))
    w_l = m_l * jnp.conj(ph)                                          # -(k_x - i k_y) conj(psi_hat) s
    rows = lax.rev(ph, (0,))                                          # rows n-1 .. 0
    rows = lax.concatenate([lax.slice_in_dim(rows, n - 1, n, axis=0),
                            lax.slice_in_dim(rows, 0, n - 1, axis=0)], 0)          # row r <- (-r) mod n
    ph_r = lax.rev(lax.slice_in_dim(rows, 1, half, axis=1), (1,))     # columns half-1 .. 1 (= N - j)
    kx_r = lax.rev(kx[1:half], (0,))                                  # half-1 .. 1 = -k_x' of those columns
    shape_r = ph_r.shape
    m_r = lax.complex(jnp.broadcast_to(s * kx_r[None, :], shape_r), jnp.broadcast_to(s * ky[:, None], shape_r))
    return lax.concatenate([w_l, m_r * ph_r], 1)                      # right half: (|k_x'| + i k_y) psi_hat(-k) s


def make_step(kind, n, nk, dt):
    """One A-SQG step on bare 2-D fields: (theta_hat, T_-1, T_-2, tables) -> (theta_hat', T).

    T = -(i k_x (u theta)^ + i k_y (v theta)^) + F0, theta_hat' = (theta_hat + c1 T + c2 T_-1 + c3 T_-2) * decay:
    GD.inner at AB3 level 2 without the terms that are exactly x1 / +0 here (checked in main()).
    """
    s = 1.0 / (n * n)
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt      # pyqg_jax AB3 level-2 coefficients, verbatim
    low = kind in ("mixed", "lean32")

    def step(th, t_n, t_o, c):
        fc = jnp.complex64 if low else th.dtype
        kx, ky = c["kx"], c["ky"]
        ph = c["kinv"] * th                                           # psi_hat, exactly as the reference
        p = jnp.fft.fft2(build_w(ph, kx, ky, s).astype(fc))           # u - i v
        q = jnp.fft.irfft2(th.astype(fc), s=(n, n))                   # theta
        uqh = jnp.fft.rfft2(jnp.real(p) * q).astype(th.dtype)
        vqh = jnp.fft.rfft2(-jnp.imag(p) * q).astype(th.dtype)
        ik = lax.complex(jnp.zeros_like(kx), kx)
        il = lax.complex(jnp.zeros_like(ky), ky)
        f0 = (jnp.pad(c["f0top"], ((0, n - PATCH), (0, nk - PATCH)))
              + jnp.pad(c["f0bot"], ((n - PATCH, 0), (0, nk - PATCH))))
        t_new = -(ik[None, :] * uqh + il[:, None] * vqh) + f0
        th_new = (th + c1 * t_new + c2 * t_n + c3 * t_o) * c["decay"]
        return th_new, t_new

    return step


def make_variant(kind, pst, n, nk, dt, stride):
    """(call, one, two): the jitted 40-step call and 1-step function (AB3State, tables) -> AB3State, and the raw
    2-step scan body. The AB3 history is a ring: each step writes the new tendency over the older slot."""
    step = make_step(kind, n, nk, dt)
    state_c = jnp.complex64 if kind == "lean32" else jnp.complex128
    # t advances by the run-time scalar c["tdt"] (float32 dt), not a compile-time constant: with a constant, XLA may fold
    # the unrolled (t + dt) + dt into t + 2 dt and round t differently from pyqg's one-add-per-step (round 1).

    def unpack(st):
        return tuple(x.qh[0].astype(state_c) for x in (st.state, st._updates[0], st._updates[1]))

    def pack(st, th, t_n, t_o, t, tc):
        cd = st.state.qh.dtype
        u0, u1 = st._updates
        return pst.AB3State(state=st.state.update(qh=th.astype(cd)[None]), t=t, tc=tc, _ablevel=jnp.uint8(2),
                            _updates=(u0.update(qh=t_n.astype(cd)[None]), u1.update(qh=t_o.astype(cd)[None])))

    def two(carry, c):
        th, t_n, t_o, t, tc = carry
        th1, t1 = step(th, t_n, t_o, c)                               # history (t1, t_n): t1 takes t_o's slot
        th2, t2 = step(th1, t1, t_n, c)                               # history (t2, t1): t2 takes t_n's slot
        return th2, t2, t1, (t + c["tdt"]) + c["tdt"], (tc + 1) + 1

    def call(st, c):
        carry = (*unpack(st), st.t, st.tc)
        carry = lax.scan(lambda cr, _: (two(cr, c), None), carry, None, length=stride // 2)[0]
        return pack(st, *carry)

    def one(st, c):
        th, t_n, t_o = unpack(st)
        th1, t1 = step(th, t_n, t_o, c)
        return pack(st, th1, t1, t_n, st.t + c["tdt"], st.tc + 1)

    return jax.jit(call), jax.jit(one), two


# ---- HLO helpers --------------------------------------------------------------------------------------------------------
def hlo_stats(txt):
    pats = {"fft": r"\bfft\(", "conditional": r"\bconditional\(", "while": r"\bwhile\(", "copy": r"\bcopy\(",
            "fusion": r"\bfusion\(", "custom_call": r"custom-call\("}
    out = {k: len(re.findall(p, txt)) for k, p in pats.items()}
    out["full_size_copies"] = len(re.findall(r"=\s*\w+" + FULL + r"\S*\s+copy\(", txt))
    ffts = []
    for line in txt.splitlines():
        if re.search(r"\bfft\(", line):
            dt = re.search(r"=\s*(\w+)\[", line)
            kind = re.search(r"fft_type=(\w+)", line)
            ffts.append(f"{kind.group(1) if kind else '?'}:{dt.group(1) if dt else '?'}")
    out["fft_ops"] = ffts
    return out


def fft_precision_ok(stats, want):
    ok = {"double": {"c128", "f64"}, "single": {"c64", "f32"}}[want]
    types = {f.split(":")[1] for f in stats["fft_ops"]}
    return bool(types) and types <= ok, sorted(types)


def main():
    t_wall = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", type=int, default=60, help="bench: 40-step calls per measurement")
    ap.add_argument("--repeats", type=int, default=3, help="bench: measurements (median reported)")
    ap.add_argument("--long-calls", type=int, default=100, help="40-step calls of the long comparison (4,000 steps)")
    ap.add_argument("--variants", default="lean64,mixed,lean32")
    ap.add_argument("--budget-s", type=float, default=780.0, help="skip optional sections past this wall time")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = [v for v in args.variants.split(",") if v]
    if not names or any(v not in GATE for v in names) or args.long_calls < PS.CALLS:
        raise SystemExit(f"ERROR SEED2_BAD_ARGS variants={names} long_calls={args.long_calls}")

    t0 = time.time()
    GD = PS.import_asqg()
    from pyqg_jax import steppers as pst
    t_import = time.time() - t0
    block = jax.block_until_ready
    copy = lambda st: jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), st)        # noqa: E731

    N, NK, DT = GD.N, GD.N // 2 + 1, GD.DT
    STRIDE, CALLS, M = PS.STRIDE, PS.CALLS, PS.M
    mdl = GD.sm.model
    dropped = {"filterfac": float(mdl.filterfac), "beta": float(mdl.beta), "U": float(mdl.U), "rek": float(mdl.rek)}
    if any(v != 0.0 for v in dropped.values()) or GD.sm.stepper.dt != DT or STRIDE % 2 or N & (N - 1):
        raise SystemExit(f"ERROR SEED2_CONFIG_CHANGED {dropped} dt={GD.sm.stepper.dt} stride={STRIDE} N={N}")
    F0 = np.asarray(GD.F0_hat[0])
    iy, ix = np.nonzero(F0)
    if not (len(ix) and ix.max() < PATCH and np.all((iy < PATCH) | (iy >= N - PATCH))):
        raise SystemExit("ERROR SEED2_FORCING_OUTSIDE_PATCH: the forcing has modes outside the 8x8 corner patches")
    tables64 = {"kinv": (mdl.f_0 / mdl.Nb) * jnp.sqrt(mdl.wv2i), "kx": mdl.kk, "ky": mdl.ll, "decay": GD.decay,
                "f0top": jnp.asarray(F0[:PATCH, :PATCH]), "f0bot": jnp.asarray(F0[N - PATCH:, :PATCH]),
                "tdt": jnp.float32(DT)}                               # the t increment, as pyqg's jnp.float32(dt)
    down = {np.dtype("complex128"): jnp.complex64, np.dtype("float64"): jnp.float32}
    tables32 = {k: v.astype(down.get(np.dtype(v.dtype), v.dtype)) for k, v in tables64.items()}
    tables = {"lean64": tables64, "mixed": tables64, "lean32": tables32}

    res = {"device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind, "jax": jax.__version__,
           "solver_sha256": GD.CONFIG["solver_sha256"], "import_s": round(t_import, 1), "stride": STRIDE,
           "calls_per_optstep": CALLS, "gate_thresholds": GATE, "fft_precision_claimed": FFT_PRECISION,
           "dropped_terms_config": dropped, "forcing_nonzero_modes": int(len(ix)),
           "synthetic": {"seed": SYN_SEED, "kd": SYN_KD, "cfl_target": SYN_CFL}, "variants": {}}

    # ---- the two level-2 start states ----------------------------------------------------------------------------
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    st_rand = GD.advance(template, 2)                                 # SQG_INIT=random of profile_solver.py

    kap = GD.kap
    white = jax.random.normal(jax.random.key(SYN_SEED), (N, N), dtype=jnp.float64)
    amp = jnp.where(kap >= GD.KF, (kap / GD.KF) ** (-4.0 / 3.0) * jnp.exp(-(kap - GD.KF) / SYN_KD), kap / GD.KF)
    amp = jnp.where((kap > 0) & (kap <= GD.KMAX), amp, 0.0)
    th = jnp.fft.rfft2(white) * amp
    u = jnp.fft.irfft2(-1j * GD.ky * th / GD.kapi, s=(N, N))
    v = jnp.fft.irfft2(1j * GD.kx * th / GD.kapi, s=(N, N))
    th = (th * (SYN_CFL / (float(jnp.max(jnp.sqrt(u ** 2 + v ** 2))) * GD.KMAX * DT))).astype(jnp.complex128)
    st_syn = GD.advance(GD.sm.initialize_stepper_state(template.state.update(qh=th[None])), 2)
    del white, amp, th, u, v
    ics = {"random": st_rand, "synthetic": st_syn}

    ref_call = jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0])           # o1.2's advance
    ref_one = jax.jit(lambda st: GD.inner(st, None)[0])
    coarse = jax.jit(lambda st: jnp.fft.irfft2(st.state.qh[0], s=(N, N))[::M, ::M].astype(jnp.float32))
    diag = jax.jit(GD.diagnostics)                                    # (values: SPE, VIE, eps_nu, eps_mu, P; E(k); umax)

    def fields(st):
        return (st.state.qh, st._updates[0].qh, st._updates[1].qh)

    @jax.jit
    def compare_dev(a, b):
        rows = []
        for x, y in zip(fields(a), fields(b)):
            d = x - y
            rows.append(jnp.stack([jnp.sum(jnp.abs(d) ** 2), jnp.sum(jnp.abs(y) ** 2), jnp.max(jnp.abs(d)),
                                   jnp.all(jnp.isfinite(x)).astype(jnp.float64),
                                   jnp.all(x == y).astype(jnp.float64)]))
        meta = jnp.stack([a.t == b.t, a.tc == b.tc, a._ablevel == b._ablevel]).astype(jnp.float64)
        return jnp.stack(rows), meta, jnp.stack([a.t, b.t]).astype(jnp.float64)

    def compare(a, b):
        """Relative L2 per AB3 field (theta_hat, T_-1, T_-2) of a vs reference b, finiteness of a, bitwise, meta.
        meta_equal = t, tc, _ablevel exactly equal (information); meta_ok = tc and _ablevel exact and t within
        T_REL_TOL (round 1b gate)."""
        rows, meta, tt = (np.asarray(x, dtype=np.float64) for x in jax.device_get(compare_dev(a, b)))
        rel = np.array([np.sqrt(r[0] / r[1]) if r[1] > 0 else (0.0 if r[0] == 0 else np.inf) for r in rows])
        den = rows[:, 1].sum()
        t_rel = abs(tt[0] - tt[1]) / abs(tt[1]) if tt[1] != 0 else abs(tt[0])
        return {"rel_l2": {k: float(x) for k, x in zip(LEAVES, rel)}, "rel_l2_max": float(np.max(rel)),
                "rel_l2_all": float(np.sqrt(rows[:, 0].sum() / den)) if den > 0 else float("nan"),
                "max_abs": {k: float(x) for k, x in zip(LEAVES, rows[:, 2])}, "finite": bool(rows[:, 3].all()),
                "fields_bitwise": bool(rows[:, 4].all()), "meta_equal": bool(meta.all()),
                "t": float(tt[0]), "t_ref": float(tt[1]), "t_rel_diff": float(t_rel),
                "meta_ok": bool(meta[1] and meta[2] and t_rel <= T_REL_TOL)}

    def spe_umax(st):
        values, _, umax = jax.device_get(diag(st))
        return float(values[0]), float(umax)

    # ---- report-only: relative error per |k| octave band [2^b, 2^(b+1)), b = 0..11; k = 0 in its own bucket ---------
    bidx = jnp.where(GD.kap >= 1, jnp.floor(jnp.log2(jnp.maximum(GD.kap, 1.0))), 12).astype(jnp.int32).ravel()

    @jax.jit
    def band_dev(x, y, idx):                                          # idx as an argument: no 33 MB baked constant
        x, y = x.astype(jnp.complex128), y.astype(jnp.complex128)
        num = jax.ops.segment_sum((jnp.abs(x - y) ** 2).ravel(), idx, num_segments=13)
        den = jax.ops.segment_sum((jnp.abs(y) ** 2).ravel(), idx, num_segments=13)
        return num, den

    def band_rel(x, y):
        num, den = (np.asarray(v, dtype=np.float64) for v in jax.device_get(band_dev(x, y, bidx)))
        return {f"{2 ** b}-{2 ** (b + 1)}": (float(np.sqrt(num[b] / den[b])) if den[b] > 0 else None) for b in range(12)}

    def bands_1step(s1, r1):
        """theta_hat and the newest tendency (_updates[0]) after 1 step, per band, vs the reference."""
        return {"theta_hat": band_rel(s1.state.qh[0], r1.state.qh[0]),
                "T_new": band_rel(s1._updates[0].qh[0], r1._updates[0].qh[0])}

    def worst_band(b):
        ks = [k for k, v in b.items() if v is not None]
        return (max(ks, key=lambda k: b[k]), b[max(ks, key=lambda k: b[k])]) if ks else ("-", float("nan"))

    def long_run(adv, st):
        """{1: state after 40 steps, CALLS: after 480, long_calls: after 4,000} from st."""
        snaps = {}
        for i in range(1, args.long_calls + 1):
            st = adv(st)
            if i in (1, CALLS):
                snaps[i] = st
        snaps[args.long_calls] = block(st)
        return snaps

    # ---- reference: 1 step and the long runs, from both start states ---------------------------------------------
    ref1, ref_long, ref_end = {}, {}, {}
    for ic, st in ics.items():
        if int(st._ablevel) != 2:
            raise SystemExit(f"ERROR SEED2_IC_NOT_LEVEL2 {ic}")
        ny = float(jnp.max(jnp.abs(st.state.qh[0, N // 2, :])) + jnp.max(jnp.abs(st.state.qh[0, :, N // 2])))
        spe0, umax0 = spe_umax(st)
        ref1[ic] = block(ref_one(st))
        t0 = time.time()
        ref_long[ic] = long_run(ref_call, st)
        spe_e, umax_e = spe_umax(ref_long[ic][args.long_calls])
        ref_end[ic] = {"coarse": np.asarray(coarse(ref_long[ic][args.long_calls]), dtype=np.float64), "spe": spe_e,
                       "finite": {k: compare(s_, s_)["finite"] for k, s_ in ref_long[ic].items()}}
        res.setdefault("ics", {})[ic] = {
            "spe0": spe0, "theta_rms0": float(np.sqrt(2 * spe0)), "cfl0": umax0 * GD.KMAX * DT,
            "nyquist_lines_max_abs": ny, "ref_spe_end": spe_e, "ref_cfl_end": umax_e * GD.KMAX * DT,
            "ref_spe_drift_end": spe_e / spe0 - 1.0, "ref_finite": ref_end[ic]["finite"],
            "ref_long_s": round(time.time() - t0, 1),
            "ref_repeat_1step_bitwise": compare(block(ref_one(st)), ref1[ic])["fields_bitwise"]}
    r1 = ref1["random"]                                               # the metric itself, on a known perturbation
    pert = r1.update(state=r1.state.update(qh=r1.state.qh * (1.0 + 1e-9)))
    res["metric_selfcheck_rel_1e-9"] = compare(pert, r1)["rel_l2"]["theta_hat"]
    del r1, pert
    txt = ref_call.lower(st_rand).compile().as_text()
    with open(os.path.join(args.out, "hlo_ref.txt"), "w") as fh:
        fh.write(txt)
    res["ref_hlo"] = hlo_stats(txt)
    ic_line = "; ".join(f"{k}: theta_rms {v['theta_rms0']:.3g}, CFL {v['cfl0']:.3f} -> {v['ref_cfl_end']:.3f} "
                        f"(4000 steps), SPE drift {v['ref_spe_drift_end']:+.3e}" for k, v in res["ics"].items())
    print(f"seed2: {res['device_kind']}, jax {jax.__version__}, import {t_import:.0f} s; {ic_line}", flush=True)

    # ---- the ORIGINAL step in float32 (job E's builder) through the same comparisons -----------------------------------
    # Round 1b: this baseline SETS the float32 variants' gate (F32_RATIO x its 1-step error from the same state, capped at
    # F32_CEIL), so it runs before the variants. rel1_over_f32_original ~ 1 or below means the restructured code (C2C
    # packing, ring, lean AB3) adds nothing beyond float32 rounding itself.
    base_err = {}
    try:
        import profile_solver_fp32 as PF
        sm32 = GD.sqg.aux(GD.p._replace(f64=False))[0]
        inner32 = PF.make_inner(sm32, GD.F0_hat.astype(jnp.complex64), GD.decay.astype(jnp.float32))
        one32 = jax.jit(lambda st: inner32(st, None)[0])
        call32 = jax.jit(lambda st: lax.scan(inner32, st, None, length=STRIDE)[0])
        dn = {np.dtype("complex128"): jnp.complex64, np.dtype("float64"): jnp.float32}
        to32 = lambda st: jax.tree_util.tree_map(lambda x: x.astype(dn[x.dtype]) if x.dtype in dn else x, st)  # noqa: E731
        to64 = lambda st: jax.tree_util.tree_map(                                                            # noqa: E731
            lambda x: x.astype(jnp.complex128) if x.dtype == jnp.complex64 else x, st)
        base = res["f32_original_baseline"] = {
            "what": "A-SQG_data.inner's four lines with pyqg_jax Precision.SINGLE, F0_hat complex64, decay float32 "
                    "(profile_solver_fp32.make_inner, job E); state cast once at the start, kept complex64",
            "gate": {}, "band_rel_1step": {}, "long": {}}
        for ic, st in ics.items():
            s1 = to64(block(one32(to32(st))))
            base["gate"][ic] = compare(s1, ref1[ic])
            base["band_rel_1step"][ic] = bands_1step(s1, ref1[ic])
            del s1
            s = to32(st)
            for _ in range(args.long_calls):
                s = call32(s)
            end = to64(block(s))
            cv = np.asarray(coarse(end), dtype=np.float64)
            spe_v, _ = spe_umax(end)
            base["long"][ic] = {"coarse_rel_l2_end": float(np.linalg.norm(cv - ref_end[ic]["coarse"])
                                                           / np.linalg.norm(ref_end[ic]["coarse"])),
                                "spe_rel_diff_end": spe_v / ref_end[ic]["spe"] - 1.0}
            del s, end
            base_err[ic] = base["gate"][ic]["rel_l2_max"]
        print(f"SEED2_F32BASE original code in float32: rel1 random={base_err['random']:.2e} "
              f"synthetic={base_err['synthetic']:.2e}; {args.long_calls * STRIDE}-step coarse rel diff "
              f"{base['long']['random']['coarse_rel_l2_end']:.2e}/{base['long']['synthetic']['coarse_rel_l2_end']:.2e} "
              f"(random/synthetic); sets the float32 gates (x{F32_RATIO:g}, cap {F32_CEIL:g})", flush=True)
    except Exception as exc:
        res["f32_original_baseline"] = {"error": f"{type(exc).__name__}: {exc}"[:2000],
                                        "traceback": traceback.format_exc()[-4000:]}
        print(f"ERROR SEED2_F32BASE_FAILED {type(exc).__name__}: float32 variants cannot be gated this run", flush=True)

    def threshold(name, ic):
        """Round 1b gate on the 1-step max relative L2 over the AB3 fields; None = not evaluable (fails the variant)."""
        if name == "lean64":
            return GATE["lean64"]
        b = base_err.get(ic)
        if b is None or not np.isfinite(b) or b <= 0:
            return None
        return min(F32_CEIL, F32_RATIO * b)

    # ---- variants: build, structure checks, gate, long comparison ------------------------------------------------
    advs, passed = {}, []
    for name in names:
        thr = {ic: threshold(name, ic) for ic in ics}
        known = [v for v in thr.values() if v is not None]
        vr = res["variants"][name] = {
            "gate_rule": ("1-step <= 1e-12" if name == "lean64" else
                          f"1-step <= min({F32_CEIL:g}, {F32_RATIO:g} x original-step float32 error, same state)"),
            "gate_threshold_by_ic": thr, "gate_threshold": max(known) if known else None,
            "t_rel_tol": T_REL_TOL, "fft_precision_claimed": FFT_PRECISION[name]}
        try:
            C = tables[name]
            call_j, one_j, two = make_variant(name, pst, N, NK, DT, STRIDE)
            t0 = time.time()
            txt = call_j.lower(st_rand, C).compile().as_text()
            vr["compile_s"] = round(time.time() - t0, 1)
            with open(os.path.join(args.out, f"hlo_{name}.txt"), "w") as fh:
                fh.write(txt)
            vr["hlo"] = hlo_stats(txt)
            prec_ok, prec_types = fft_precision_ok(vr["hlo"], FFT_PRECISION[name])
            vr["fft_dtypes"] = prec_types
            reasons = [] if prec_ok else [f"fft_precision:{','.join(prec_types)}"]
            if name == "lean32":                                      # no silent promotion inside the loop body
                sds = jax.ShapeDtypeStruct
                carry = (sds((N, NK), jnp.complex64),) * 3 + (sds((), jnp.float32), sds((), jnp.uint32))
                body = jax.jit(two).lower(carry, C).compile().as_text()
                vr["body_full_size_f64_c128"] = len(re.findall(r"(?:c128|f64)" + FULL, body))
                if vr["body_full_size_f64_c128"]:
                    reasons.append(f"promoted:{vr['body_full_size_f64_c128']}_full_size_f64_c128_in_body")
            try:
                ca = one_j.lower(st_rand, C).compile().cost_analysis()
                ca = ca[0] if isinstance(ca, (list, tuple)) else ca
                vr["cost_one_step"] = {k: float(ca[k]) for k in ("flops", "bytes accessed") if k in ca}
            except Exception as exc:                                  # informational only
                vr["cost_one_step"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            adv = lambda st, f=call_j, C=C: f(st, C)                  # noqa: E731  (drop-in: AB3State -> AB3State)
            vr["gate"], vr["long"] = {}, {}
            for ic, st in ics.items():
                s1 = block(one_j(st, C))
                g = vr["gate"][ic] = compare(s1, ref1[ic])
                try:                                                  # report-only: never gates, never raises
                    b1 = vr.setdefault("band_rel_1step", {})[ic] = bands_1step(s1, ref1[ic])
                    if ic == "synthetic":
                        wb, wv = worst_band(b1["theta_hat"])
                        print(f"SEED2_BANDS {name} synthetic theta_hat worst |k| band {wb}: rel {wv:.2e} "
                              f"(report only)", flush=True)
                except Exception as exc:
                    vr.setdefault("band_rel_1step", {})[ic] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                del s1
                if thr[ic] is None:
                    reasons.append(f"no_f32_baseline_{ic}")
                elif not g["rel_l2_max"] <= thr[ic]:
                    reasons.append(f"rel1_{ic}={g['rel_l2_max']:.3e}>{thr[ic]:.3e}")
                if not g["finite"]:
                    reasons.append(f"nonfinite_1step_{ic}")
                if not g["meta_ok"]:
                    reasons.append(f"tc_ablevel_or_t_1step_{ic}(t_rel={g['t_rel_diff']:.1e})")
                snaps = long_run(adv, st)
                lg = vr["long"][ic] = {}
                for k, s_ in snaps.items():
                    c_ = lg[f"{k * STRIDE}_steps"] = compare(s_, ref_long[ic][k])
                    if not c_["finite"] and ref_end[ic]["finite"][k]:
                        reasons.append(f"nonfinite_{k * STRIDE}_steps_{ic}")
                    if not c_["meta_ok"]:
                        reasons.append(f"tc_ablevel_or_t_{k * STRIDE}_steps_{ic}(t_rel={c_['t_rel_diff']:.1e})")
                lg["t_exact_all"] = all(lg[f"{k * STRIDE}_steps"]["meta_equal"] for k in snaps)   # information
                end = snaps[args.long_calls]
                cv = np.asarray(coarse(end), dtype=np.float64)
                cr = ref_end[ic]["coarse"]
                spe_v, _ = spe_umax(end)
                lg["coarse_rel_l2_end"] = float(np.linalg.norm(cv - cr) / np.linalg.norm(cr))
                lg["spe_end"], lg["spe_rel_diff_end"] = spe_v, spe_v / ref_end[ic]["spe"] - 1.0
                lg["spe_drift_end"] = spe_v / res["ics"][ic]["spe0"] - 1.0
                lg["bitwise_480"] = bool(lg[f"{CALLS * STRIDE}_steps"]["fields_bitwise"]
                                         and lg[f"{CALLS * STRIDE}_steps"]["meta_equal"])
                del snaps, end
            vr["gate_pass"], vr["gate_reasons"] = not reasons, reasons
            n_end = f"{args.long_calls * STRIDE}"
            thr_s = "/".join("n.a." if thr[ic] is None else f"{thr[ic]:.1e}" for ic in ics)
            detail = (f"rel1 random={vr['gate']['random']['rel_l2_max']:.2e} "
                      f"synthetic={vr['gate']['synthetic']['rel_l2_max']:.2e} (thr {thr_s}); "
                      f"t exact={vr['long']['random']['t_exact_all']}/{vr['long']['synthetic']['t_exact_all']}; "
                      f"bitwise480={vr['long']['random']['bitwise_480']}/{vr['long']['synthetic']['bitwise_480']}; "
                      f"{n_end}-step coarse rel diff {vr['long']['random']['coarse_rel_l2_end']:.2e}/"
                      f"{vr['long']['synthetic']['coarse_rel_l2_end']:.2e}, dSPE "
                      f"{vr['long']['random']['spe_rel_diff_end']:+.2e}/{vr['long']['synthetic']['spe_rel_diff_end']:+.2e}"
                      f" (random/synthetic)")
            if reasons:
                print(f"ERROR SEED2_GATE_FAIL {name} {';'.join(reasons)} | {detail}", flush=True)
            else:
                passed.append(name)
                print(f"SEED2_GATE_OK {name} {detail}", flush=True)
            advs[name] = adv
        except Exception as exc:
            vr["gate_pass"], vr["error"] = False, f"{type(exc).__name__}: {exc}"[:2000]
            vr["traceback"] = traceback.format_exc()[-6000:]
            msg = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
            print(f"ERROR SEED2_GATE_FAIL {name} exception:{type(exc).__name__}:{msg}", flush=True)

    for name in names:                                                # information: precision vs restructuring
        g = res["variants"][name].get("gate", {})
        res["variants"][name]["rel1_over_f32_original"] = {
            ic: (g[ic]["rel_l2_max"] / base_err[ic] if ic in g and base_err.get(ic) else None) for ic in ics}
    del ref_long, ref1

    # ---- timing: profile_solver.py's method -------------------------------------------------------------------------
    def bench(fn):
        st = copy(st_rand)
        for _ in range(3):
            st = fn(st)
        block(st)
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            for _ in range(args.calls):
                st = fn(st)
            block(st)
            times.append(time.perf_counter() - t0)
        return 1e3 * statistics.median(times) / (args.calls * STRIDE)

    ms_ref = bench(ref_call)
    res["ref_ms_per_step"] = round(ms_ref, 4)
    for name, adv in advs.items():
        vr = res["variants"][name]
        try:
            ms = bench(adv)
            vr["ms_per_step"], vr["speedup"] = round(ms, 4), round(ms_ref / ms, 3)
            print(f"SEED2_BENCH {name} ms_ref={ms_ref:.4f} ms_cand={ms:.4f} speedup={ms_ref / ms:.3f} "
                  f"gate={'OK' if vr['gate_pass'] else 'FAIL'}", flush=True)
        except Exception as exc:
            vr["bench_error"] = f"{type(exc).__name__}: {exc}"[:2000]
            print(f"ERROR SEED2_BENCH_FAIL {name} {type(exc).__name__}", flush=True)

    # ---- the o1.2 optimizer-step pattern: 12 x (40 steps, coarse frame) ---------------------------------------------
    def optstep_s(adv, reps=3):
        def opt_step(st):
            frames = []
            for _ in range(CALLS):
                st = adv(st)
                frames.append(coarse(st))
            return st, frames
        st, fr = opt_step(copy(st_rand))
        block((st, fr))
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            st, fr = opt_step(st)
            block((st, fr))
            times.append(time.perf_counter() - t0)
        return statistics.median(times)

    timed = sorted((res["variants"][n]["ms_per_step"], n) for n in passed if "ms_per_step" in res["variants"][n])
    res["fastest_passing"] = timed[0][1] if timed else None
    if timed:
        s_ref = optstep_s(ref_call)
        res["ref_optstep_s"] = round(s_ref, 4)
        for i, (_, name) in enumerate(timed):
            if i and time.time() - t_wall > args.budget_s:
                res["variants"][name]["optstep_skipped"] = "budget"
                continue
            s = optstep_s(advs[name])
            res["variants"][name]["optstep_s"], res["variants"][name]["optstep_speedup"] = round(s, 4), round(s_ref / s, 3)
            if i == 0:
                print(f"SEED2_OPTSTEP {name} s_ref={s_ref:.4f} s_cand={s:.4f} speedup={s_ref / s:.3f} "
                      f"(fastest passing; 12 x 40 steps + coarse)", flush=True)
    else:
        print("ERROR SEED2_NO_PASSING_VARIANT", flush=True)

    # ---- optional: standalone FFT costs (does C2C avoid the C2R copy + scal?) ----------------------------------------
    if time.time() - t_wall < args.budget_s:
        try:
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
            del r64, c128, cases, r
        except Exception as exc:
            res["fft_ms_per_call_4096"] = {"error": f"{type(exc).__name__}: {exc}"[:500]}

    ms = jax.devices()[0].memory_stats() or {}
    res["jax_memory_GiB"] = {k: round(ms[k] / 2**30, 2) for k in ("bytes_in_use", "peak_bytes_in_use") if k in ms}
    res["wall_s"] = round(time.time() - t_wall, 1)
    path = os.path.join(args.out, "seed2.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"SEED2_DONE {path} passing={passed} fastest={res['fastest_passing']} wall={res['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:                                     # one greppable line, then the traceback
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR SEED2_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
