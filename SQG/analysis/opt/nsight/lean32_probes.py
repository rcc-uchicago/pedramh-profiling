"""lean32_probes.py -- exact rewrites of seed 2's `lean32` solver call, timed and profiled by the Nsight job.

Each probe computes lean32's arithmetic (seed2_opt.make_step("lean32", ...)) with a different schedule, so it should
match seed2_opt's lean32 bit for bit; profile_lean_nsight.py checks that before it times anything:
  lean32_r2        lean32 rebuilt here: 2-slot ring, scan unrolled by 2, as seed2_opt.make_variant. The control: if it
                   is not bitwise equal to seed 2's lean32, this module's step is not seed 2's and the probes mean nothing.
  lean32_u1        + seed 1's cos(x*0) trigger (lean_solver.py, `unroll1`): the 1/|k| and decay tables are multiplied by
                   cos(real / imag(theta_hat) * zero), zero = t*0 from the loop carry, i.e. exactly 1. In float64 it
                   made XLA give each thread 1 element instead of 4 (seed 1: 8.00 -> 6.40 ms/step); job F found the
                   4-element pattern reading 4x the AB3 fusion's data.
  lean32_ring3     seed 1's 3-slot ring (lean_solver.ring_advance: scan unrolled by 3, then 1 remainder step). lean32's
                   2-slot ring still has 2 full-size tendency copies per loop iteration (wrapped_copy.3/.4 in job
                   7815096's hlo_lean32.txt); seed 1's 3-slot loop bodies have none (job 7815637's hlo_seed1_lean*.txt).
  lean32_u1_ring3  both.
Valid where seed 2's lean32 is (AB3 level 2; filterfac = beta = U = rek = 0; forcing inside the 8x8 corners):
tables32() runs seed2_opt's start-up checks.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)                                          # SQG/analysis/opt
sys.path.insert(0, os.path.join(OPT, "seed2"))
sys.path.insert(0, os.path.join(OPT, "seed1"))
import seed2_opt as S2                                               # noqa: E402  (build_w, PATCH, make_variant)
import lean_solver as LS                                             # noqa: E402  (ring_advance; stdlib at import)

PROBES = {"lean32_r2": {"u1": False, "ring": 2}, "lean32_u1": {"u1": True, "ring": 2},
          "lean32_ring3": {"u1": False, "ring": 3}, "lean32_u1_ring3": {"u1": True, "ring": 3}}


def tables32(GD):
    """seed2_opt.main()'s run-time tables, cast to float32 / complex64 as it does for lean32, after its config checks."""
    import numpy as np
    import jax.numpy as jnp
    N, DT, mdl = GD.N, GD.DT, GD.sm.model
    dropped = {"filterfac": float(mdl.filterfac), "beta": float(mdl.beta), "U": float(mdl.U), "rek": float(mdl.rek)}
    if any(v != 0.0 for v in dropped.values()) or GD.sm.stepper.dt != DT or N & (N - 1):
        raise SystemExit(f"ERROR NSIGHT_CONFIG_CHANGED {dropped} dt={GD.sm.stepper.dt} N={N}")
    F0 = np.asarray(GD.F0_hat[0])
    iy, ix = np.nonzero(F0)
    P = S2.PATCH
    if not (len(ix) and ix.max() < P and np.all((iy < P) | (iy >= N - P))):
        raise SystemExit("ERROR NSIGHT_FORCING_OUTSIDE_PATCH: the forcing has modes outside the 8x8 corner patches")
    t64 = {"kinv": (mdl.f_0 / mdl.Nb) * jnp.sqrt(mdl.wv2i), "kx": mdl.kk, "ky": mdl.ll, "decay": GD.decay,
           "f0top": jnp.asarray(F0[:P, :P]), "f0bot": jnp.asarray(F0[N - P:, :P]), "tdt": jnp.float32(DT)}
    down = {np.dtype("complex128"): jnp.complex64, np.dtype("float64"): jnp.float32}
    return {k: v.astype(down.get(np.dtype(v.dtype), v.dtype)) for k, v in t64.items()}


def make_step(n, nk, dt, u1):
    """seed2_opt.make_step("lean32", n, nk, dt), line for line, with the optional u1 factor and the step's time t as
    an extra argument (only u1 reads it). step(th, t_n, t_o, c, t) -> (th', T)."""
    import jax.numpy as jnp
    from jax import lax
    s = 1.0 / (n * n)
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt      # pyqg_jax AB3 level-2 coefficients, verbatim
    P = S2.PATCH

    def step(th, t_n, t_o, c, t):
        fc = jnp.complex64
        kx, ky, kinv, decay = c["kx"], c["ky"], c["kinv"], c["decay"]
        if u1:
            zero = t * jnp.float32(0)                                 # +0 from the carry: not foldable, not hoistable
            kinv = kinv * jnp.cos(jnp.real(th) * zero)                # == kinv; the cosine changes XLA's loop emitter
            decay = decay * jnp.cos(jnp.imag(th) * zero)              # == decay (imag: no CSE with the line above)
        ph = kinv * th                                                # psi_hat, exactly as seed 2
        p = jnp.fft.fft2(S2.build_w(ph, kx, ky, s).astype(fc))       # u - i v
        q = jnp.fft.irfft2(th.astype(fc), s=(n, n))                   # theta
        uqh = jnp.fft.rfft2(jnp.real(p) * q).astype(th.dtype)
        vqh = jnp.fft.rfft2(-jnp.imag(p) * q).astype(th.dtype)
        ik = lax.complex(jnp.zeros_like(kx), kx)
        il = lax.complex(jnp.zeros_like(ky), ky)
        f0 = (jnp.pad(c["f0top"], ((0, n - P), (0, nk - P)))
              + jnp.pad(c["f0bot"], ((n - P, 0), (0, nk - P))))
        t_new = -(ik[None, :] * uqh + il[:, None] * vqh) + f0
        th_new = (th + c1 * t_new + c2 * t_n + c3 * t_o) * decay
        return th_new, t_new

    return step


def make_call(pst, n, nk, dt, stride, *, u1=False, ring=2):
    """jax.jit f(AB3State, tables32) -> AB3State `stride` steps later, as seed2_opt.make_variant("lean32")'s call:
    the state is cast to complex64 on entry and back on exit; t advances by the run-time c["tdt"] once per step."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    if ring not in (2, 3) or (ring == 2 and stride % 2):
        raise ValueError(f"ring={ring} stride={stride}")
    step = make_step(n, nk, dt, u1)

    def unpack(st):
        return tuple(x.qh[0].astype(jnp.complex64) for x in (st.state, st._updates[0], st._updates[1]))

    def pack(st, th, t_n, t_o, t, tc):
        cd = st.state.qh.dtype
        u0, u1_ = st._updates
        return pst.AB3State(state=st.state.update(qh=th.astype(cd)[None]), t=t, tc=tc, _ablevel=jnp.uint8(2),
                            _updates=(u0.update(qh=t_n.astype(cd)[None]), u1_.update(qh=t_o.astype(cd)[None])))

    def call(st, c):
        th, t_n, t_o = unpack(st)
        if ring == 2:
            def two(carry, _):
                th, t_n, t_o, t, tc = carry
                th1, t1 = step(th, t_n, t_o, c, t)                    # t1 takes t_o's slot
                th2, t2 = step(th1, t1, t_n, c, t + c["tdt"])         # t2 takes t_n's slot
                return (th2, t2, t1, (t + c["tdt"]) + c["tdt"], (tc + 1) + 1), None
            out = lax.scan(two, (th, t_n, t_o, st.t, st.tc), None, length=stride // 2)[0]
            return pack(st, *out)

        def stp(qh, tp, tpp, t, tc):                                  # lean_solver.ring_advance's step signature
            th1, t1 = step(qh, tp, tpp, c, t)
            return th1, t1, t + c["tdt"], tc + 1

        def scan(body, init, xs, length):
            return lax.scan(body, init, xs, length=length)

        return pack(st, *LS.ring_advance(stp, th, t_n, t_o, st.t, st.tc, stride, scan, jnp.zeros_like))

    return jax.jit(call)
