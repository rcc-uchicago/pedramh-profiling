"""lean32_precise.py -- seed 2's float32 solver step with the two float32 precision losses removed.

o1.2 learns the 0.02 delta theta(t + 0.02) - theta(t). In float32 two things damage small increments:
  1. The damping factor decay = mask * exp(-(NU k^2 + MU/k^2) DT) is 1 - (1e-7 .. 5e-6). float32 numbers just below 1
     are 6e-8 apart, so the damping RATE of each mode is quantized: up to 20% off at k = 4-16, 5% in the forcing ring,
     0.1% at k = 1 (deterministic per mode, applied every step).
  2. Each step adds an increment to theta and rounds the sum to float32 (6e-8 of theta). Where the increment is only
     a few ulps of theta and varies slowly (the large scales, where o1.2's delta is smallest), the rounding goes the
     same way step after step: a bias, not noise. Any float32 STATE has this, whatever the damping's form.
test_f32_update.py (one forced mode at k = 8.4, 100,000 steps, emulated float32): the 4,000-step delta is 31% off
for lean32 and for lean32e, 8.8e-8 for lean32ek.
Variants (each a drop-in 40-step call AB3State -> AB3State, complex128 outside, as seed2_opt's lean32):
  lean32e   lean32 with the damping as theta' = s - e*s, e = float32(-expm1(-rate DT)) (1 where |k| > KMAX, 0 at k = 0),
            and the three AB3 terms summed before they are added to theta. Same bytes as lean32. A CONTROL: with a
            float32 state the decrement e*s (~1-2 ulps of theta for the weakly damped modes) is itself rounded onto
            theta's ulp grid, so this alone does not fix the bias (test_f32_update.py: same 31% delta error as lean32
            for the k = 8.4 mode).
  lean32ek  lean32e + compensated (Kahan / TwoSum) state: theta is carried as hi + lo, two complex64 arrays. Each step
            the rounding error of hi + increment (TwoSum) and of hi - e*hi (Fast2Sum, |e| <= 1) goes into lo, which
            is added back next step. At the call's exit hi + lo is summed in complex128; at entry the complex128
            state is split into hi = c64(theta) and lo = c64(theta - hi). One more complex64 array read and written per
            step (+67 MB each way).
Both: tendencies T_-1, T_-2, the FFTs and the products stay complex64 / float32 as in lean32; t advances by the run-time
c["tdt"]; the history is seed 2's 2-slot ring (scan unrolled by 2; an odd remainder step is done after the scan).
Valid where seed 2's lean32 is (AB3 level 2; filterfac = beta = U = rek = 0; forcing in the 8x8 corners).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(OPT, "seed2"))
sys.path.insert(0, os.path.join(OPT, "nsight"))
import seed2_opt as S2                                               # noqa: E402  (build_w, PATCH)
import lean32_probes as LP                                           # noqa: E402  (tables32: seed 2's checks)
import f32_update as FU                                              # noqa: E402  (the state update, unit-tested)

VARIANTS = {"lean32e": {"comp": False}, "lean32ek": {"comp": True}}


def tables(GD):
    """lean32's float32 tables plus `edec` = float32(1 - decay) computed as -expm1(-rate DT) in float64, with
    A-SQG_data.py's own definition of decay: (kap <= KMAX) * exp(-(NU kap^2 + MU pos / kapi^2) DT)."""
    import jax.numpy as jnp
    c = LP.tables32(GD)
    rate = GD.NU * GD.kap ** 2 + GD.MU * GD.pos / GD.kapi ** 2
    e64 = jnp.where(GD.kap <= GD.KMAX, -jnp.expm1(-rate * GD.DT), 1.0)
    c["edec"] = e64.astype(jnp.float32)
    return c


def edec_check(GD, c):
    """max |(1 - edec) - decay| / max(decay, tiny) over the modes, in float64: edec reproduces GD.decay."""
    import jax.numpy as jnp
    d = 1.0 - c["edec"].astype(jnp.float64)
    return float(jnp.max(jnp.abs(d - GD.decay)))


def make_step(n, nk, dt, comp):
    """step(th, lo, t_n, t_o, c) -> (th', lo', T). lo is the compensation (zeros and ignored if not comp)."""
    import jax.numpy as jnp
    from jax import lax
    s = 1.0 / (n * n)
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt      # pyqg_jax AB3 level-2 coefficients, verbatim
    P = S2.PATCH

    def step(th, lo, t_n, t_o, c):
        kx, ky = c["kx"], c["ky"]
        ph = c["kinv"] * th                                           # psi_hat from the float32 head, as lean32
        p = jnp.fft.fft2(S2.build_w(ph, kx, ky, s).astype(jnp.complex64))
        q = jnp.fft.irfft2(th, s=(n, n))
        uqh = jnp.fft.rfft2(jnp.real(p) * q).astype(th.dtype)
        vqh = jnp.fft.rfft2(-jnp.imag(p) * q).astype(th.dtype)
        ik = lax.complex(jnp.zeros_like(kx), kx)
        il = lax.complex(jnp.zeros_like(ky), ky)
        f0 = (jnp.pad(c["f0top"], ((0, n - P), (0, nk - P)))
              + jnp.pad(c["f0bot"], ((n - P, 0), (0, nk - P))))
        t_new = -(ik[None, :] * uqh + il[:, None] * vqh) + f0
        inc = (c1 * t_new + c2 * t_n) + c3 * t_o                      # summed first: one rounding at theta's scale
        if not comp:
            return FU.plain_update(th, inc, c["edec"]), lo, t_new
        hi1, lo1 = FU.compensated_update(th, lo, inc, c["edec"])
        return hi1, lo1, t_new

    return step


def make_call(pst, n, nk, dt, stride, *, comp=False):
    """jax.jit f(AB3State, tables) -> AB3State `stride` steps later (any stride >= 1)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    step = make_step(n, nk, dt, comp)

    def unpack(st):
        th64 = st.state.qh[0]
        hi = th64.astype(jnp.complex64)
        lo = (th64 - hi.astype(th64.dtype)).astype(jnp.complex64) if comp else jnp.zeros_like(hi)
        return hi, lo, st._updates[0].qh[0].astype(jnp.complex64), st._updates[1].qh[0].astype(jnp.complex64)

    def pack(st, hi, lo, t_n, t_o, t, tc):
        cd = st.state.qh.dtype
        th = hi.astype(cd) + lo.astype(cd) if comp else hi.astype(cd)
        u0, u1 = st._updates
        return pst.AB3State(state=st.state.update(qh=th[None]), t=t, tc=tc, _ablevel=jnp.uint8(2),
                            _updates=(u0.update(qh=t_n.astype(cd)[None]), u1.update(qh=t_o.astype(cd)[None])))

    def call(st, c):
        hi, lo, t_n, t_o = unpack(st)
        carry = (hi, lo, t_n, t_o, st.t, st.tc)
        if stride // 2:
            def two(cr, _):
                hi, lo, t_n, t_o, t, tc = cr
                hi1, lo1, t1 = step(hi, lo, t_n, t_o, c)              # t1 takes t_o's slot
                hi2, lo2, t2 = step(hi1, lo1, t1, t_n, c)             # t2 takes t_n's slot
                return (hi2, lo2, t2, t1, (t + c["tdt"]) + c["tdt"], (tc + 1) + 1), None
            carry = lax.scan(two, carry, None, length=stride // 2)[0]
        if stride % 2:
            hi, lo, t_n, t_o, t, tc = carry
            hi1, lo1, t1 = step(hi, lo, t_n, t_o, c)
            carry = (hi1, lo1, t1, t_n, t + c["tdt"], tc + 1)
        return pack(st, *carry)

    return jax.jit(call)
