"""lean32_fast.py -- two float32 solver calls for the taxonomy job (item 3), built on lean32_precise.py's step.

  lean32ek_fast  lean32ek's arithmetic with the Nsight job's two schedule changes (lean32_u1_ring3, notes Part X):
                 seed 1's cos(x*0) trigger (u1: the 1/|k| and e tables are multiplied by cos(real / imag(theta) * zero),
                 zero = t*0 from the loop carry, i.e. exactly 1) and seed 1's 3-slot tendency ring. Every value is the
                 same as lean32ek's, so the job checks it bit for bit against lean_precise's lean32ek. Estimate ~2.7 ms
                 per step (lean32_u1_ring3 2.524 ms + the hi/lo array's ~0.15 ms); unmeasured.
  lean32sr       lean32e with the state rounded STOCHASTICALLY (the noise-only control from ANALOGUES.md; Paxton 2022,
                 Kimpson 2023): s = theta + inc and u = s - e*s are formed in float64 (exact for s at these
                 magnitudes), then u is rounded to one of its two float32 neighbours with probability proportional to
                 closeness, so E[SR(u)] = u. Mechanism B's systematic sub-ulp loss becomes a zero-mean random walk. The
                 random bits are jax.random.bits(fold_in(key, tc)), one uint64 per real component per step.
The tendency code is lean32_precise.make_step's, line for line. Valid where lean32 is (AB3 level 2; filterfac = beta
= U = rek = 0; forcing in the 8x8 corners).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(OPT, "delta"))
sys.path.insert(0, os.path.join(OPT, "seed1"))
import lean32_precise as LP                                          # noqa: E402  (S2, FU, tables)
import lean_solver as LS                                             # noqa: E402  (ring_advance)

S2, FU = LP.S2, LP.FU
SR_DROP = 29                                                         # float64 -> float32: 52 - 23 fraction bits


def sr_to_f32(x, bits):
    """Stochastic rounding of a float64 array to float32: add uniform random bits below float32's last place to the
    magnitude, then truncate (IEEE magnitudes are monotone in their bit patterns, so a carry lands in the next binade).
    Exact zeros stay zero. The truncated float64 has a zero low 29 bits, so the cast to float32 is exact."""
    import jax.numpy as jnp
    from jax import lax
    mask = jnp.uint64((1 << SR_DROP) - 1)
    b = lax.bitcast_convert_type(x, jnp.uint64)
    b = (b + (bits & mask)) & jnp.invert(mask)
    return lax.bitcast_convert_type(b, jnp.float64).astype(jnp.float32)


def make_step(n, nk, dt, mode, u1):
    """step(th, lo, t_n, t_o, c, t, tc) -> (th', lo', T). mode 'ek' = lean32ek's compensated update; 'sr' = the
    stochastically rounded plain update (lo unused)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    s = 1.0 / (n * n)
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt      # pyqg_jax AB3 level-2 coefficients, verbatim
    P = S2.PATCH

    def step(th, lo, t_n, t_o, c, t, tc):
        kx, ky, kinv, edec = c["kx"], c["ky"], c["kinv"], c["edec"]
        if u1:
            zero = t * jnp.float32(0)                                 # +0 from the carry: not foldable, not hoistable
            kinv = kinv * jnp.cos(jnp.real(th) * zero)                # == kinv
            edec = edec * jnp.cos(jnp.imag(th) * zero)                # == edec (imag: no CSE with the line above)
        ph = kinv * th                                                # psi_hat from the float32 head, as lean32
        p = jnp.fft.fft2(S2.build_w(ph, kx, ky, s).astype(jnp.complex64))
        q = jnp.fft.irfft2(th, s=(n, n))
        uqh = jnp.fft.rfft2(jnp.real(p) * q).astype(th.dtype)
        vqh = jnp.fft.rfft2(-jnp.imag(p) * q).astype(th.dtype)
        ik = lax.complex(jnp.zeros_like(kx), kx)
        il = lax.complex(jnp.zeros_like(ky), ky)
        f0 = (jnp.pad(c["f0top"], ((0, n - P), (0, nk - P)))
              + jnp.pad(c["f0bot"], ((n - P, 0), (0, nk - P))))
        t_new = -(ik[None, :] * uqh + il[:, None] * vqh) + f0
        inc = (c1 * t_new + c2 * t_n) + c3 * t_o                      # summed first, as lean32_precise
        if mode == "ek":
            hi1, lo1 = FU.compensated_update(th, lo, inc, edec)
            return hi1, lo1, t_new
        s64 = th.astype(jnp.complex128) + inc.astype(jnp.complex128)
        e64 = edec.astype(jnp.float64)
        u = s64 - e64 * s64
        bits = jax.random.bits(jax.random.fold_in(c["key"], tc), (2,) + u.shape, jnp.uint64)
        th1 = lax.complex(sr_to_f32(jnp.real(u), bits[0]), sr_to_f32(jnp.imag(u), bits[1]))
        return th1, lo, t_new

    return step


def make_call(pst, n, nk, dt, stride, *, mode, u1=False, ring=2):
    """jax.jit f(AB3State, tables) -> AB3State `stride` steps later; entry/exit exactly as lean32_precise.make_call
    (complex128 state outside; hi = c64(theta), lo = c64(theta - hi) for 'ek'). tables = LP.tables(GD) (+ 'key' for
    'sr'). ring 2 = lean32_precise's 2-slot scan (stride even); ring 3 = seed 1's 3-slot ring_advance."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    if mode not in ("ek", "sr") or ring not in (2, 3) or (ring == 2 and stride % 2):
        raise ValueError(f"mode={mode} ring={ring} stride={stride}")
    comp = mode == "ek"
    step = make_step(n, nk, dt, mode, u1)

    def unpack(st):
        th64 = st.state.qh[0]
        hi = th64.astype(jnp.complex64)
        lo = (th64 - hi.astype(th64.dtype)).astype(jnp.complex64) if comp else jnp.zeros_like(hi)
        return hi, lo, st._updates[0].qh[0].astype(jnp.complex64), st._updates[1].qh[0].astype(jnp.complex64)

    def pack(st, hi, lo, t_n, t_o, t, tc):
        cd = st.state.qh.dtype
        th = hi.astype(cd) + lo.astype(cd) if comp else hi.astype(cd)
        u0, u1_ = st._updates
        return pst.AB3State(state=st.state.update(qh=th[None]), t=t, tc=tc, _ablevel=jnp.uint8(2),
                            _updates=(u0.update(qh=t_n.astype(cd)[None]), u1_.update(qh=t_o.astype(cd)[None])))

    def call(st, c):
        hi, lo, t_n, t_o = unpack(st)
        if ring == 2:
            def two(cr, _):
                hi, lo, t_n, t_o, t, tc = cr
                hi1, lo1, t1 = step(hi, lo, t_n, t_o, c, t, tc)        # t1 takes t_o's slot
                hi2, lo2, t2 = step(hi1, lo1, t1, t_n, c, t + c["tdt"], tc + 1)
                return (hi2, lo2, t2, t1, (t + c["tdt"]) + c["tdt"], (tc + 1) + 1), None
            out = lax.scan(two, (hi, lo, t_n, t_o, st.t, st.tc), None, length=stride // 2)[0]
            return pack(st, *out)

        def stp(qh, tp, tpp, t, tc):                                  # lean_solver.ring_advance's step signature
            h, l = qh
            h1, l1, t1 = step(h, l, tp, tpp, c, t, tc)
            return (h1, l1), t1, t + c["tdt"], tc + 1

        def scan(body, init, xs, length):
            return lax.scan(body, init, xs, length=length)

        (h, l), t_n, t_o, t, tc = LS.ring_advance(stp, (hi, lo), t_n, t_o, st.t, st.tc, stride, scan, jnp.zeros_like)
        return pack(st, h, l, t_n, t_o, t, tc)

    return jax.jit(call)
