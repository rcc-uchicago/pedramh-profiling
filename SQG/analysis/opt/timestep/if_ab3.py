"""if_ab3.py -- the exact integrating-factor AB3 step (IF-AB3) next to production's, for the time-step probe.

Production (pyqg_jax AB3 at level 2, A-SQG_data.py) steps
    prod:  theta' = E * (theta + b0 N_n + b1 N_{n-1} + b2 N_{n-2}),          E = mask * exp(-(NU k^2 + MU / k^2) h)
i.e. one E for all three history terms. The exact integrating factor (Lawson / IF-AB3; research round 2, TS-2)
multiplies each older tendency by the decay accumulated since it was computed:
    if:    theta' = E(h) * (theta + b0 N_n + b1 E(h1) N_{n-1} + b2 E(h1 + h2) N_{n-2})
(h1, h2 = the spacings of the history; uniform steps: E(h), E(h)^2). Derivation: w = e^{lambda t} theta obeys
dw/dt = e^{lambda t} N, and AB3 on w is exact when e^{lambda t} N is a quadratic in t. prod's missing factors cost a
first-order error ~ (lambda h / 2) h N per step (notes Part XII, P5: the D80/D30 ratio -1.42..-1.51, theory -1.5).

Everything here is float64 except make_call_if32. The 40-step call mirrors lean_solver.make_advance (seed 1,
bitwise production at h = DT) operation for operation, with two differences, so ONE compile serves every h:
  - the AB3 weights c["c1"], c["c2"], c["c3"] and the decay table are run-time operands (prod: the same values as
    the compile-time constants, so the result differs from lean_u1 at most by the rounding of b * T vs folded
    constants; the probe checks <= 1e-12 at h = DT);
  - scheme "if" multiplies the two older tendencies by E and E * E inside the same fusion.
transition(): the two steps after switching from DT history to a step h (variable-step AB3 weights from
dt_ladder.ab3_coeffs; for "if" also the elapsed-time factors E(h1), E(h1 + h2)).
make_call_if32: lean32ek_fast (taxonomy/lean32_fast.py: compensated float32, cos(x*0) trigger, 3-slot ring) with
the IF factors written exactly in float32 via edec = 1 - E: E*T = T - edec*T, E^2*T = T - edec*(2 - edec)*T.

The scalar forms (step_scalar) are pure Python: test_if_ab3.py checks IF-AB3's exactness and prod's first-order
error with the standard library only. jax is imported inside the functions.
"""
import math

H_FRAME = 2e-4          # o1.2's frame spacing (STRIDE x DT)
CHUNK = 40              # steps per jitted call, as o1.2's advance


def ab3_coeffs(h, h1, h2):
    """Variable-step AB3 weights (dt_ladder.ab3_coeffs, verbatim): the integrals over [t_n, t_n + h] of the quadratic
    Lagrange basis through t_n, t_n - h1, t_n - h1 - h2."""
    s = h1 + h2
    hh, hhh = h * h, h * h * h
    b0 = (hhh / 3 + (h1 + s) * hh / 2 + h1 * s * h) / (h1 * s)
    b1 = -(hhh / 3 + s * hh / 2) / (h1 * h2)
    b2 = (hhh / 3 + h1 * hh / 2) / (s * h2)
    return b0, b1, b2


def step_scalar(scheme, theta, hist, h, h1, h2, lam):
    """One step of dtheta/dt = -lam theta + N(t) for a scalar mode. hist = (N_n, N_{n-1}, N_{n-2}) at t_n, t_n - h1,
    t_n - h1 - h2. Returns theta at t_n + h."""
    b0, b1, b2 = ab3_coeffs(h, h1, h2)
    n0, n1, n2 = hist
    E = math.exp(-lam * h)
    if scheme == "prod":
        return E * (theta + b0 * n0 + b1 * n1 + b2 * n2)
    return E * (theta + b0 * n0 + b1 * math.exp(-lam * h1) * n1 + b2 * math.exp(-lam * (h1 + h2)) * n2)


def steps_after_switch(nsub):
    """(transition steps, uniform steps) in one 0.02 lag (100 frames) at h = H_FRAME / nsub from DT (= nsub 40)
    history."""
    if nsub == 40:
        return 0, 100 * 40
    return 2, 100 * nsub - 2


# ---- float64 (jax) ---------------------------------------------------------------------------------------------------
def make_advance_rt(GD, nsteps, *, ncol, scheme):
    """jit f(st, c) -> AB3State nsteps later. c = rt_consts(...). lean_solver.make_advance's step (unroll1 trigger on)
    with run-time AB3 weights; scheme "if" decays the two older tendencies by E, E*E."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from pyqg_jax import steppers as pst
    import lean_solver as LS
    if scheme not in ("prod", "if"):
        raise ValueError(scheme)
    m = GD.sm.model
    N = int(GD.N)
    nk = N // 2 + 1
    shape = (N, N)
    ik_b = jnp.expand_dims(m._ik, (0, 1))
    il_b = jnp.expand_dims(m._il, (0, -1))
    neg_il_b = jnp.negative(il_b)
    f32 = jnp.float32

    def step(qh, Tp, Tpp, t, tc, c):
        zero = (t * f32(0)).astype(jnp.float64)
        one = t * f32(0) + f32(1)
        s = c["s"] * jnp.cos(jnp.real(qh[0]) * zero)            # == s (seed 1's unroll1 trigger)
        ph = s * qh
        uh = neg_il_b * ph
        vh = ik_b * ph
        u = jnp.fft.irfftn(uh, axes=(-2, -1), s=shape)
        v = jnp.fft.irfftn(vh, axes=(-2, -1), s=shape)
        q = jnp.fft.irfftn(qh, axes=(-2, -1), s=shape)
        uqh = jnp.fft.rfftn(u * q, axes=(-2, -1))
        vqh = jnp.fft.rfftn(v * q, axes=(-2, -1))
        F = jnp.pad(c["frows"] * one, ((0, 0), (0, nk - ncol)))[None]
        T = jnp.negative(ik_b * uqh + il_b * vqh) + F
        dfac = c["decay"] * jnp.cos(jnp.imag(qh[0]) * zero)     # == decay
        if scheme == "prod":
            new = qh + c["c1"] * T + c["c2"] * Tp + c["c3"] * Tpp
        else:
            new = qh + c["c1"] * T + c["c2"] * (dfac * Tp) + c["c3"] * ((dfac * dfac) * Tpp)
        return new * dfac, T, t + c["tdt"], tc + 1

    def advance(st, c):
        def stp(qh, Tp, Tpp, t, tc):
            return step(qh, Tp, Tpp, t, tc, c)

        def scan(body, init, xs, length):
            return lax.scan(body, init, xs, length=length)

        qh, newest, prev, t, tc = LS.ring_advance(stp, st.state.qh, st._updates[0].qh, st._updates[1].qh,
                                                  st.t, st.tc, nsteps, scan, jnp.zeros_like)
        return pst.AB3State(state=st.state.update(qh=qh), t=t, tc=tc, _ablevel=jnp.uint8(2),
                            _updates=(st._updates[0].update(qh=newest), st._updates[1].update(qh=prev)))

    return jax.jit(advance)


def make_transition(GD, *, ncol, scheme):
    """jit f(st, c, b0, b1, b2, f1, f2) -> AB3State one step later: weights (b0, b1, b2), older tendencies multiplied
    by f1, f2 (prod: pass 1.0), damping c["decay"] = E(h). dt_ladder.make_var_step's body with the factors."""
    import jax
    import jax.numpy as jnp
    from pyqg_jax import steppers as pst
    m = GD.sm.model
    N = int(GD.N)
    nk = N // 2 + 1
    shape = (N, N)
    ik_b = jnp.expand_dims(m._ik, (0, 1))
    il_b = jnp.expand_dims(m._il, (0, -1))
    neg_il_b = jnp.negative(il_b)
    f32 = jnp.float32

    def one(st, c, b0, b1, b2, f1, f2):
        qh, Tp, Tpp, t = st.state.qh, st._updates[0].qh, st._updates[1].qh, st.t
        one_ = t * f32(0) + f32(1)
        ph = c["s"] * qh
        u = jnp.fft.irfftn(neg_il_b * ph, axes=(-2, -1), s=shape)
        v = jnp.fft.irfftn(ik_b * ph, axes=(-2, -1), s=shape)
        q = jnp.fft.irfftn(qh, axes=(-2, -1), s=shape)
        uqh = jnp.fft.rfftn(u * q, axes=(-2, -1))
        vqh = jnp.fft.rfftn(v * q, axes=(-2, -1))
        F = jnp.pad(c["frows"] * one_, ((0, 0), (0, nk - ncol)))[None]
        T = jnp.negative(ik_b * uqh + il_b * vqh) + F
        new = qh + b0 * T + b1 * (f1 * Tp) + b2 * (f2 * Tpp)
        return pst.AB3State(state=st.state.update(qh=new * c["decay"]), t=t + c["tdt"], tc=st.tc + 1,
                            _ablevel=jnp.uint8(2), _updates=(st._updates[0].update(qh=T), st._updates[1].update(qh=Tp)))

    return jax.jit(one)


def decay_table(GD, h):
    """A-SQG_data.py's decay with DT -> h (float64 (N, N/2+1)); GD.decay itself at h == DT."""
    import jax.numpy as jnp
    if h == GD.DT:
        return jnp.asarray(GD.decay, jnp.float64)
    return ((GD.kap <= GD.KMAX).astype(jnp.float64)
            * jnp.exp(-(GD.NU * GD.kap ** 2 + GD.MU * GD.pos / GD.kapi ** 2) * h))


def rt_consts(GD, base, h):
    """Run-time operands for step size h: lean_solver.build_consts' s / frows plus the AB3 weights, E(h), float32 dt."""
    import jax.numpy as jnp
    c1, c2, c3 = (23 / 12) * h, (-16 / 12) * h, (5 / 12) * h          # pyqg_jax's level-2 expressions
    return dict(s=base["s"], frows=base["frows"], decay=decay_table(GD, h), tdt=jnp.float32(h),
                c1=jnp.float64(c1), c2=jnp.float64(c2), c3=jnp.float64(c3))


class Ladder:
    """One 0.02 lag at h = H_FRAME / nsub from a level-2 state with DT history, for either scheme. Compiled calls are
    shared across h (run-time weights): CHUNK-step calls, the remainder call, the transition step."""

    def __init__(self, GD, ncol, base):
        self.GD, self.ncol, self.base = GD, ncol, base
        self._adv, self._tr, self._c = {}, {}, {}

    def adv(self, scheme, n):
        k = (scheme, n)
        if k not in self._adv:
            self._adv[k] = make_advance_rt(self.GD, n, ncol=self.ncol, scheme=scheme)
        return self._adv[k]

    def consts(self, h):
        if h not in self._c:
            self._c[h] = rt_consts(self.GD, self.base, h)
        return self._c[h]

    def run(self, st, scheme, nsub, on_chunk=None):
        import jax.numpy as jnp
        if scheme not in self._tr:
            self._tr[scheme] = make_transition(self.GD, ncol=self.ncol, scheme=scheme)
        DT = self.GD.DT
        h = DT if nsub == 40 else H_FRAME / nsub
        c = self.consts(h)
        ntr, nuni = steps_after_switch(nsub)
        if ntr:
            f64 = jnp.float64
            hist = ((DT, DT), (h, DT))                               # spacings at the two transition steps
            for h1, h2 in hist:
                b0, b1, b2 = ab3_coeffs(h, h1, h2)
                if scheme == "prod":
                    f1 = f2 = f64(1.0)
                else:
                    f1, f2 = decay_table(self.GD, h1), decay_table(self.GD, h1 + h2)
                st = self._tr[scheme](st, c, f64(b0), f64(b1), f64(b2), f1, f2)
        full, rem = divmod(nuni, CHUNK)
        call = self.adv(scheme, CHUNK)
        for _ in range(full):
            st = call(st, c)
            if on_chunk is not None:
                on_chunk(st)
        if rem:
            st = self.adv(scheme, rem)(st, c)
        return st


# ---- float32: lean32ek_fast with the IF factors ------------------------------------------------------------------------
def make_call_if32(pst, n, nk, dt, stride):
    """lean32_fast.make_call(mode="ek", u1=True, ring=3) with inc = c1 T + c2 (T_-1 - e T_-1) + c3 (T_-2 - e(2 - e)
    T_-2), e = c["edec"] = float32(1 - E). Same tables (lean32_precise.tables)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    import lean32_fast as LF
    import lean_solver as LS
    S2, FU = LF.S2, LF.FU
    s = 1.0 / (n * n)
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt
    P = S2.PATCH

    def step(th, lo, t_n, t_o, c, t, tc):
        kx, ky, kinv, edec = c["kx"], c["ky"], c["kinv"], c["edec"]
        zero = t * jnp.float32(0)
        kinv = kinv * jnp.cos(jnp.real(th) * zero)
        edec = edec * jnp.cos(jnp.imag(th) * zero)
        ph = kinv * th
        p = jnp.fft.fft2(S2.build_w(ph, kx, ky, s).astype(jnp.complex64))
        q = jnp.fft.irfft2(th, s=(n, n))
        uqh = jnp.fft.rfft2(jnp.real(p) * q).astype(th.dtype)
        vqh = jnp.fft.rfft2(-jnp.imag(p) * q).astype(th.dtype)
        ik = lax.complex(jnp.zeros_like(kx), kx)
        il = lax.complex(jnp.zeros_like(ky), ky)
        f0 = (jnp.pad(c["f0top"], ((0, n - P), (0, nk - P)))
              + jnp.pad(c["f0bot"], ((n - P, 0), (0, nk - P))))
        t_new = -(ik[None, :] * uqh + il[:, None] * vqh) + f0
        e2 = edec * (jnp.float32(2) - edec)                           # 1 - E^2, in float32 relative precision
        inc = (c1 * t_new + c2 * (t_n - edec * t_n)) + c3 * (t_o - e2 * t_o)
        hi1, lo1 = FU.compensated_update(th, lo, inc, edec)
        return hi1, lo1, t_new

    def unpack(st):
        th64 = st.state.qh[0]
        hi = th64.astype(jnp.complex64)
        lo = (th64 - hi.astype(th64.dtype)).astype(jnp.complex64)
        return hi, lo, st._updates[0].qh[0].astype(jnp.complex64), st._updates[1].qh[0].astype(jnp.complex64)

    def pack(st, hi, lo, t_n, t_o, t, tc):
        cd = st.state.qh.dtype
        th = hi.astype(cd) + lo.astype(cd)
        u0, u1_ = st._updates
        return pst.AB3State(state=st.state.update(qh=th[None]), t=t, tc=tc, _ablevel=jnp.uint8(2),
                            _updates=(u0.update(qh=t_n.astype(cd)[None]), u1_.update(qh=t_o.astype(cd)[None])))

    def call(st, c):
        hi, lo, t_n, t_o = unpack(st)

        def stp(qh, tp, tpp, t, tc):
            h, l = qh
            h1, l1, t1 = step(h, l, tp, tpp, c, t, tc)
            return (h1, l1), t1, t + c["tdt"], tc + 1

        def scan(body, init, xs, length):
            return lax.scan(body, init, xs, length=length)

        (h, l), t_n, t_o, t, tc = LS.ring_advance(stp, (hi, lo), t_n, t_o, st.t, st.tc, stride, scan, jnp.zeros_like)
        return pack(st, h, l, t_n, t_o, t, tc)

    return jax.jit(call)
