"""dt_ladder.py -- production's A-SQG step at another time step h, started from a state whose AB3 history is at DT.

Item 3 of the 2026-11-06 team plan (TEAM_REPORT.md, "dt ladder"): how much does o1.2's 0.02 delta change if the
solver steps at h = 2e-4/n instead of DT = 5e-6 = 2e-4/40? n = 30 and 32 are the candidate larger steps (1.33x and
1.25x fewer steps); n = 80 (2.5e-6) is the Richardson arm that estimates production's own time error.

The step is seed 1's float64 lean step (lean_solver.make_advance, bitwise production at h = DT) with three changes for
h != DT, all in the integrating-factor AB3 scheme production uses, theta' = (theta + sum_j b_j T_j) * decay_h:
  1. decay_h = (|k| <= KMAX) exp(-(NU |k|^2 + MU / |k|^2) h), A-SQG_data.py's own formula with DT -> h.
  2. The first two steps after the switch use variable-step AB3 weights (ab3_coeffs): the history T(t_n), T(t_n - DT),
     T(t_n - 2 DT) was taken at spacing DT. Step 1 uses spacings (DT, DT), step 2 (h, DT); from step 3 the weights are
     uniform, (23/12, -16/12, 5/12) h.
  3. The float32 time counter advances by float32(h).
At h = DT the ladder is 100 calls of make_advance(<GD with dt = DT>, 40, unroll1=True) with GD.decay, built through the
same dt shim as every other h: the same code as lean_u1, so its frames must equal lean_u1's bit for bit (the job's
control of the shim). The transition steps mirror lean_solver's step operation for operation, with the three weights as
run-time scalars.

ab3_coeffs is pure arithmetic (int / Fraction / float): test_dt_ladder.py checks it in exact arithmetic with the
standard library only. jax is imported inside the functions.
"""

H_FRAME = 2e-4          # o1.2's frame spacing (STRIDE x DT); every ladder step divides it
LAG_T = 0.02            # o1.2's pair lag


def ab3_coeffs(h, h1, h2):
    """Variable-step AB3 weights (b0, b1, b2) for a step of size h from t_n with the history T(t_n), T(t_n - h1),
    T(t_n - h1 - h2): the integrals over [t_n, t_n + h] of the quadratic Lagrange basis through those three times.
    Exact in any field. h1 = h2 = h gives (23/12 h, -16/12 h, 5/12 h)."""
    s = h1 + h2
    hh, hhh = h * h, h * h * h
    b0 = (hhh / 3 + (h1 + s) * hh / 2 + h1 * s * h) / (h1 * s)
    b1 = -(hhh / 3 + s * hh / 2) / (h1 * h2)
    b2 = (hhh / 3 + h1 * hh / 2) / (s * h2)
    return b0, b1, b2


def transition(h, h0):
    """The weights of the two steps after switching from spacing h0 to h: [(b0, b1, b2) of step 1, of step 2]."""
    return [ab3_coeffs(h, h0, h0), ab3_coeffs(h, h, h0)]


def steps_per_lag(nsub):
    """Steps of size H_FRAME / nsub in one 0.02 lag (100 frames)."""
    return 100 * nsub


def decay_table(GD, h):
    """A-SQG_data.py's `decay` with DT replaced by h (float64, (N, N/2+1)). For h == GD.DT this is GD.decay itself."""
    import jax.numpy as jnp
    if h == GD.DT:
        return jnp.asarray(GD.decay, jnp.float64)
    return ((GD.kap <= GD.KMAX).astype(jnp.float64)
            * jnp.exp(-(GD.NU * GD.kap ** 2 + GD.MU * GD.pos / GD.kapi ** 2) * h))


def make_var_step(GD, ncol):
    """jit f(st, consts, b0, b1, b2, decay, tdt) -> st one step later with AB3 weights (b0, b1, b2) and damping `decay`.
    The body is lean_solver.make_advance's step without unroll1/split, operation for operation."""
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

    def one(st, c, b0, b1, b2, decay, tdt):
        qh, Tp, Tpp, t = st.state.qh, st._updates[0].qh, st._updates[1].qh, st.t
        one_ = t * f32(0) + f32(1)
        ph = c["s"] * qh
        uh = neg_il_b * ph
        vh = ik_b * ph
        u = jnp.fft.irfftn(uh, axes=(-2, -1), s=shape)
        v = jnp.fft.irfftn(vh, axes=(-2, -1), s=shape)
        q = jnp.fft.irfftn(qh, axes=(-2, -1), s=shape)
        uqh = jnp.fft.rfftn(u * q, axes=(-2, -1))
        vqh = jnp.fft.rfftn(v * q, axes=(-2, -1))
        F = jnp.pad(c["frows"] * one_, ((0, 0), (0, nk - ncol)))[None]
        T = jnp.negative(ik_b * uqh + il_b * vqh) + F
        new = qh + b0 * T + b1 * Tp + b2 * Tpp
        return pst.AB3State(state=st.state.update(qh=new * decay), t=t + tdt, tc=st.tc + 1, _ablevel=jnp.uint8(2),
                            _updates=(st._updates[0].update(qh=T), st._updates[1].update(qh=Tp)))

    return jax.jit(one)


def make_ladder(GD, nsub, ncol, stride=40):
    """f(st) -> the AB3State after one 0.02 lag at h = H_FRAME / nsub, from a level-2 state with DT history.

    The uniform steps run as `stride`-step calls of lean_solver.make_advance(<GD with dt = h>, stride, unroll1=True),
    plus one shorter call for the remainder, i.e. the same call shape as lean_u1 in o1.2. nsub = 40 (h = DT): no
    transition, GD.decay, 100 such calls, built through the same dt shim, so it must equal lean_u1 bit for bit (the
    control). Otherwise: 2 transition steps, then steps_per_lag(nsub) - 2 uniform steps at h."""
    import types
    import jax.numpy as jnp
    import lean_solver as LS

    DT = GD.DT
    n = steps_per_lag(nsub)
    h = DT if nsub == 40 else H_FRAME / nsub
    consts = LS.build_consts(GD, ncol)
    shim = types.SimpleNamespace(N=GD.N, sm=types.SimpleNamespace(model=GD.sm.model,
                                                                  stepper=types.SimpleNamespace(dt=h)))
    dec = decay_table(GD, h)
    ch = dict(consts, decay=dec, tdt=jnp.float32(h))
    n_uni = n if nsub == 40 else n - 2
    full, rem = divmod(n_uni, stride)
    call = LS.make_advance(shim, stride, ncol=ncol, unroll1=True)
    tail = LS.make_advance(shim, rem, ncol=ncol, unroll1=True) if rem else None
    var = make_var_step(GD, ncol) if nsub != 40 else None
    w = transition(h, DT) if nsub != 40 else []
    f64 = jnp.float64

    def run(st):
        for b0, b1, b2 in w:
            st = var(st, consts, f64(b0), f64(b1), f64(b2), dec, jnp.float32(h))
        for _ in range(full):
            st = call(st, ch)
        return tail(st, ch) if tail is not None else st

    return run
