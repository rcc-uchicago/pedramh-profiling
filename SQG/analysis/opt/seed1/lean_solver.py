"""lean_solver.py -- seed 1: the A-SQG solver step in float64 JAX without its dead work (same mathematics).

`make_advance(GD, nsteps, ...)` builds a drop-in replacement for o1.2's
    advance = jax.jit(lambda st: lax.scan(GD.inner, st, None, length=STRIDE)[0])
from the imported, unmodified A-SQG_data module GD (hash-pinned; nothing here edits it). It takes and returns the same
pyqg_jax AB3State, so coarse(), checkpoints and the training loop see no difference. It is valid once the AB3 level
is 2, i.e. for any state at least two steps after a (re)start. `preconditions` checks every exactness claim below and
refuses to build if one fails.

What changes against GD.inner, and why the computed values do not:
  1. pyqg's spectral filter is not applied. filterfac = 0 makes `filtr` exactly 1 everywhere (checked): x*1 = x.
  2. pyqg's background-gradient term -(i k_x Q_y) psi_hat is not added. Q_y = beta = 0 (checked), so its coefficient
     is exactly 0 and the term only ever added +-0.
  3. U = 0: `(u + Ubg) * q` is `u * q` (XLA already dropped the +0 in production).
  4. The fixed forcing F0_hat is non-zero only in its first few k_x columns (5 for the 3.5 +- 0.5 ring). It is added
     as those columns zero-padded inside the fused kernel, instead of a dense 134 MB complex array read every step.
     The padded array equals F0_hat bit for bit (checked).
  5. decay multiplies as a real float64 array passed at run time (67 MB). XLA had folded it into a 134 MB complex
     constant; (a+ib)*(d+0i) and (a+ib)*d give the same values.
  6. The SQG inversion factor (f_0/N_b) sqrt(1/|k|^2) is evaluated once, with the same IEEE operations, instead of
     every step.
  7. AB3 at level 2 with constant coefficients: pyqg_jax's runtime lax.switch on _ablevel (one host sync per step) is
     gone. Job A measured this change bitwise identical.
  8. The two stored tendencies live in a 3-slot ring and the scan body is unrolled by 3, so the slots change roles:
     each new tendency is written into the slot whose old value is dead. This removes the two full-field copies per
     step of pyqg_jax's history rotation `new_updates = (updates, updates_p)` under lax.scan.
  9. (round 1b) The float32 time counter t advances by a run-time scalar consts["tdt"] = float32(dt), one add per
     step as pyqg. Round 1 (job 7814300) used a compile-time f32(dt), and t came out different from the reference
     after 40 steps, probably because XLA folded the unrolled (t + dt) + dt. No field reads t.
Options (neither changes a value):
  unroll1  multiplies one operand of each complex128 loop fusion by cos(x*0) == 1, so the fusion contains a cosine.
           XLA's loop emitter then gives each thread 1 element instead of 4. Nsight Compute (job 7814044) found that
           the 4-element complex128 pattern makes the AB3 fusion read 4.0 GB from DRAM for ~1.0 GB of unique data.
  split    puts an optimization barrier between the tendency and the AB3 update, i.e. two fusions with fewer operands.

The module imports only the standard library at import time (jax and numpy inside the functions), so the ring logic
can be unit-tested without JAX: test_lean_ring.py.
"""


class PreconditionError(RuntimeError):
    """A configuration fact the lean step relies on does not hold; the lean step must not be used."""


def ring_advance(step, qh, newest, prev, t, tc, nsteps, scan, zeros_like):
    """`nsteps` AB3 steps with the stored tendencies in 3 rotating slots.

    step(qh, T_prev, T_prevprev, t, tc) -> (qh, T_new, t, tc) is one AB3 step. `newest`/`prev` are pyqg_jax's
    _updates[0]/_updates[1]. scan(body, init, xs, length) is lax.scan; zeros_like makes the third slot, whose initial
    content is never read. Returns (qh, newest, prev, t, tc).

    At the top of each 3-step iteration slot A holds the newest tendency, B the one before and C nothing live. Every
    step writes its tendency into the slot that the previous step read for the last time, and after three steps the
    roles are back where they started, so no slot is ever copied into another.
    """
    nloop, rem = divmod(nsteps, 3)
    if nloop:
        def body(carry, _):
            qh, A, B, C, t, tc = carry
            qh, C, t, tc = step(qh, A, B, t, tc)     # newest -> C (nothing live there)
            qh, B, t, tc = step(qh, C, A, t, tc)     # newest -> B (read for the last time by the step above)
            qh, A, t, tc = step(qh, B, C, t, tc)     # newest -> A (read for the last time by the step above)
            return (qh, A, B, C, t, tc), None
        qh, newest, prev, _, t, tc = scan(body, (qh, newest, prev, zeros_like(newest), t, tc), None, nloop)[0]
    for _ in range(rem):
        qh, T, t, tc = step(qh, newest, prev, t, tc)
        newest, prev = T, newest
    return qh, newest, prev, t, tc


def preconditions(GD):
    """Check every fact that makes the lean step compute the same values as GD.inner.

    Returns (info dict, ncol): ncol = number of leading k_x columns that hold the forcing. Raises PreconditionError.
    """
    import numpy as np
    import jax.numpy as jnp
    from pyqg_jax import steppers as pst

    m = GD.sm.model
    info = {}

    def need(name, ok, **detail):
        info[name] = {"ok": bool(ok), **detail} if detail else bool(ok)
        if not ok:
            raise PreconditionError(name)

    need("stepper_is_AB3", isinstance(GD.sm.stepper, pst.AB3Stepper))
    need("one_layer", m.nz == 1)
    need("filterfac_is_0", float(m.filterfac) == 0.0)
    need("filtr_all_1", bool(np.all(np.asarray(m.filtr) == 1.0)))
    need("beta_is_0", float(m.beta) == 0.0)
    need("ikQy_all_0", bool(np.all(np.asarray(m._ikQy) == 0)))
    need("U_is_0", float(m.U) == 0.0 and bool(np.all(np.asarray(m.Ubg) == 0)))
    need("rek_is_0", float(m.rek) == 0.0)
    need("f0_over_Nb_is_1", float(m.f_0) / float(m.Nb) == 1.0)
    F0 = np.ascontiguousarray(np.asarray(GD.F0_hat))
    need("forcing_c128_1xNxnk", F0.dtype == np.complex128 and F0.ndim == 3 and F0.shape[0] == 1)
    nk = F0.shape[-1]
    cols = np.nonzero(np.any(F0[0] != 0, axis=0))[0]
    need("forcing_nonzero", cols.size > 0)
    ncol = int(cols.max()) + 1
    padded = np.ascontiguousarray(np.asarray(jnp.pad(jnp.asarray(F0[0, :, :ncol]), ((0, 0), (0, nk - ncol))))[None])
    need("forcing_pad_bitwise", padded.shape == F0.shape and np.array_equal(padded.view(np.uint64), F0.view(np.uint64)),
         ncol=ncol, nonzero_entries=int(np.count_nonzero(F0)))
    decay = np.asarray(GD.decay)
    need("decay_real_f64_Nxnk", decay.dtype == np.float64 and decay.shape == F0.shape[1:])
    return info, ncol


def build_consts(GD, ncol):
    """Run-time operands of the lean step (arguments, not closure constants, so XLA cannot fold them into dense
    constants): the SQG inversion factor, the real decay factor, the forcing's non-zero columns and the t increment."""
    import jax.numpy as jnp
    m = GD.sm.model
    return {
        "s": (m.f_0 / m.Nb) * jnp.sqrt(m.wv2i),        # the same IEEE ops as pyqg's _apply_a_ph, evaluated once
        "decay": jnp.asarray(GD.decay, jnp.float64),
        "frows": jnp.asarray(GD.F0_hat[0, :, :ncol]),
        "tdt": jnp.float32(GD.sm.stepper.dt),          # pyqg_jax: t + jnp.float32(dt) per step (item 9)
    }


def make_advance(GD, nsteps, *, ncol, unroll1=False, split=False):
    """jax.jit-compiled f(st, consts) -> the AB3State `nsteps` steps later (consts = build_consts(GD, ncol)).

    The step mirrors GD.inner operation for operation (same operand order, same complex arithmetic) minus the dead
    terms listed in the module docstring, so XLA has the chance to produce the same bits.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from pyqg_jax import steppers as pst
    try:
        barrier = jax.lax.optimization_barrier
    except AttributeError:                                     # jax 0.4.28 keeps it private
        from jax._src.ad_checkpoint import _optimization_barrier as barrier

    m = GD.sm.model
    dt = GD.sm.stepper.dt
    N = int(GD.N)
    nk = N // 2 + 1
    shape = (N, N)
    ik_b = jnp.expand_dims(m._ik, (0, 1))                       # i k_x, (1, 1, nk)
    il_b = jnp.expand_dims(m._il, (0, -1))                      # i k_y, (1, N, 1)
    neg_il_b = jnp.negative(il_b)
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt  # pyqg_jax AB3Stepper level-2 branch, verbatim
    f32 = jnp.float32

    def step(qh, Tp, Tpp, t, tc, c):
        # Both scalars below come from the loop carry, so XLA can neither fold them nor hoist what they feed out of
        # the loop: `one` keeps the forcing's zero-padding inside the AB3 kernel, `zero` feeds the unroll1 cosines.
        zero = (t * f32(0)).astype(jnp.float64)                 # +-0
        one = t * f32(0) + f32(1)                               # 1.0 exactly
        s = c["s"]
        if unroll1:
            s = s * jnp.cos(jnp.real(qh[0]) * zero)             # == s; the cosine makes XLA use 1 element per thread
        ph = s * qh                                             # psi_hat = (f_0/N_b) |k|^-1 theta_hat
        uh = neg_il_b * ph
        vh = ik_b * ph
        u = jnp.fft.irfftn(uh, axes=(-2, -1), s=shape)
        v = jnp.fft.irfftn(vh, axes=(-2, -1), s=shape)
        q = jnp.fft.irfftn(qh, axes=(-2, -1), s=shape)
        uqh = jnp.fft.rfftn(u * q, axes=(-2, -1))
        vqh = jnp.fft.rfftn(v * q, axes=(-2, -1))
        F = jnp.pad(c["frows"] * one, ((0, 0), (0, nk - ncol)))[None]   # == F0_hat bit for bit
        T = jnp.negative(ik_b * uqh + il_b * vqh) + F           # pyqg dqhdt (beta term dropped) + forcing
        if split:
            T = barrier(T)
        new = qh + c1 * T + c2 * Tp + c3 * Tpp                  # AB3 level 2, pyqg's operation order
        dfac = c["decay"]
        if unroll1:
            dfac = dfac * jnp.cos(jnp.imag(qh[0]) * zero)       # == decay (imag, not real: no CSE with the one above)
        return new * dfac, T, t + c["tdt"], tc + 1                # run-time dt: no folding (item 9)

    def advance(st, c):
        def stp(qh, Tp, Tpp, t, tc):
            return step(qh, Tp, Tpp, t, tc, c)

        def scan(body, init, xs, length):
            return lax.scan(body, init, xs, length=length)

        qh, newest, prev, t, tc = ring_advance(stp, st.state.qh, st._updates[0].qh, st._updates[1].qh,
                                               st.t, st.tc, nsteps, scan, jnp.zeros_like)
        return pst.AB3State(state=st.state.update(qh=qh), t=t, tc=tc, _ablevel=jnp.uint8(2),
                            _updates=(st._updates[0].update(qh=newest), st._updates[1].update(qh=prev)))

    return jax.jit(advance)


def o12_advance(GD, stride, **opts):
    """What o1.2 would use instead of its `advance`: st -> st after `stride` steps (checks preconditions first)."""
    _, ncol = preconditions(GD)
    consts = build_consts(GD, ncol)
    fn = make_advance(GD, stride, ncol=ncol, **opts)
    return lambda st: fn(st, consts)
