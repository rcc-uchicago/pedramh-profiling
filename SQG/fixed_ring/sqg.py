"""Forced-dissipative surface QG on top of pyqg_jax.sqg_model.SQGModel.

Nondimensional setup: L = 2*pi so wavenumbers are integers (dk = 1), and
f_0/Nb = 1 so the SQG inversion is psi_hat = theta_hat/|k| (Held et al. 1995).

Two useful facts used throughout (both verified in tests.ipynb):
  * surface KE = 0.5*<|grad psi|^2> = 0.5*<theta^2> exactly, because
    psi_hat = theta_hat/|k|  =>  k^2|psi_k|^2 = |theta_k|^2.
    So the KE spectrum IS the surface-buoyancy-variance spectrum: it is the
    FORWARD-cascading invariant, and Blumen's k^-5/3 range lives at k > kf.
  * the conserved "energy" that cascades INVERSELY is sum |theta_k|^2/(2|k|)
    (= 0.5*<psi*theta>); we monitor it to check the drag arrests it.

WARNING (measured, see notebook test 0): pyqg_jax's built-in `rek` friction adds
rek*k^2*psi_hat to dq/dt.  For QG psi_hat ~ -q_hat/k^2 so that damps, but for SQG
psi_hat = +theta_hat/|k|, so the built-in term is ANTI-damping (growth rate
+rek*|k|).  We therefore always pass rek=0.0 and apply our own drag here.
"""

import functools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from pyqg_jax import sqg_model, steppers, state as pst

jax.config.update("jax_enable_x64", True)  # only ENABLES f64; f32 stays the default dtype


class P(NamedTuple):
    """All hyperparameters. Nondimensional units: L=2*pi, f_0/Nb=1, and eps=1 sets the time unit.
    At the defaults this gives KE=3.6, theta_rms=u_rms=2.7, u_max=8.6."""

    nx: int = 512        # grid points per side; filter cuts at k=0.65*nx/2=166, so kf=14 -> 166 is 1.07 decades of inertial range
    L: float = 2 * np.pi # domain size; 2*pi makes wavenumbers integers (dk=1) so isotropic spectra bin exactly
    kf: float = 14.0     # forcing centre wavenumber. Trade-off, both measured: lower kf lengthens the forward range but
                         # shortens the inverse range kd->kf, forcing a harder drag that short-circuits the injection
                         # (kf=10 gave a forward flux of only 0.44*eps; kf=14 gives 0.61*eps with the same drag)
    dkf: float = 1.0     # forcing ring half-width, |k| in [kf-dkf, kf+dkf]; 3 shells = isotropic but still narrow-band
    eps: float = 1.0     # injection rate of surface KE = 0.5*<theta^2>; SETS THE UNITS. Eddy rate at kf is
                         # eps^(1/3)*kf^(2/3) = 5.8, so the turnover time at the forcing scale is 0.17
    r: float = 3.0       # large-scale drag rate at k << kd [1/time]; r=3 exceeds the eddy rate eps^(1/3)k^(2/3)=1.4..2.9
                         # for k<=3, so the inverse cascade is arrested there. Measured E(1)/E(peak) = 1e-4: no pileup
    nd: float = 4.0      # drag steepness, D(k) = r/(1+(k/kd)^nd). SCALE-SELECTIVE ON PURPOSE. A uniform drag (nd=0)
                         # strong enough to stop the k=1 pileup also kills the cascade at kf (measured nonlin/drag=0.34,
                         # 98% of the energy stuck in the forcing ring); nd=2 still ate 93% of the injection
    kd: float = 3.0      # drag cutoff wavenumber; D=r for k<kd, D=r*(kd/k)^4 above. D(kf)=r/475=0.006, i.e. 0.001 of the
                         # eddy rate at kf, so the drag is invisible to the forward inertial range
    dt: float = 2e-4     # timestep. Measured u_max=8.6 => CFL = u_max*dt/dx = 0.14 and dt*u_max*k_max = 0.29,
                         # comfortably inside the AB3 imaginary-axis stability limit (~0.72)
    filterfac: float = 23.6  # pyqg's exponential spectral filter, exp(-filterfac*(k*dx-0.65pi)^4) applied for k*dx>0.65pi.
                         # CHOICE: we keep this as the ONLY small-scale sink rather than adding explicit hyperviscosity,
                         # because it is pyqg-native, adds no timestep constraint, and is exactly 1 at kf (0.65*pi/dx=166
                         # >> kf=14) so it cannot fight the forcing ring. Verified: the flux plateau extends to k=166
                         # with no bottleneck pile-up just below the cutoff
    f64: bool = False    # float64. Default f32 is fine for trajectories, but the butterfly twins need it: an f32
                         # roundoff of ~1e-7 relative would swamp any delta0 below that


# ---------------------------------------------------------------- model pieces

def model(p):
    return sqg_model.SQGModel(
        nx=p.nx, L=p.L,
        rek=0.0,            # MUST be 0: built-in SQG friction has the wrong sign (see module docstring)
        filterfac=p.filterfac,
        beta=0.0, Nb=1.0, f_0=1.0, U=0.0,   # f_0/Nb=1 -> psi_hat=theta_hat/|k|; no beta, no background shear
        precision=pst.Precision.DOUBLE if p.f64 else pst.Precision.SINGLE,
    )


def aux(p):
    """Model, stepper, forcing-ring mask, drag factor, Hermitian weights, |k|."""
    m = model(p)
    sm = steppers.SteppedModel(m, steppers.AB3Stepper(dt=p.dt))
    kap = m.wv                                                    # |k| on the rfft half-grid, (ny, nx//2+1)
    ring = ((kap >= p.kf - p.dkf) & (kap <= p.kf + p.dkf)).astype(m._dtype_real)
    drag = p.r / (1.0 + (kap / p.kd) ** p.nd)                      # scale-selective large-scale drag rate D(k)
    decay = jnp.exp(-drag * p.dt).astype(m._dtype_real)           # exact integrating factor for the linear drag
    w = jnp.asarray(_herm_w(p.nx), m._dtype_real)                 # rfft double-counting weights for Parseval
    return sm, ring, decay, w, kap


def _herm_w(nx):
    """Weights making sum(w*|rfft2(f)/N|^2) == mean(f^2) on the rfft half-grid."""
    w = np.full(nx // 2 + 1, 2.0)
    w[0] = 1.0
    if nx % 2 == 0:
        w[-1] = 1.0
    return w[None, :]


def force(key, p, ring, dtype):
    """White-in-time, isotropic, ring-supported forcing increment (already * sqrt(dt)).

    Random phases with the amplitude renormalised every step so the injection is
    EXACTLY eps: theta -> theta + f*sqrt(dt) with f independent of theta gives
    d(0.5<theta^2>)/dt = 0.5*<f^2>, so we set rms(f) = sqrt(2*eps).
    """
    ny, nk = ring.shape
    n = jax.random.normal(key, (2, ny, nk), dtype=dtype)
    fh = ring * lax.complex(n[0], n[1])
    f = jnp.fft.irfft2(fh, s=(ny, p.nx))                       # real, ring-supported (irfft symmetrises)
    f = f * (jnp.sqrt(2.0 * p.eps) / jnp.sqrt(jnp.mean(f**2)))  # fix injection rate exactly
    return jnp.fft.rfft2(f)[None] * jnp.sqrt(p.dt)


def diag(qh, p, w, kap):
    """(KE, E) = (0.5<theta^2>, sum|theta_k|^2/(2|k|)); E is the inverse-cascading invariant."""
    a = w * jnp.abs(qh[0] / (p.nx * p.nx)) ** 2
    ke = 0.5 * jnp.sum(a)
    en = 0.5 * jnp.sum(jnp.where(kap > 0, a / jnp.where(kap > 0, kap, 1.0), 0.0))
    return ke, en


# ---------------------------------------------------------------- time stepping

@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def run(p, n_out, n_inner, snaps, st, key):
    """n_out outer steps of n_inner dt each. Returns (final state, t, ke, en[, theta]).

    Order inside one dt: AB3 advection (+ pyqg exponential filter) -> exact linear
    drag -> additive forcing increment. Forcing is added as an Euler-Maruyama
    increment rather than fed to AB3, which is the correct treatment of white noise.
    """
    sm, ring, decay, w, kap = aux(p)
    cdt = jnp.complex128 if p.f64 else jnp.complex64
    rdt = jnp.float64 if p.f64 else jnp.float32

    def inner(s, k):
        s = sm.step_model(s)
        qh = s.state.qh * decay + force(k, p, ring, rdt).astype(cdt)
        return s.update(state=s.state.update(qh=qh)), None

    def outer(s, k):
        s, _ = lax.scan(inner, s, jax.random.split(k, n_inner))
        ke, en = diag(s.state.qh, p, w, kap)
        out = (s.t, ke, en)
        if snaps:
            out = out + (jnp.fft.irfft2(s.state.qh[0], s=(p.nx, p.nx)).astype(jnp.float32),)
        return s, out

    return lax.scan(outer, st, jax.random.split(key, n_out))


def init(p, key, amp=1e-3):
    """Small random broadband theta, spectrally smooth, as a spinup seed."""
    sm, _, _, _, kap = aux(p)
    m = sm.model
    th = amp * jax.random.normal(key, (p.nx, p.nx), dtype=m._dtype_real)
    qh = jnp.fft.rfft2(th)[None] * jnp.exp(-((kap / p.kf) ** 2))  # seed near/below kf so spinup is short
    return sm.initialize_stepper_state(m.create_initial_state(key).update(qh=qh.astype(m._dtype_complex)))


def state_from_theta(p, th):
    """Wrap a physical theta field into a fresh stepper state (AB3 restarts; 2-step transient)."""
    sm, *_ = aux(p)
    m = sm.model
    qh = jnp.fft.rfft2(jnp.asarray(th, m._dtype_real))[None].astype(m._dtype_complex)
    return sm.initialize_stepper_state(m.create_initial_state(jax.random.key(0)).update(qh=qh))


def theta(st, p):
    return np.asarray(jnp.fft.irfft2(st.state.qh[0], s=(p.nx, p.nx)))


def steady(x, t=None):
    """Statistically steady if the linear drift over the last half of the record is smaller than
    2 sigma of that record. Comparing the drift to a fixed % of the mean does not work: KE here
    wanders by ~2.4% rms with a long autocorrelation, so any short window shows a spurious 'trend'."""
    x = np.asarray(x)
    h = len(x) // 2
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t)
    drift = np.polyfit(t[h:], x[h:], 1)[0] * (t[-1] - t[h])
    return bool(abs(drift) < 2 * x[h:].std()), float(drift), float(x[h:].std())


def spinup(p, key, n_out=200, n_inner=250):
    """Run from a small random seed to statistical steady state."""
    st = init(p, key)
    st, (t, ke, en) = run(p, n_out, n_inner, False, st, key)
    ke, en, t = np.asarray(ke), np.asarray(en), np.asarray(t)
    return st, t, ke, en, steady(ke, t)[0] and steady(en, t)[0]


# ---------------------------------------------------------------- diagnostics

def kespec(th, nx=None):
    """Isotropic surface-KE spectrum E(k), integers k=1..nx/2, with sum(E)=0.5<theta^2>.
    th: (..., ny, nx) real. Averages over leading axes."""
    th = np.asarray(th)
    nx = nx or th.shape[-1]
    a = 0.5 * _herm_w(nx) * np.abs(np.fft.rfft2(th, axes=(-2, -1)) / (nx * nx)) ** 2
    if a.ndim > 2:
        a = a.reshape(-1, *a.shape[-2:]).mean(0)
    kk = np.fft.rfftfreq(nx, 1.0 / nx)
    ll = np.fft.fftfreq(nx, 1.0 / nx)
    kap = np.sqrt(kk[None, :] ** 2 + ll[:, None] ** 2)
    ib = np.round(kap).astype(int)
    kmax = nx // 2
    e = np.bincount(ib.ravel(), a.ravel(), minlength=kmax + 1)[: kmax + 1]
    return np.arange(1, kmax + 1), e[1:]          # dk=1, so E(k) is already a density


def perturb(th, key, delta, kband=(80.0, 160.0)):
    """Add a random perturbation confined to high wavenumbers, with rms = delta*rms(theta).
    High-k on purpose: the butterfly test asks whether error injected at the smallest resolved
    scales climbs up-scale, so it must not project onto the energy-containing scales at t=0.
    Default kband suits nx=512; scale it with nx (the notebook uses [0.48,0.96]*0.65*nx/2)."""
    th = np.asarray(th)
    nx = th.shape[-1]
    kk = np.fft.rfftfreq(nx, 1.0 / nx)
    ll = np.fft.fftfreq(nx, 1.0 / nx)
    kap = np.sqrt(kk[None, :] ** 2 + ll[:, None] ** 2)
    m = ((kap >= kband[0]) & (kap <= kband[1])).astype(float)
    n = np.asarray(jax.random.normal(key, (2, *m.shape), dtype=jnp.float64))
    dth = np.fft.irfft2(m * (n[0] + 1j * n[1]), s=(nx, nx))
    return th + dth * (delta * th.std() / dth.std())


def perlin(nx, seed, std=1e-5, res=16):
    """Tileable 2-D Perlin noise field, rescaled to exactly `std`.

    Tileable is essential: the domain is periodic, so a non-tileable perturbation would
    inject a seam discontinuity that is not a physical SQG mode. Perlin power sits at
    LARGE scales (res=16 peaks at k~9, i.e. the energy-containing scale of this flow),
    unlike the high-k perturbation used by perturb() for the butterfly test."""
    import perlin_numpy
    a = perlin_numpy.generate_perlin_noise_2d(
        (nx, nx), (res, res), tileable=(True, True), rng=np.random.default_rng(seed))
    return (a - a.mean()) * (std / a.std())


def bvflux(th, p, nmax=32):
    """Spectral flux Pi(k) of surface KE (= buoyancy variance/2) across wavenumber k.
    A plateau at Pi = eps identifies the true forward inertial range.
    Pi(k) = -sum_{k'<=k} T(k'), T = rate of change of 0.5<theta^2> from advection alone."""
    sm, _, _, w, kap = aux(p)
    m = sm.model
    th = jnp.asarray(np.atleast_3d(np.asarray(th)).reshape(-1, p.nx, p.nx)[:nmax], m._dtype_real)
    base = m.create_initial_state(jax.random.key(0))

    def one(t):
        qh = jnp.fft.rfft2(t)[None].astype(m._dtype_complex)
        return jnp.real(jnp.conj(qh[0]) * m.get_updates(base.update(qh=qh)).qh[0])

    T = np.asarray(jax.vmap(one)(th).mean(0)) * np.asarray(w) / p.nx**4
    ib = np.round(np.asarray(kap)).astype(int)
    kmax = p.nx // 2
    tk = np.bincount(ib.ravel(), T.ravel(), minlength=kmax + 1)[: kmax + 1]
    return np.arange(1, kmax + 1), -np.cumsum(tk)[1:]


def save(path, **kw):
    np.savez_compressed(path, **kw)


def load_theta(path):
    """Last theta field from a checkpoint/trajectory .npz, or a bare .npy frame."""
    d = np.load(path)
    if isinstance(d, np.ndarray):                      # plain .npy
        return d[-1] if d.ndim == 3 else d
    return d["theta"][-1] if d["theta"].ndim == 3 else d["theta"]
