"""test_hybrid_math.py -- the index algebra of hybrid_callbacks.py / hybrid_step.py on a small grid (numpy only; runs on
Python 3.6 + numpy 1.17, so on a login node). Prints HYBRID_MATH_TEST_OK.

hybrid_callbacks.py's own source runs here under a tiny `numba` shim (int64 = int, float32/64 = float, complex64/128 =
complex), with its module constants rebound to N = 48, KM = 16. So the callbacks' index logic is tested, not a copy.
The Warp update's Hermitian split and tendency are restated in numpy (they are a few lines; the GPU job compares the
full step against the JAX steps). Checks, all in float64:
  1. p = fft2(w) is u - i v, with u = irfft2(-i k_y psi_hat), v = irfft2(i k_x psi_hat) (lean_u1's convention), w from
     seed2_opt.build_w's formula (restated in numpy below);
  2. the load callback for w reproduces build_w on every element (both halves, the mirror, the mask);
  3. the C2R load callback reproduces theta_hat on the half plane;
  4. Z = conj(p) theta through the z callback, fft2, the store callback, then the Hermitian split, gives
     rfft2(u theta) and rfft2(v theta) on the kept half block, and the tendency -(i k_x A + i k_y B).
"""
import os
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Nt, KMt = 48, 16


def load_callbacks():
    src = open(os.path.join(HERE, "hybrid_callbacks.py")).read().replace("import numba\n", "")
    shim = types.SimpleNamespace(int64=int, float32=float, float64=float, complex64=complex, complex128=complex)
    ns = {"numba": shim, "__name__": "hybrid_callbacks_shim"}
    exec(compile(src, "hybrid_callbacks.py", "exec"), ns)
    cr, cc = 2 * KMt + 1, KMt + 1
    ns.update(N=Nt, NK=Nt // 2 + 1, KM=KMt, CR=cr, CC=cc, B=cr * cc, B2=2 * cr * cc, HALFN=Nt // 2,
              S32=1.0 / (Nt * Nt), S64=1.0 / (Nt * Nt), ZERO32=0j, ZERO64=0j)
    return ns


def build_w_ref(ph, kx, ky, s):
    """seed2_opt.build_w in numpy."""
    n, half = ph.shape[0], ph.shape[0] // 2
    w_l = (-s * kx[None, :] + 1j * s * ky[:, None]) * np.conj(ph)
    rows = ph[(-np.arange(n)) % n, :]                                 # row r <- (-r) mod n
    ph_r = rows[:, 1:half][:, ::-1]                                   # columns half-1 .. 1 (= N - j)
    kx_r = kx[1:half][::-1]
    m_r = s * kx_r[None, :] + 1j * s * ky[:, None]
    return np.concatenate([w_l, m_r * ph_r], 1)


def check(name, ok, detail=""):
    if not ok:
        raise SystemExit("ERROR HYBRID_MATH_TEST %s %s" % (name, detail))
    print("ok %s %s" % (name, detail))


def main():
    rng = np.random.RandomState(0)
    n, nk, km = Nt, Nt // 2 + 1, KMt
    cr, cc, b = 2 * km + 1, km + 1, (2 * km + 1) * (km + 1)
    s = 1.0 / (n * n)
    kx = np.arange(nk, dtype=float)
    ky = np.fft.fftfreq(n, 1.0 / n)
    kap = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    th = np.fft.rfft2(rng.standard_normal((n, n)))
    th[kap > km] = 0                                                  # the 2/3 mask (radial), as the decay table
    kinv = np.where(kap > 0, 1.0 / np.where(kap > 0, kap, 1.0), 0.0)
    ph = kinv * th
    # 1. sign convention
    w = build_w_ref(ph, kx, ky, s)
    p = np.fft.fft2(w)
    u = np.fft.irfft2(-1j * ky[:, None] * ph, s=(n, n))
    v = np.fft.irfft2(1j * kx[None, :] * ph, s=(n, n))
    e1 = np.max(np.abs(p - (u - 1j * v))) / np.max(np.abs(u))
    check("p_is_u_minus_iv", e1 < 1e-12, "%.1e" % e1)
    # compact layout, as hybrid_step.to_compact
    row_idx = np.concatenate([np.arange(n - km, n), np.arange(0, km + 1)])
    thc = th[row_idx, :cc]
    kic = kinv[row_idx, :cc]
    U = np.concatenate([np.stack([thc.real, thc.imag], -1).reshape(-1), kic.reshape(-1)])
    ns = load_callbacks()
    # 2. the w load callback on every element
    wh = np.array([ns["prolog_w64"](None, o, U, None) for o in range(n * n)]).reshape(n, n)
    e2 = np.max(np.abs(wh - w)) / np.max(np.abs(w))
    check("prolog_w_equals_build_w", e2 < 1e-14, "%.1e" % e2)
    # 3. the C2R load callback
    qin = np.array([ns["prolog_q64"](None, o, U, None) for o in range(n * nk)]).reshape(n, nk)
    check("prolog_q_equals_theta", np.array_equal(qin, th))
    # 4. z callback, fft2, store callback, Hermitian split
    q_u = np.fft.irfft2(qin, s=(n, n)) * (n * n)                      # cuFFT's C2R is unnormalised
    pf, qf = p.reshape(-1), q_u.reshape(-1)
    Z = np.array([ns["prolog_z64"](pf, o, qf, None) for o in range(n * n)])
    Zh = np.fft.fft2(Z.reshape(n, n)).reshape(-1)
    Zc = np.zeros(cr * cr, dtype=complex)
    for o in range(n * n):
        ns["epilog_z64"](None, o, Zh[o], Zc, None)
    Zc = Zc.reshape(cr, cr)
    r = np.arange(cr)[:, None]
    c = np.arange(cc)[None, :]
    z = Zc[r, c + km]
    zm = Zc[2 * km - r, km - c]
    A = (z + np.conj(zm)) * 0.5
    Bm = (z - np.conj(zm)) / 2j
    q = np.fft.irfft2(th, s=(n, n))
    uq_ref = np.fft.rfft2(u * q)[row_idx, :cc]
    vq_ref = np.fft.rfft2(v * q)[row_idx, :cc]
    e4a = np.max(np.abs(A - uq_ref)) / np.max(np.abs(uq_ref))
    e4b = np.max(np.abs(Bm - vq_ref)) / np.max(np.abs(vq_ref))
    check("split_A_is_rfft2_u_theta", e4a < 1e-12, "%.1e" % e4a)
    check("split_B_is_rfft2_v_theta", e4b < 1e-12, "%.1e" % e4b)
    # the Warp kernel's component form of B and the tendency
    ar, ai = (z.real + zm.real) * 0.5, (z.imag - zm.imag) * 0.5
    br, bi = (z.imag + zm.imag) * 0.5, -((z.real - zm.real) * 0.5)
    check("component_form_A", np.allclose(ar + 1j * ai, A, rtol=0, atol=1e-12 * np.max(np.abs(A))))
    check("component_form_B", np.allclose(br + 1j * bi, Bm, rtol=0, atol=1e-12 * np.max(np.abs(Bm))))
    kxc = c.astype(float)
    kyc = (r - km).astype(float)
    tr = -((-(kxc * ai)) + (-(kyc * bi)))
    ti = -((kxc * ar) + (kyc * br))
    t_ref = -(1j * kx[None, :cc] * np.fft.rfft2(u * q)[row_idx, :cc]
              + 1j * ky[row_idx][:, None] * np.fft.rfft2(v * q)[row_idx, :cc])
    e5 = np.max(np.abs((tr + 1j * ti) - t_ref)) / np.max(np.abs(t_ref))
    check("tendency", e5 < 1e-12, "%.1e" % e5)
    print("HYBRID_MATH_TEST_OK")


if __name__ == "__main__":
    main()
