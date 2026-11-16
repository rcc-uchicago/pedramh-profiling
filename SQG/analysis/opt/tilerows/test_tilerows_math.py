"""test_tilerows_math.py -- the decomposition of tilerows_step.py on a small grid (numpy; Python 3.6 ok). Prints
TILEROWS_MATH_TEST_OK.

The Warp kernels' index logic is restated here per element (N = 48, KM = 16) and checked against direct FFTs:
  1. build_w on the kept columns (j -> jx: 0..KM, N-KM..N-1; mirror for k_x < 0; zeros where |k_y| > KM), forward
     column FFTs on the kept columns only, forward row FFTs  ==  fft2(build_w(psi_hat)) = u - i v;
  2. expand (C2R scratch) == theta_hat on the half plane;
  3. Z = conj(p) theta, forward row FFTs, forward column FFTs on the kept columns, Hermitian split on the compact
     block  ==  rfft2(u theta), rfft2(v theta) on the kept half block.
"""
import numpy as np

Nt, KMt = 48, 16


def main():
    rng = np.random.RandomState(1)
    n, nk, km = Nt, Nt // 2 + 1, KMt
    ck, cr, cc = 2 * km + 1, 2 * km + 1, km + 1
    s = 1.0 / (n * n)
    kx = np.arange(nk, dtype=float)
    ky = np.fft.fftfreq(n, 1.0 / n)
    kap = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    th = np.fft.rfft2(rng.standard_normal((n, n)))
    th[kap > km] = 0
    kinv = np.where(kap > 0, 1.0 / np.where(kap > 0, kap, 1.0), 0.0)
    row_idx = np.concatenate([np.arange(n - km, n), np.arange(0, km + 1)])
    hi, ki = th[row_idx, :cc], kinv[row_idx, :cc]
    # reference: u - i v and the two RFFTs
    ph = kinv * th
    u = np.fft.irfft2(-1j * ky[:, None] * ph, s=(n, n))
    v = np.fft.irfft2(1j * kx[None, :] * ph, s=(n, n))
    q = np.fft.irfft2(th, s=(n, n))
    # 1. build_w on kept columns (the kernel's logic, element by element)
    W = np.zeros((n, n), dtype=complex)
    for iy in range(n):
        kyv = iy if iy < n // 2 else iy - n
        for j in range(ck):
            jx = j if j <= km else j + (n - ck)
            if jx <= n // 2:
                kxv, kr = jx, kyv
            else:
                kxv, kr = n - jx, -kyv
            out = 0j
            if -km <= kr <= km:
                t = hi[kr + km, kxv]
                p = t * ki[kr + km, kxv]
                if jx <= n // 2:
                    out = (-s * jx + 1j * s * kyv) * np.conj(p)
                else:
                    out = (s * (n - jx) + 1j * s * kyv) * p
            W[iy, jx] = out
    kept_cols = np.r_[0:km + 1, n - km:n]
    W[:, kept_cols] = np.fft.fft(W[:, kept_cols], axis=0)            # column pass on kept columns only
    p_rows = np.fft.fft(W, axis=1)                                   # row pass
    e1 = np.max(np.abs(p_rows - (u - 1j * v))) / np.max(np.abs(u))
    assert e1 < 1e-12, e1
    print("ok p_is_u_minus_iv %.1e" % e1)
    # 2. expand
    TH = np.zeros((n, nk), dtype=complex)
    for iy in range(n):
        kyv = iy if iy < n // 2 else iy - n
        for c in range(nk):
            if c <= km and -km <= kyv <= km:
                TH[iy, c] = hi[kyv + km, c]
    assert np.array_equal(TH, th)
    q_u = np.fft.irfft2(TH, s=(n, n)) * (n * n)
    print("ok expand")
    # 3. Z rows, kept-column pass, split
    Z = np.conj(p_rows) * (q_u * s)
    Z = np.fft.fft(Z, axis=1)
    Z[:, kept_cols] = np.fft.fft(Z[:, kept_cols], axis=0)
    A = np.zeros((cr, cc), dtype=complex)
    Bm = np.zeros((cr, cc), dtype=complex)
    for r in range(cr):
        iy = r + (n - km) if r < km else r - km
        for c in range(cc):
            z = Z[iy, c]
            zm = Z[(n - iy) % n, (n - c) % n]
            A[r, c] = complex((z.real + zm.real) * 0.5, (z.imag - zm.imag) * 0.5)
            Bm[r, c] = complex((z.imag + zm.imag) * 0.5, -((z.real - zm.real) * 0.5))
    uq = np.fft.rfft2(u * q)[row_idx, :cc]
    vq = np.fft.rfft2(v * q)[row_idx, :cc]
    ea = np.max(np.abs(A - uq)) / np.max(np.abs(uq))
    eb = np.max(np.abs(Bm - vq)) / np.max(np.abs(vq))
    assert ea < 1e-12 and eb < 1e-12, (ea, eb)
    print("ok split %.1e %.1e" % (ea, eb))
    print("TILEROWS_MATH_TEST_OK")


if __name__ == "__main__":
    main()
