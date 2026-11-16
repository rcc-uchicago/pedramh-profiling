"""test_inrow_math.py -- inrow_step.py's theta path and re-indexing on a small grid (numpy; Python 3.6 ok). Prints
INROW_MATH_TEST_OK.

  1. FFT row order: hi_fft[ROW_IDX] = compact, and the kernels' index maps (build_w: k_y -> k_y mod N; update: compact
     row r -> FFT row) agree with ROW_IDX.
  2. theta: the inverse column pass on the k_x = 0..KM columns of hi_fft (out of place), TH[:, 0] /= 2, a one-sided
     row inverse zero-padded to N, q_u = 2 Re  ==  N^2 irfft2(theta_hat)  (tilerows' expand + C2R), to 1e-12.
  3. the same with a non-Hermitian k_x = 0 column (Im A_0 != 0 on rows): still equal to numpy's irfft2, which drops
     Im of the DC bin, as the formula does.
"""
import numpy as np

Nt, KMt = 48, 16


def theta_inrow(hi_fft, n, cc, ccp):
    th = np.zeros((n, ccp), dtype=complex)
    th[:, :cc] = np.fft.ifft(hi_fft, axis=0) * n                     # cuFFT C2C INVERSE, unnormalised
    th[:, 0] *= 0.5                                                  # halve0
    g = np.zeros((n, n), dtype=complex)
    g[:, :ccp] = th                                                  # bounds-checked (1, N) tile load
    return 2.0 * (np.fft.ifft(g, axis=1) * n).real                   # tile_ifft (unnormalised), re2


def main():
    rng = np.random.RandomState(2)
    n, nk, km = Nt, Nt // 2 + 1, KMt
    cr, cc = 2 * km + 1, km + 1
    ccp = (cc + 7) // 8 * 8                                          # a padded pitch, as CCP (never written)
    row_idx = np.concatenate([np.arange(n - km, n), np.arange(0, km + 1)])
    kx = np.arange(nk, dtype=float)
    ky = np.fft.fftfreq(n, 1.0 / n)
    kap = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    th = np.fft.rfft2(rng.standard_normal((n, n)))
    th[kap > km] = 0
    th[np.abs(ky) > km, :] = 0
    th[:, cc:] = 0
    compact = th[row_idx, :cc]
    # 1. layout and index maps
    hi_fft = np.zeros((n, cc), dtype=complex)
    hi_fft[row_idx] = compact
    assert np.array_equal(hi_fft, th[:, :cc])                        # FFT row order == the half plane's own rows
    for kr in range(-km, km + 1):
        fr = kr if kr >= 0 else kr + n                               # build_w
        assert fr == row_idx[kr + km]
    for r in range(cr):
        iy = r + (n - km) if r < km else r - km                      # update
        assert iy == row_idx[r]
    print("ok layout")
    # 2. theta
    q_ref = np.fft.irfft2(th, s=(n, n)) * (n * n)
    q = theta_inrow(hi_fft, n, cc, ccp)
    e2 = np.max(np.abs(q - q_ref)) / np.max(np.abs(q_ref))
    assert e2 < 1e-12, e2
    print("ok theta %.1e" % e2)
    # 3. a non-Hermitian k_x = 0 column
    th3 = th.copy()
    pert = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.3
    pert[np.abs(ky) > km] = 0
    th3[:, 0] += pert
    hi3 = th3[:, :cc].copy()
    q3_ref = np.fft.irfft2(th3, s=(n, n)) * (n * n)
    q3 = theta_inrow(hi3, n, cc, ccp)
    e3 = np.max(np.abs(q3 - q3_ref)) / np.max(np.abs(q3_ref))
    assert e3 < 1e-12, e3
    print("ok non_hermitian_dc %.1e" % e3)
    print("INROW_MATH_TEST_OK")


if __name__ == "__main__":
    main()
