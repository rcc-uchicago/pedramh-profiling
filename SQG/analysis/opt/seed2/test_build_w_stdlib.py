"""test_build_w_stdlib.py -- stdlib-only check of seed2_opt.build_w's index mapping, signs and scaling.

seed2_opt.build_w packs u and v into one forward complex FFT: fft2(W) = u - i v, with W = conj(u_hat + i v_hat) / N^2
built on the full plane from the rfft half plane. Its right half is the Hermitian mirror, row (-r) mod N and column
N - j. This test mirrors build_w's lax ops one for one in pure Python on an 8 x 8 grid. It compares the result with
the reference path, u = irfft2(-i k_y psi_hat) and v = irfft2(i k_x psi_hat), using C2R semantics (Hermitian
completion of the half plane). It also checks that a wrong sign or a mirror without the row flip is caught. No
numpy, no jax: runs anywhere (`python3 test_build_w_stdlib.py`). Prints SEED2_BUILDW_TEST_OK.
"""
import cmath
import random

N = 8
HALF = N // 2
NK = HALF + 1
KX = [float(i) for i in range(NK)]                                   # rfftfreq(N, 1/N)
KY = [float(i if i < HALF else i - N) for i in range(N)]             # fftfreq(N, 1/N)


def dft2(a, sign):
    """Full 2-D DFT of an N x N complex list, exponent sign*2*pi*i*(ky*y + kx*x)/N, no scaling."""
    out = [[0j] * N for _ in range(N)]
    for y in range(N):
        for x in range(N):
            acc = 0j
            for r in range(N):
                for c in range(N):
                    acc += a[r][c] * cmath.exp(sign * 2j * cmath.pi * (r * y + c * x) / N)
            out[y][x] = acc
    return out


def irfft2_c2r(h):
    """C2R semantics: complete the half plane Hermitian-wise, inverse DFT, real part, 1/N^2."""
    full = [[h[r][c] if c <= HALF else h[(-r) % N][N - c].conjugate() for c in range(N)] for r in range(N)]
    return [[v.real / N ** 2 for v in row] for row in dft2(full, +1)]


def build_w_py(ph, kx, ky, s, flip_rows=True, sign=1.0):
    """seed2_opt.build_w, lax op by lax op (rev, concatenate, slice_in_dim, rev)."""
    w_l = [[complex(-s * kx[c], s * ky[r]) * ph[r][c].conjugate() for c in range(HALF + 1)] for r in range(N)]
    rows = ph[::-1]                                                  # lax.rev(ph, (0,))
    rows = [rows[N - 1]] + rows[0:N - 1] if flip_rows else list(ph)  # row r <- (-r) mod N
    ph_r = [list(reversed(row[1:HALF])) for row in rows]             # columns HALF-1 .. 1
    kx_r = list(reversed(kx[1:HALF]))
    w_r = [[complex(sign * s * kx_r[c], s * ky[r]) * ph_r[r][c] for c in range(HALF - 1)] for r in range(N)]
    return [w_l[r] + w_r[r] for r in range(N)]


def main():
    rng = random.Random(7)
    theta = [[rng.gauss(0.0, 1.0) for _ in range(N)] for _ in range(N)]
    full = dft2([[complex(v) for v in row] for row in theta], -1)
    for r in range(N):                                               # zero the Nyquist row/column (2/3 damping does)
        for c in range(N):
            if r == HALF or c == HALF:
                full[r][c] = 0j
    th_hat = [[full[r][c] for c in range(NK)] for r in range(N)]     # rfft2 half plane
    ph = [[th_hat[r][c] / ((KX[c] ** 2 + KY[r] ** 2) ** 0.5) if (r or c) else 0j for c in range(NK)] for r in range(N)]

    u = irfft2_c2r([[-1j * KY[r] * ph[r][c] for c in range(NK)] for r in range(N)])
    v = irfft2_c2r([[1j * KX[c] * ph[r][c] for c in range(NK)] for r in range(N)])
    scale = max(max(abs(x) for x in row) for row in u + v)

    def err(w):
        p = dft2(w, -1)
        return max(max(abs(p[y][x].real - u[y][x]), abs(-p[y][x].imag - v[y][x])) for y in range(N) for x in range(N))

    s = 1.0 / N ** 2
    good = err(build_w_py(ph, KX, KY, s)) / scale
    bad_sign = err(build_w_py(ph, KX, KY, s, sign=-1.0)) / scale
    bad_rows = err(build_w_py(ph, KX, KY, s, flip_rows=False)) / scale
    ok =good < 1e-12 and bad_sign > 1e-3 and bad_rows > 1e-3
    print(f"build_w vs C2R path: max rel err {good:.2e} (needs < 1e-12); a wrong mirror sign gives {bad_sign:.2e}, "
          f"no row flip gives {bad_rows:.2e} (both must be > 1e-3)")
    print("SEED2_BUILDW_TEST_OK" if ok else "ERROR SEED2_BUILDW_TEST_FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
