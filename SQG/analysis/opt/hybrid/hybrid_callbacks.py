"""hybrid_callbacks.py -- the cuFFT LTO load / store callbacks of hybrid_step.py, as numba device functions.

Compiled by nvmath.fft.compile_prolog / compile_epilog (numba-cuda -> LTO-IR); imported only where numba exists.
Module-level functions and constants (no closures): one set for float32 / complex64 (suffix 32), one for float64 /
complex128 (suffix 64). Formulas in hybrid_step.py's docstring; `offset` is the flat index into the operand's
natural row-major layout ((N, N) for the C2C plans, (N, NK) for the C2R input).
U = [theta_hat (re, im interleaved, compact) | kinv (compact)] as one float array; row = k_y + KM, col = k_x.
"""
import numba
import numpy as np

N = 4096
NK = N // 2 + 1
KM = 1365
CR = 2 * KM + 1
CC = KM + 1
B = CR * CC
B2 = 2 * B
HALFN = N // 2
S32 = np.float32(1.0 / (N * N))
S64 = np.float64(1.0 / (N * N))
ZERO32 = np.complex64(0)
ZERO64 = np.complex128(0)


# ---- float32 ----------------------------------------------------------------------------------------------------------
def prolog_w32(data_in, offset, U, unused):
    o = numba.int64(offset)
    iy = o // N
    jx = o - iy * N
    ky = iy if iy < HALFN else iy - N
    if jx <= HALFN:
        kx = jx
        kr = ky
    else:
        kx = N - jx
        kr = -ky
    if kx > KM or kr > KM or kr < -KM:
        return ZERO32
    idx = (kr + KM) * CC + kx
    ki = U[B2 + idx]
    ph = numba.complex64(complex(U[2 * idx] * ki, U[2 * idx + 1] * ki))
    if jx <= HALFN:
        m = numba.complex64(complex(-S32 * numba.float32(jx), S32 * numba.float32(ky)))
        return m * ph.conjugate()
    m = numba.complex64(complex(S32 * numba.float32(N - jx), S32 * numba.float32(ky)))
    return m * ph


def prolog_q32(data_in, offset, U, unused):
    o = numba.int64(offset)
    iy = o // NK
    jx = o - iy * NK
    ky = iy if iy < HALFN else iy - N
    if jx > KM or ky > KM or ky < -KM:
        return ZERO32
    idx = (ky + KM) * CC + jx
    return numba.complex64(complex(U[2 * idx], U[2 * idx + 1]))


def prolog_z32(data_in, offset, q, unused):
    p = data_in[offset]
    qq = q[offset] * S32
    return numba.complex64(complex(p.real * qq, -(p.imag * qq)))


def epilog_z32(data_out, offset, value, Zc, unused):
    o = numba.int64(offset)
    iy = o // N
    jx = o - iy * N
    ky = iy if iy < HALFN else iy - N
    kx = jx if jx < HALFN else jx - N
    if kx > KM or kx < -KM or ky > KM or ky < -KM:
        return
    Zc[(ky + KM) * CR + (kx + KM)] = value


# ---- float64 ----------------------------------------------------------------------------------------------------------
def prolog_w64(data_in, offset, U, unused):
    o = numba.int64(offset)
    iy = o // N
    jx = o - iy * N
    ky = iy if iy < HALFN else iy - N
    if jx <= HALFN:
        kx = jx
        kr = ky
    else:
        kx = N - jx
        kr = -ky
    if kx > KM or kr > KM or kr < -KM:
        return ZERO64
    idx = (kr + KM) * CC + kx
    ki = U[B2 + idx]
    ph = complex(U[2 * idx] * ki, U[2 * idx + 1] * ki)
    if jx <= HALFN:
        m = complex(-S64 * numba.float64(jx), S64 * numba.float64(ky))
        return m * ph.conjugate()
    m = complex(S64 * numba.float64(N - jx), S64 * numba.float64(ky))
    return m * ph


def prolog_q64(data_in, offset, U, unused):
    o = numba.int64(offset)
    iy = o // NK
    jx = o - iy * NK
    ky = iy if iy < HALFN else iy - N
    if jx > KM or ky > KM or ky < -KM:
        return ZERO64
    idx = (ky + KM) * CC + jx
    return complex(U[2 * idx], U[2 * idx + 1])


def prolog_z64(data_in, offset, q, unused):
    p = data_in[offset]
    qq = q[offset] * S64
    return complex(p.real * qq, -(p.imag * qq))


def epilog_z64(data_out, offset, value, Zc, unused):
    o = numba.int64(offset)
    iy = o // N
    jx = o - iy * N
    ky = iy if iy < HALFN else iy - N
    kx = jx if jx < HALFN else jx - N
    if kx > KM or kx < -KM or ky > KM or ky < -KM:
        return
    Zc[(ky + KM) * CR + (kx + KM)] = value


SETS = {
    "float32": {"w": (prolog_w32, "complex64", "float32"), "q": (prolog_q32, "complex64", "float32"),
                "z": (prolog_z32, "complex64", "float32"), "ze": (epilog_z32, "complex64", "complex64")},
    "float64": {"w": (prolog_w64, "complex128", "float64"), "q": (prolog_q64, "complex128", "float64"),
                "z": (prolog_z64, "complex128", "float64"), "ze": (epilog_z64, "complex128", "complex128")},
}
