"""warp_kernels.py -- the Warp tile kernels of warp_fft_probe.py (import after warp 1.17.0 is on sys.path).

Factories return a fresh kernel per (dtype, tile shape) in its own module ("unique": content-hashed, so two
factories with identical code would share one; the scalar-fallback row kernel therefore has its own name).
All transforms are along a tile's last axis and unnormalised (Warp's convention); N is the solver's 4096.
"""
from typing import Any

import warp as wp

N = 4096


@wp.func
def ident(a: Any):
    return a


@wp.func
def conj_mul_f(p: wp.vec2f, q: wp.vec2f):
    """conj(p) * q."""
    return wp.vec2f(p[0] * q[0] + p[1] * q[1], p[0] * q[1] - p[1] * q[0])


@wp.func
def conj_mul_d(p: wp.vec2d, q: wp.vec2d):
    """conj(p) * q."""
    return wp.vec2d(p[0] * q[0] + p[1] * q[1], p[0] * q[1] - p[1] * q[0])


def row(T, RR, inverse=False):
    """(RR, N) row tiles; launch dims (N // RR,)."""
    NN = N
    if inverse:
        @wp.kernel(module="unique")
        def row_ifft(x: wp.array2d(dtype=T), y: wp.array2d(dtype=T)):
            i, _l = wp.tid()
            t = wp.tile_load(x, shape=(RR, NN), offset=(i * RR, 0))
            wp.tile_ifft(t)
            wp.tile_store(y, t, offset=(i * RR, 0))
        return row_ifft

    @wp.kernel(module="unique")
    def row_fft(x: wp.array2d(dtype=T), y: wp.array2d(dtype=T)):
        i, _l = wp.tid()
        t = wp.tile_load(x, shape=(RR, NN), offset=(i * RR, 0))
        wp.tile_fft(t)
        wp.tile_store(y, t, offset=(i * RR, 0))
    return row_fft


def row_fallback(T, RR):
    """row() under another name: compiled while wp.config.enable_mathdx_fft is False (the scalar path)."""
    NN = N

    @wp.kernel(module="unique")
    def row_fft_scalar_fallback(x: wp.array2d(dtype=T), y: wp.array2d(dtype=T)):
        i, _l = wp.tid()
        t = wp.tile_load(x, shape=(RR, NN), offset=(i * RR, 0))
        wp.tile_fft(t)
        wp.tile_store(y, t, offset=(i * RR, 0))
    return row_fft_scalar_fallback


def col(T, K, mode):
    """Column FFT, K columns per block; launch dims (N // K,).
    strided: (N, K) shared strip -> transpose view -> register copy -> FFT -> transpose back -> same layout.
    tstore:  as strided, but stores the (K, N) result as rows of y (y = the transform, transposed).
    tview:   x and y are transposed (strided) array views; the (K, N) register tile is loaded through the view."""
    NN = N
    if mode == "strided":
        @wp.kernel(module="unique")
        def col_strided(x: wp.array2d(dtype=T), y: wp.array2d(dtype=T)):
            j, _l = wp.tid()
            s = wp.tile_load(x, shape=(NN, K), offset=(0, j * K), storage="shared")
            r = wp.tile_map(ident, wp.tile_transpose(s))
            wp.tile_fft(r)
            wp.tile_store(y, wp.tile_transpose(r), offset=(0, j * K))
        return col_strided
    if mode == "tstore":
        @wp.kernel(module="unique")
        def col_tstore(x: wp.array2d(dtype=T), y: wp.array2d(dtype=T)):
            j, _l = wp.tid()
            s = wp.tile_load(x, shape=(NN, K), offset=(0, j * K), storage="shared")
            r = wp.tile_map(ident, wp.tile_transpose(s))
            wp.tile_fft(r)
            wp.tile_store(y, r, offset=(j * K, 0))
        return col_tstore
    if mode == "tview":
        @wp.kernel(module="unique")
        def col_tview(x: wp.array2d(dtype=T), y: wp.array2d(dtype=T)):
            j, _l = wp.tid()
            r = wp.tile_load(x, shape=(K, NN), offset=(j * K, 0))
            wp.tile_fft(r)
            wp.tile_store(y, r, offset=(j * K, 0))
        return col_tview
    raise ValueError(mode)


def transpose(T, TT):
    """NVIDIA's example_fft_poisson_navier_stokes_2d.py transpose: (TT, TT) shared tiles; launch dims
    (N // TT, N // TT)."""
    @wp.kernel(module="unique")
    def tiled_transpose(x: wp.array2d(dtype=T), y: wp.array2d(dtype=T)):
        i, j, _l = wp.tid()
        t = wp.tile_load(x, shape=(TT, TT), offset=(i * TT, j * TT), storage="shared")
        wp.tile_store(y, wp.tile_transpose(t), offset=(j * TT, i * TT))
    return tiled_transpose


def fused3(T):
    """The step's row-kernel proxy: z_row = fft(conj(ifft(a_row)) * ifft(b_row)), 3 FFTs per row; launch (N,)."""
    NN = N
    cm = conj_mul_f if T == wp.vec2f else conj_mul_d

    @wp.kernel(module="unique")
    def fused_rows(a: wp.array2d(dtype=T), b: wp.array2d(dtype=T), z: wp.array2d(dtype=T)):
        i, _l = wp.tid()
        ta = wp.tile_load(a, shape=(1, NN), offset=(i, 0))
        tb = wp.tile_load(b, shape=(1, NN), offset=(i, 0))
        wp.tile_ifft(ta)
        wp.tile_ifft(tb)
        tz = wp.tile_map(cm, ta, tb)
        wp.tile_fft(tz)
        wp.tile_store(z, tz, offset=(i, 0))
    return fused_rows


@wp.kernel(module="unique")
def half_k(x: wp.array(dtype=wp.vec2f), y: wp.array(dtype=wp.vec2f)):
    """The bridge-overhead kernel: y = x / 2 on a tiny array."""
    i = wp.tid()
    y[i] = x[i] * 0.5


@wp.kernel(module="unique")
def double_k(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    """The environment check: y = 2 x."""
    i = wp.tid()
    y[i] = 2.0 * x[i]
