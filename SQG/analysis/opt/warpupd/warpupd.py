"""warpupd.py -- Warp SIMT kernels inside the JAX lean32ek_fast step, through warp/warp_xla.py's custom call.

The user's idea (2026-11-12): the hybrid's Warp update kernel took 0.21 ms where XLA's update fusion takes more; call
such kernels from the JAX step. Warp is used as SIMT code here (one thread per mode / grid point via wp.tid(), with
the step's irregular accesses written explicitly: the Hermitian mirror, the 2/3 mask, the compact gather), and every
FFT stays XLA's cuFFT call (no callbacks: they doubled the FFTs, job 7840175). Variants of the 40-step call
(AB3State -> AB3State, complex128 outside, lean32ek's entry / exit; `_if` = the IF-AB3 factors of if_ab3.py):
  xla_compact   pure XLA, state + history on the compact kept block (2731 x 1366): the control (research #3)
  warp_full     lean32ek_fast with its tendency + compensated update replaced by one Warp kernel, full half-plane
  warp_compact  as xla_compact, the update done by one Warp kernel reading the kept rows of XLA's rfft2 outputs
  warp_simt     XLA does only the FFTs: Warp builds w (mirrored gather from the compact state), Warp forms
                Z = conj(p) theta, XLA's fft2(Z) replaces the two rfft2s (research #4), and Warp does the Hermitian
                split + tendency + update on the compact block
The arithmetic of each variant is lean32ek's, component for component (Warp kernels compiled with fuse_fp=False).
Only the FFT decomposition (warp_simt) or a different fusion can change rounding; the probe gates every variant.
Imports jax / warp inside the builders; constants only at import.
"""
import numpy as np

N = 4096
NK = N // 2 + 1
KM = 1365
CR = 2 * KM + 1
CC = KM + 1
P = 8                                                                 # forcing patch (seed2_opt.PATCH)
ROW_IDX = np.concatenate([np.arange(N - KM, N), np.arange(0, KM + 1)])  # full-plane row of compact row r (k_y = r - KM)
VARIANTS = ("xla_compact", "warp_full", "warp_compact", "warp_simt")


# ---- Warp kernels (factories: the AB3 weights are compile-time constants) -------------------------------------------
def kernels(dt, if_mode):
    import warp as wp
    opts = {"fuse_fp": False, "enable_backward": False}
    C1 = np.float32((23 / 12) * dt)                                   # as JAX rounds the Python float for complex64
    C2 = np.float32((-16 / 12) * dt)
    C3 = np.float32((5 / 12) * dt)
    IFM = int(bool(if_mode))
    S = np.float32(1.0 / (N * N))
    vec6 = wp.types.vector(length=6, dtype=wp.float32)               # (hi', lo', T) of one mode

    @wp.func
    def update(a: wp.vec2f, b: wp.vec2f, kx: wp.float32, ky: wp.float32, fr: wp.float32, fi: wp.float32,
               t_n: wp.vec2f, t_o: wp.vec2f, h: wp.vec2f, l: wp.vec2f, e: wp.float32):
        """lean32ek: T = -(i kx A + i ky B) + F0; inc; f32_update.compensated_update. Returns vec6 (h', l', T)."""
        tr = -((-(kx * a[1])) + (-(ky * b[1]))) + fr
        ti = -((kx * a[0]) + (ky * b[0])) + fi
        if IFM == 0:
            ir = (C1 * tr + C2 * t_n[0]) + C3 * t_o[0]
            ii = (C1 * ti + C2 * t_n[1]) + C3 * t_o[1]
        else:
            e2 = e * (wp.float32(2.0) - e)
            ir = (C1 * tr + C2 * (t_n[0] - e * t_n[0])) + C3 * (t_o[0] - e2 * t_o[0])
            ii = (C1 * ti + C2 * (t_n[1] - e * t_n[1])) + C3 * (t_o[1] - e2 * t_o[1])
        yr = ir + l[0]
        a_r = h[0] + yr
        bp_r = a_r - h[0]
        r_r = (h[0] - (a_r - bp_r)) + (yr - bp_r)
        ea_r = e * a_r
        u_r = a_r - ea_r
        r2_r = (a_r - u_r) - ea_r
        yi = ii + l[1]
        a_i = h[1] + yi
        bp_i = a_i - h[1]
        r_i = (h[1] - (a_i - bp_i)) + (yi - bp_i)
        ea_i = e * a_i
        u_i = a_i - ea_i
        r2_i = (a_i - u_i) - ea_i
        return vec6(u_r, u_i, r2_r + (r_r - e * r_r), r2_i + (r_i - e * r_i), tr, ti)

    @wp.func
    def forcing(f0t: wp.array2d(dtype=wp.vec2f), f0b: wp.array2d(dtype=wp.vec2f), iy: int, c: int):
        f = wp.vec2f(0.0, 0.0)
        if c < P:
            if iy < P:
                f = f0t[iy, c]
            elif iy >= N - P:
                f = f0b[iy - (N - P), c]
        return f

    @wp.func
    def row_of(r: int):
        iy = r - KM
        if r < KM:
            iy = r + (N - KM)
        return iy

    @wp.kernel(module="unique", module_options=opts)
    def upd_compact(uqh: wp.array2d(dtype=wp.vec2f), vqh: wp.array2d(dtype=wp.vec2f),
                    tn: wp.array2d(dtype=wp.vec2f), to: wp.array2d(dtype=wp.vec2f),
                    hi: wp.array2d(dtype=wp.vec2f), lo: wp.array2d(dtype=wp.vec2f), edec: wp.array2d(dtype=wp.float32),
                    f0t: wp.array2d(dtype=wp.vec2f), f0b: wp.array2d(dtype=wp.vec2f),
                    hi_o: wp.array2d(dtype=wp.vec2f), lo_o: wp.array2d(dtype=wp.vec2f),
                    tn_o: wp.array2d(dtype=wp.vec2f)):
        r, c = wp.tid()
        iy = row_of(r)
        f = forcing(f0t, f0b, iy, c)
        o = update(uqh[iy, c], vqh[iy, c], wp.float32(c), wp.float32(r - KM), f[0], f[1],
                   tn[r, c], to[r, c], hi[r, c], lo[r, c], edec[r, c])
        hi_o[r, c] = wp.vec2f(o[0], o[1])
        lo_o[r, c] = wp.vec2f(o[2], o[3])
        tn_o[r, c] = wp.vec2f(o[4], o[5])

    @wp.kernel(module="unique", module_options=opts)
    def upd_full(uqh: wp.array2d(dtype=wp.vec2f), vqh: wp.array2d(dtype=wp.vec2f),
                 tn: wp.array2d(dtype=wp.vec2f), to: wp.array2d(dtype=wp.vec2f),
                 hi: wp.array2d(dtype=wp.vec2f), lo: wp.array2d(dtype=wp.vec2f), edec: wp.array2d(dtype=wp.float32),
                 f0t: wp.array2d(dtype=wp.vec2f), f0b: wp.array2d(dtype=wp.vec2f),
                 hi_o: wp.array2d(dtype=wp.vec2f), lo_o: wp.array2d(dtype=wp.vec2f), tn_o: wp.array2d(dtype=wp.vec2f)):
        iy, c = wp.tid()
        ky = iy
        if iy >= N // 2:
            ky = iy - N
        f = forcing(f0t, f0b, iy, c)
        o = update(uqh[iy, c], vqh[iy, c], wp.float32(c), wp.float32(ky), f[0], f[1],
                   tn[iy, c], to[iy, c], hi[iy, c], lo[iy, c], edec[iy, c])
        hi_o[iy, c] = wp.vec2f(o[0], o[1])
        lo_o[iy, c] = wp.vec2f(o[2], o[3])
        tn_o[iy, c] = wp.vec2f(o[4], o[5])

    @wp.kernel(module="unique", module_options=opts)
    def build_w(hi: wp.array2d(dtype=wp.vec2f), kinv: wp.array2d(dtype=wp.float32), w: wp.array2d(dtype=wp.vec2f)):
        """seed2_opt.build_w on the full plane from the compact state: one thread per output element; k_x < 0 reads
        the mirrored mode psi_hat(-k) (row -k_y, column N - j)."""
        iy, jx = wp.tid()
        ky = iy
        if iy >= N // 2:
            ky = iy - N
        kx = jx
        kr = ky
        if jx > N // 2:
            kx = N - jx
            kr = -ky
        out = wp.vec2f(0.0, 0.0)
        if kx <= KM and kr <= KM and kr >= -KM:
            t = hi[kr + KM, kx]
            ki = kinv[kr + KM, kx]
            p_r = t[0] * ki
            p_i = t[1] * ki
            mi = S * wp.float32(ky)
            if jx <= N // 2:
                mr = -(S * wp.float32(jx))
                out = wp.vec2f(mr * p_r - mi * (-p_i), mr * (-p_i) + mi * p_r)  # m * conj(ph), XLA's (ac - bd, ad + bc)
            else:
                mr = S * wp.float32(N - jx)
                out = wp.vec2f(mr * p_r - mi * p_i, mr * p_i + mi * p_r)        # m * ph(-k)
        w[iy, jx] = out

    @wp.kernel(module="unique", module_options=opts)
    def zprod(p: wp.array2d(dtype=wp.vec2f), q: wp.array2d(dtype=wp.float32), z: wp.array2d(dtype=wp.vec2f)):
        """Z = conj(p) theta = (real(p) theta, -imag(p) theta): lean32's two RFFT operands in one complex field."""
        iy, jx = wp.tid()
        a = p[iy, jx]
        b = q[iy, jx]
        z[iy, jx] = wp.vec2f(a[0] * b, (-a[1]) * b)

    @wp.kernel(module="unique", module_options=opts)
    def upd_split(zh: wp.array2d(dtype=wp.vec2f), tn: wp.array2d(dtype=wp.vec2f), to: wp.array2d(dtype=wp.vec2f),
                  hi: wp.array2d(dtype=wp.vec2f), lo: wp.array2d(dtype=wp.vec2f), edec: wp.array2d(dtype=wp.float32),
                  f0t: wp.array2d(dtype=wp.vec2f), f0b: wp.array2d(dtype=wp.vec2f),
                  hi_o: wp.array2d(dtype=wp.vec2f), lo_o: wp.array2d(dtype=wp.vec2f), tn_o: wp.array2d(dtype=wp.vec2f)):
        """Hermitian split of fft2(Z) on the compact block: A = (Z(k) + conj Z(-k)) / 2, B = (Z(k) - conj Z(-k)) / 2i,
        then the update. Z(-k) is the mirrored element ((N - iy) % N, (N - c) % N)."""
        r, c = wp.tid()
        iy = row_of(r)
        z = zh[iy, c]
        zm = zh[(N - iy) % N, (N - c) % N]
        half = wp.float32(0.5)
        a = wp.vec2f((z[0] + zm[0]) * half, (z[1] - zm[1]) * half)
        b = wp.vec2f((z[1] + zm[1]) * half, -((z[0] - zm[0]) * half))
        f = forcing(f0t, f0b, iy, c)
        o = update(a, b, wp.float32(c), wp.float32(r - KM), f[0], f[1], tn[r, c], to[r, c], hi[r, c], lo[r, c],
                   edec[r, c])
        hi_o[r, c] = wp.vec2f(o[0], o[1])
        lo_o[r, c] = wp.vec2f(o[2], o[3])
        tn_o[r, c] = wp.vec2f(o[4], o[5])

    return {"upd_compact": upd_compact, "upd_full": upd_full, "build_w": build_w, "zprod": zprod,
            "upd_split": upd_split}


# ---- the JAX calls ---------------------------------------------------------------------------------------------------
def make_call(pst, variant, dt, stride, *, if_mode=False):
    """jax.jit f(AB3State, tables) -> AB3State `stride` steps later. tables = lean32_precise.tables(GD) (+ compact
    copies from compact_tables). Entry / exit as lean32_fast.make_call (complex128 outside; hi = c64(theta), lo =
    c64(theta - hi)); the history between calls is kept only on the compact block for the compact variants (it never
    reaches the state outside it: the decay table carries the mask)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    import lean32_fast as LF
    import lean_solver as LS
    import warp_xla as WX
    S2, FU = LF.S2, LF.FU
    if variant not in VARIANTS:
        raise ValueError(variant)
    n, nk = N, NK
    s = 1.0 / (n * n)
    c1, c2, c3 = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt
    compact = variant != "warp_full"
    K = kernels(dt, if_mode) if variant != "xla_compact" else None
    c64 = jnp.complex64
    sc, sf, sz = (CR, CC), (n, nk), (n, n)

    if K is not None:
        upd_c = WX.warp_call(K["upd_compact"], [(sc, "complex64")] * 3, launch_dims=sc, block_dim=256)
        upd_f = WX.warp_call(K["upd_full"], [(sf, "complex64")] * 3, launch_dims=sf, block_dim=256)
        bw = WX.warp_call(K["build_w"], [(sz, "complex64")], launch_dims=sz, block_dim=256)
        zp = WX.warp_call(K["zprod"], [(sz, "complex64")], launch_dims=sz, block_dim=256)
        ups = WX.warp_call(K["upd_split"], [(sc, "complex64")] * 3, launch_dims=sc, block_dim=256)

    def kept(a):                                                      # full (N, NK) -> compact (CR, CC)
        return jnp.concatenate([a[n - KM:, :CC], a[:KM + 1, :CC]], 0)

    def expand(a):                                                    # compact -> full, zeros elsewhere
        full = jnp.concatenate([a[KM:], jnp.zeros((n - CR, CC), a.dtype), a[:KM]], 0)
        return jnp.pad(full, ((0, 0), (0, nk - CC)))

    def f0_compact(c):
        patch = jnp.concatenate([c["f0bot"], c["f0top"]], 0)          # rows k_y = -8 .. 7
        return jnp.pad(patch, ((KM - P, CR - KM - P), (0, CC - P)))

    def step(th, lo, t_n, t_o, c, t, tc):
        zero = t * jnp.float32(0)
        if variant == "warp_simt":
            w = bw(th, c["kinv_c"])[0]
            p = jnp.fft.fft2(w)
            q = jnp.fft.irfft2(expand(th), s=(n, n))
            zh = jnp.fft.fft2(zp(p, q)[0])
            hi1, lo1, t1 = ups(zh, t_n, t_o, th, lo, c["edec_c"], c["f0top"], c["f0bot"])
            return hi1, lo1, t1
        th_f = expand(th) if compact else th
        kinv = c["kinv"] * jnp.cos(jnp.real(th_f) * zero)             # lean32ek_fast's u1 trigger (== kinv)
        ph = kinv * th_f
        p = jnp.fft.fft2(S2.build_w(ph, c["kx"], c["ky"], s).astype(c64))
        q = jnp.fft.irfft2(th_f, s=(n, n))
        uqh = jnp.fft.rfft2(jnp.real(p) * q).astype(c64)
        vqh = jnp.fft.rfft2(-jnp.imag(p) * q).astype(c64)
        if variant == "warp_full":
            return tuple(upd_f(uqh, vqh, t_n, t_o, th, lo, c["edec"], c["f0top"], c["f0bot"]))
        if variant == "warp_compact":
            return tuple(upd_c(uqh, vqh, t_n, t_o, th, lo, c["edec_c"], c["f0top"], c["f0bot"]))
        # xla_compact: lean32_fast's tendency + update on the kept block
        uq, vq = kept(uqh), kept(vqh)
        kx, ky = c["kx"][:CC], c["ky_c"]
        ik = lax.complex(jnp.zeros_like(kx), kx)
        il = lax.complex(jnp.zeros_like(ky), ky)
        t_new = -(ik[None, :] * uq + il[:, None] * vq) + f0_compact(c)
        edec = c["edec_c"] * jnp.cos(jnp.imag(th) * zero)              # the u1 trigger on the compact fusion
        if if_mode:
            e2 = edec * (jnp.float32(2) - edec)
            inc = (c1 * t_new + c2 * (t_n - edec * t_n)) + c3 * (t_o - e2 * t_o)
        else:
            inc = (c1 * t_new + c2 * t_n) + c3 * t_o
        hi1, lo1 = FU.compensated_update(th, lo, inc, edec)
        return hi1, lo1, t_new

    def unpack(st):
        th64 = st.state.qh[0]
        hi = th64.astype(c64)
        lo = (th64 - hi.astype(th64.dtype)).astype(c64)
        tn_, to_ = st._updates[0].qh[0].astype(c64), st._updates[1].qh[0].astype(c64)
        if compact:
            return kept(hi), kept(lo), kept(tn_), kept(to_)
        return hi, lo, tn_, to_

    def pack(st, hi, lo, t_n, t_o, t, tc):
        cd = st.state.qh.dtype
        if compact:
            hi, lo, t_n, t_o = expand(hi), expand(lo), expand(t_n), expand(t_o)
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


def compact_tables(c32):
    """lean32_precise.tables + the compact copies the variants read: edec_c, kinv_c, ky_c."""
    import jax.numpy as jnp
    rows = jnp.asarray(ROW_IDX)
    out = dict(c32)
    out["edec_c"] = jnp.asarray(c32["edec"])[rows, :CC]
    kinv = jnp.asarray(c32["kinv"])
    kinv = kinv.reshape(kinv.shape[-2:]) if kinv.ndim == 3 else kinv
    out["kinv_c"] = jnp.broadcast_to(kinv, (N, NK))[rows, :CC]
    out["ky_c"] = jnp.asarray(c32["ky"]).reshape(-1)[rows]
    return out
