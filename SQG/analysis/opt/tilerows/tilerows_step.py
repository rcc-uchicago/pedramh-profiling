"""tilerows_step.py -- the A-SQG step with Warp tile FFTs (cuFFTDx) for the row passes and cuFFT for the column passes.

User (2026-11-12), on Warp's tile FFTs: "blocks of GPU threads cooperate on shared-memory tiles and execute fast,
vendor-tuned FFTs without leaving the [kernel]". The Warp probe (job 7840069) found tile row FFTs near cuFFT (231 vs
206 us) and a fused 3-FFT row kernel 40% cheaper than separate passes, but tile COLUMN FFTs 2x slower than cuFFT. So:
rows in Warp tiles (fused with the pointwise work), columns in cuFFT (plain strided 1-D plans, no callbacks: callbacks
doubled the FFTs, job 7840175), and the irregular-access work in one-thread-per-element Warp kernels (wp.tid()).
Per step (float32, lean32ek's arithmetic; optional IF-AB3):
  1. build_w   (SIMT)  w(k) = s(-k_x + i k_y) conj(psi_hat) on the kept columns of W (N x N, FFT order), every row
                       (zeros where |k_y| > KMAX), from the compact state; k_x < 0 by the Hermitian mirror.
  2. cuFFT             forward column FFTs on the kept columns only (two strided batched plans: k_x = 0..KM and
                       -KM..-1), in place: W(y, k_x).
  3. expand    (SIMT)  the compact theta_hat into a full (N, NK) scratch (cuFFT's C2R overwrites its input).
  4. cuFFT             C2R 2-D: q_u = N^2 theta.
  5. rows      (tile)  one block per row y: tile_fft(W row) = p = u - i v; Z = conj(p) q_u s; tile_fft(Z); store the row.
  6. cuFFT             forward column FFTs on the kept columns of Z: Z_hat.
  7. update    (SIMT)  Hermitian split A = (Z(k) + conj Z(-k))/2, B = (Z(k) - conj Z(-k))/2i on the compact block,
                       tendency, AB3 (or IF-AB3) increment, lean32ek's compensated update (fuse_fp=False).
The state and history live on the compact block (rows k_y + KM, cols 0..KM); non-kept columns of W / Z are never
written (zero from allocation, and skipped by the column plans).
"""
import numpy as np

N = 4096
NK = N // 2 + 1
KM = 1365
CR = 2 * KM + 1
CC = KM + 1
CK = 2 * KM + 1                                                        # kept columns of a full row
P = 8
ROW_IDX = np.concatenate([np.arange(N - KM, N), np.arange(0, KM + 1)])
FWD = -1


def to_compact(a):
    return np.ascontiguousarray(np.asarray(a)[ROW_IDX, :CC])


def from_compact(c, dtype):
    out = np.zeros((N, NK), dtype=dtype)
    out[ROW_IDX, :CC] = np.asarray(c)
    return out


def kernels(dt, if_mode, block_dim):
    import warp as wp
    opts = {"fuse_fp": False, "enable_backward": False}
    C1, C2, C3 = (np.float32((23 / 12) * dt), np.float32((-16 / 12) * dt), np.float32((5 / 12) * dt))
    IFM = int(bool(if_mode))
    S = np.float32(1.0 / (N * N))
    NN = N

    @wp.kernel(module="unique", module_options=opts)
    def build_w(hi: wp.array2d(dtype=wp.vec2f), kinv: wp.array2d(dtype=wp.float32), W: wp.array2d(dtype=wp.vec2f)):
        iy, j = wp.tid()                                             # j over the CK kept columns
        jx = j
        if j > KM:
            jx = j + (N - CK)
        ky = iy
        if iy >= N // 2:
            ky = iy - N
        kx = jx
        kr = ky
        if jx > N // 2:
            kx = N - jx
            kr = -ky
        out = wp.vec2f(0.0, 0.0)
        if kr <= KM and kr >= -KM:
            t = hi[kr + KM, kx]
            ki = kinv[kr + KM, kx]
            p_r = t[0] * ki
            p_i = t[1] * ki
            mi = S * wp.float32(ky)
            if jx <= N // 2:
                mr = -(S * wp.float32(jx))
                out = wp.vec2f(mr * p_r - mi * (-p_i), mr * (-p_i) + mi * p_r)
            else:
                mr = S * wp.float32(N - jx)
                out = wp.vec2f(mr * p_r - mi * p_i, mr * p_i + mi * p_r)
        W[iy, jx] = out

    @wp.kernel(module="unique", module_options=opts)
    def expand(hi: wp.array2d(dtype=wp.vec2f), th: wp.array2d(dtype=wp.vec2f)):
        iy, c = wp.tid()
        ky = iy
        if iy >= N // 2:
            ky = iy - N
        v = wp.vec2f(0.0, 0.0)
        if c <= KM and ky <= KM and ky >= -KM:
            v = hi[ky + KM, c]
        th[iy, c] = v

    @wp.func
    def zmul(a: wp.vec2f, q: wp.float32):
        qq = q * S                                                   # theta = q_u / N^2, exact
        r = a * wp.float32(0.0)
        r[0] = a[0] * qq
        r[1] = (-a[1]) * qq                                          # conj(p) theta (lean32: real(p) q, -imag(p) q)
        return r

    @wp.kernel(module="unique", module_options=opts)
    def rows(W: wp.array2d(dtype=wp.vec2f), q: wp.array2d(dtype=wp.float32), Z: wp.array2d(dtype=wp.vec2f)):
        y, _l = wp.tid()
        t = wp.tile_load(W, shape=(1, NN), offset=(y, 0))
        wp.tile_fft(t)                                               # p row = (u - i v)(y, :)
        qr = wp.tile_load(q, shape=(1, NN), offset=(y, 0))
        z = wp.tile_map(zmul, t, qr)
        wp.tile_fft(z)
        wp.tile_store(Z, z, offset=(y, 0))

    @wp.kernel(module="unique", module_options=opts)
    def update(Zh: wp.array2d(dtype=wp.vec2f), hi: wp.array2d(dtype=wp.vec2f), lo: wp.array2d(dtype=wp.vec2f),
               tn: wp.array2d(dtype=wp.vec2f), to: wp.array2d(dtype=wp.vec2f), edec: wp.array2d(dtype=wp.float32),
               f0: wp.array2d(dtype=wp.vec2f), tnew: wp.array2d(dtype=wp.vec2f)):
        r, c = wp.tid()
        iy = r - KM
        if r < KM:
            iy = r + (N - KM)
        z = Zh[iy, c]
        zm = Zh[(N - iy) % N, (N - c) % N]
        half = wp.float32(0.5)
        ar = (z[0] + zm[0]) * half
        ai = (z[1] - zm[1]) * half
        br = (z[1] + zm[1]) * half
        bi = -((z[0] - zm[0]) * half)
        kx = wp.float32(c)
        ky = wp.float32(r - KM)
        fr = wp.float32(0.0)
        fi = wp.float32(0.0)
        pr = r - (KM - P)
        if pr >= 0 and pr < 2 * P and c < P:
            f = f0[pr, c]
            fr = f[0]
            fi = f[1]
        tr = -((-(kx * ai)) + (-(ky * bi))) + fr
        ti = -((kx * ar) + (ky * br)) + fi
        t_n = tn[r, c]
        t_o = to[r, c]
        e = edec[r, c]
        if IFM == 0:
            ir = (C1 * tr + C2 * t_n[0]) + C3 * t_o[0]
            ii = (C1 * ti + C2 * t_n[1]) + C3 * t_o[1]
        else:
            e2 = e * (wp.float32(2.0) - e)
            ir = (C1 * tr + C2 * (t_n[0] - e * t_n[0])) + C3 * (t_o[0] - e2 * t_o[0])
            ii = (C1 * ti + C2 * (t_n[1] - e * t_n[1])) + C3 * (t_o[1] - e2 * t_o[1])
        h = hi[r, c]
        l = lo[r, c]
        yr = ir + l[0]
        a_r = h[0] + yr
        bp_r = a_r - h[0]
        rr = (h[0] - (a_r - bp_r)) + (yr - bp_r)
        ea_r = e * a_r
        u_r = a_r - ea_r
        r2_r = (a_r - u_r) - ea_r
        yi = ii + l[1]
        a_i = h[1] + yi
        bp_i = a_i - h[1]
        ri = (h[1] - (a_i - bp_i)) + (yi - bp_i)
        ea_i = e * a_i
        u_i = a_i - ea_i
        r2_i = (a_i - u_i) - ea_i
        hi[r, c] = wp.vec2f(u_r, u_i)
        lo[r, c] = wp.vec2f(r2_r + (rr - e * rr), r2_i + (ri - e * ri))
        tnew[r, c] = wp.vec2f(tr, ti)

    return {"build_w": build_w, "expand": expand, "rows": rows, "update": update}


class TileRows:
    """The float32 step (lean32ek numerics) on torch buffers. tables: kinv, edec (compact float32), f0patch
    (2P x P complex64, rows k_y = -8..7)."""

    def __init__(self, tables, dt, *, if_mode=False, block_dim=256, stream=None):
        import torch
        import warp as wp
        from nvmath.bindings import cufft as cb
        self.torch, self.wp, self.cb = torch, wp, cb
        dev = "cuda"
        self.B = block_dim
        self.K = kernels(dt, if_mode, block_dim)
        self.hi = torch.zeros((CR, CC), dtype=torch.complex64, device=dev)
        self.lo = torch.zeros((CR, CC), dtype=torch.complex64, device=dev)
        self.T = [torch.zeros((CR, CC), dtype=torch.complex64, device=dev) for _ in range(3)]
        self.slot = [0, 1, 2]
        self.kinv = torch.as_tensor(np.asarray(tables["kinv"]), dtype=torch.float32, device=dev)
        self.edec = torch.as_tensor(np.asarray(tables["edec"]), dtype=torch.float32, device=dev)
        self.f0 = torch.as_tensor(np.asarray(tables["f0patch"]), dtype=torch.complex64, device=dev)
        self.W = torch.zeros((N, N), dtype=torch.complex64, device=dev)
        self.Z = torch.zeros((N, N), dtype=torch.complex64, device=dev)
        self.th = torch.zeros((N, NK), dtype=torch.complex64, device=dev)
        self.q = torch.zeros((N, N), dtype=torch.float32, device=dev)
        self.stream = stream if stream is not None else torch.cuda.Stream()
        torch.cuda.synchronize()
        self.ws = wp.stream_from_torch(self.stream)
        v = (lambda t: wp.from_torch(torch.view_as_real(t), dtype=wp.vec2f))   # noqa: E731
        self.w_hi, self.w_lo, self.w_T = v(self.hi), v(self.lo), [v(t) for t in self.T]
        self.w_W, self.w_Z, self.w_th, self.w_f0 = v(self.W), v(self.Z), v(self.th), v(self.f0)
        self.w_q, self.w_kinv, self.w_edec = wp.from_torch(self.q), wp.from_torch(self.kinv), wp.from_torch(self.edec)
        # cuFFT plans: kept-column FFTs (strided batched 1-D, in place) and the 2-D C2R
        s = self.stream.cuda_stream
        self.col_a = cb.create()                                     # k_x = 0 .. KM        (KM + 1 columns)
        cb.make_plan_many(self.col_a, 1, [N], [N], N, 1, [N], N, 1, cb.Type.C2C, KM + 1)
        self.col_b = cb.create()                                     # k_x = -KM .. -1      (KM columns)
        cb.make_plan_many(self.col_b, 1, [N], [N], N, 1, [N], N, 1, cb.Type.C2C, KM)
        self.c2r = cb.create()
        cb.make_plan2d(self.c2r, N, N, cb.Type.C2R)
        for h in (self.col_a, self.col_b, self.c2r):
            cb.set_stream(h, s)
        self.off_b = (N - KM) * 8                                    # bytes to column N - KM
        self.ptr = {"W": self.W.data_ptr(), "Z": self.Z.data_ptr(), "th": self.th.data_ptr(), "q": self.q.data_ptr()}

    def pack(self, theta128, t_n, t_o):
        torch = self.torch
        th = np.asarray(theta128)
        outside = th.copy()
        outside[ROW_IDX, :CC] = 0
        if np.any(outside != 0):
            raise ValueError("state is not zero outside the compact kept block")
        thc = to_compact(th)
        hi = thc.astype(np.complex64)
        self.hi.copy_(torch.as_tensor(hi))
        self.lo.copy_(torch.as_tensor((thc - hi.astype(np.complex128)).astype(np.complex64)))
        self.slot = [0, 1, 2]
        self.T[0].copy_(torch.as_tensor(to_compact(t_n).astype(np.complex64)))
        self.T[1].copy_(torch.as_tensor(to_compact(t_o).astype(np.complex64)))
        torch.cuda.synchronize()

    def theta(self):
        self.stream.synchronize()
        th = self.hi.cpu().numpy().astype(np.complex128) + self.lo.cpu().numpy().astype(np.complex128)
        return from_compact(th, np.complex128)

    def ops(self):
        """The step as a list of (name, callable): the per-op split times these."""
        wp, cb, K, p, ws, B = self.wp, self.cb, self.K, self.ptr, self.ws, self.B
        return [
            ("build_w", lambda: wp.launch(K["build_w"], dim=(N, CK), inputs=[self.w_hi, self.w_kinv, self.w_W],
                                          stream=ws)),
            ("cufft_cols_w", lambda: (cb.exec_c2c(self.col_a, p["W"], p["W"], FWD),
                                      cb.exec_c2c(self.col_b, p["W"] + self.off_b, p["W"] + self.off_b, FWD))),
            ("expand", lambda: wp.launch(K["expand"], dim=(N, NK), inputs=[self.w_hi, self.w_th], stream=ws)),
            ("cufft_c2r", lambda: cb.exec_c2r(self.c2r, p["th"], p["q"])),
            ("tile_rows", lambda: wp.launch_tiled(K["rows"], dim=[N], inputs=[self.w_W, self.w_q, self.w_Z],
                                                  block_dim=B, stream=ws)),
            ("cufft_cols_z", lambda: (cb.exec_c2c(self.col_a, p["Z"], p["Z"], FWD),
                                      cb.exec_c2c(self.col_b, p["Z"] + self.off_b, p["Z"] + self.off_b, FWD))),
            ("update", self._update),
        ]

    def _update(self):
        a, b, f = self.slot
        self.wp.launch(self.K["update"], dim=(CR, CC), inputs=[self.w_Z, self.w_hi, self.w_lo, self.w_T[a],
                                                                self.w_T[b], self.w_edec, self.w_f0, self.w_T[f]],
                       stream=self.ws)
        self.slot = [f, a, b]

    def step(self):
        for _, op in self.ops():
            op()

    def run(self, n):
        ops = [op for _, op in self.ops()]
        for _ in range(n):
            for op in ops:
                op()
