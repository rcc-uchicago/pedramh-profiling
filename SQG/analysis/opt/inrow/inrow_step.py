"""inrow_step.py -- the tile-row step with theta's inverse real FFT folded into the row kernel ("in-tile C2R",
candidate 1 of warp/WARP_BUILTINS_AUDIT_2026-11-14.md). Predictions + gates: PREDICTIONS_inrow.md (written first).

tilerows_step.TileRows makes q_u = N^2 theta with expand (SIMT) + a cuFFT 2-D C2R (0.079 + 0.302 ms of 1.578 in job
7840376), and its row kernel reads q_u back from memory. Here:
  - hi is stored in FFT row order, (N, CC): row k_y mod N, column k_x = 0..KM; the rows between +-KM are zero and
    never written. theta's inverse column pass is then one out-of-place cuFFT C2C (batch CC) straight from the state
    into TH (N, CCP). No expand, and no 2-D C2R.
  - halve0 (SIMT, N threads): TH[:, 0] *= 1/2 (exact).
  - rows_ir (tile kernel, one block per row y): G = the TH row, loaded bounds-checked into a (1, N) tile (k_x > KM read
    as zero), tile_ifft(G), q_u = 2 Re G (exact); then as tilerows: tile_fft(W row) = p, Z = conj(p) q_u S, tile_fft(Z).
Identity: with G_0 = A_0 / 2 and G_k = A_k (1 <= k <= KM), 2 Re sum_k G_k e^{2 pi i k x / N} = Re A_0 + 2 Re sum_{k>=1}
A_k e^{2 pi i k x / N}, the Hermitian (C2R) inverse with Im A_0 dropped, as numpy's irfft does (test_inrow_math.py).
The row kernel computes theta first and keeps only the real tile live during W's FFT (the warp probe's fused 3-FFT row
kernel, with two complex tiles live, ran at 0.549 ms vs 0.302 for the 2-FFT kernel: job 7840069).
build_w, the W / Z column passes and update are tilerows_step's arithmetic, re-indexed for the FFT-order hi.
"""
import numpy as np

import tilerows_step as TS

N, NK, KM, CR, CC, CK, P = TS.N, TS.NK, TS.KM, TS.CR, TS.CC, TS.CK, TS.P
ROW_IDX = TS.ROW_IDX                                                   # compact row r -> FFT row (k_y mod N)
CCP = (CC + 31) // 32 * 32                                             # TH row pitch: 1376 = 11,008 B, 128-B aligned
FWD, INV = -1, 1
ORDERS = ("theta_first", "w_first")


def to_fft_rows(compact, dtype):
    """(CR, CC) compact block -> (N, CC) FFT row order, zero rows between +-KM."""
    out = np.zeros((N, CC), dtype=dtype)
    out[ROW_IDX] = np.asarray(compact)
    return out


def kernels(dt, if_mode, order):
    import warp as wp
    if order not in ORDERS:
        raise ValueError(f"order {order!r} not in {ORDERS}")
    opts = {"fuse_fp": False, "enable_backward": False}
    # float32 values held as Python floats (exact): Warp rejects numpy scalars as compile-time constants (job 7840360)
    C1, C2, C3 = (float(np.float32((23 / 12) * dt)), float(np.float32((-16 / 12) * dt)),
                  float(np.float32((5 / 12) * dt)))
    IFM = int(bool(if_mode))
    S = float(np.float32(1.0 / (N * N)))
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
            fr = kr                                                  # FFT row of k_y = kr
            if kr < 0:
                fr = kr + N
            t = hi[fr, kx]
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
    def halve0(TH: wp.array2d(dtype=wp.vec2f)):
        i = wp.tid()
        TH[i, 0] = TH[i, 0] * wp.float32(0.5)

    @wp.func
    def re2(g: wp.vec2f):
        return g[0] + g[0]                                           # q_u = 2 Re G (exact)

    @wp.func
    def zmul(a: wp.vec2f, q: wp.float32):                            # tilerows_step.zmul
        qq = q * S                                                   # theta = q_u / N^2, exact
        r = a * wp.float32(0.0)
        r[0] = a[0] * qq
        r[1] = (-a[1]) * qq                                          # conj(p) theta
        return r

    if order == "theta_first":
        @wp.kernel(module="unique", module_options=opts)
        def rows_ir(W: wp.array2d(dtype=wp.vec2f), TH: wp.array2d(dtype=wp.vec2f), Z: wp.array2d(dtype=wp.vec2f)):
            y, _l = wp.tid()
            g = wp.tile_load(TH, shape=(1, NN), offset=(y, 0))       # k_x = 0..KM; columns >= CCP read as zero
            wp.tile_ifft(g)
            qr = wp.tile_map(re2, g)                                 # only the real tile stays live
            t = wp.tile_load(W, shape=(1, NN), offset=(y, 0))
            wp.tile_fft(t)                                           # p row = (u - i v)(y, :)
            z = wp.tile_map(zmul, t, qr)
            wp.tile_fft(z)
            wp.tile_store(Z, z, offset=(y, 0))
    else:
        @wp.kernel(module="unique", module_options=opts)
        def rows_ir(W: wp.array2d(dtype=wp.vec2f), TH: wp.array2d(dtype=wp.vec2f), Z: wp.array2d(dtype=wp.vec2f)):
            y, _l = wp.tid()
            t = wp.tile_load(W, shape=(1, NN), offset=(y, 0))
            wp.tile_fft(t)
            g = wp.tile_load(TH, shape=(1, NN), offset=(y, 0))
            wp.tile_ifft(g)
            qr = wp.tile_map(re2, g)
            z = wp.tile_map(zmul, t, qr)
            wp.tile_fft(z)
            wp.tile_store(Z, z, offset=(y, 0))

    @wp.kernel(module="unique", module_options=opts)
    def theta_rows(TH: wp.array2d(dtype=wp.vec2f), q: wp.array2d(dtype=wp.float32)):
        """rows_ir's theta half alone, stored (the probe's q_u check)."""
        y, _l = wp.tid()
        g = wp.tile_load(TH, shape=(1, NN), offset=(y, 0))
        wp.tile_ifft(g)
        qr = wp.tile_map(re2, g)
        wp.tile_store(q, qr, offset=(y, 0))

    @wp.kernel(module="unique", module_options=opts)
    def update(Zh: wp.array2d(dtype=wp.vec2f), hi: wp.array2d(dtype=wp.vec2f), lo: wp.array2d(dtype=wp.vec2f),
               tn: wp.array2d(dtype=wp.vec2f), to: wp.array2d(dtype=wp.vec2f), edec: wp.array2d(dtype=wp.float32),
               f0: wp.array2d(dtype=wp.vec2f), tnew: wp.array2d(dtype=wp.vec2f)):
        r, c = wp.tid()
        iy = r - KM                                                  # FFT row of compact row r
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
        h = hi[iy, c]
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
        hi[iy, c] = wp.vec2f(u_r, u_i)
        lo[r, c] = wp.vec2f(r2_r + (rr - e * rr), r2_i + (ri - e * ri))
        tnew[r, c] = wp.vec2f(tr, ti)

    return {"build_w": build_w, "halve0": halve0, "rows_ir": rows_ir, "theta_rows": theta_rows, "update": update}


class InRow:
    """tilerows_step.TileRows with the in-tile C2R. Same interface (pack, theta, ops, step, run); self.hi is the
    (N, CC) FFT-row-order buffer (compact_hi() gives the (CR, CC) block)."""

    def __init__(self, tables, dt, *, if_mode=False, block_dim=256, order="theta_first", stream=None):
        import torch
        import warp as wp
        from nvmath.bindings import cufft as cb
        self.torch, self.wp, self.cb = torch, wp, cb
        dev = "cuda"
        self.B = block_dim
        self.K = kernels(dt, if_mode, order)
        self.hi = torch.zeros((N, CC), dtype=torch.complex64, device=dev)
        self.lo = torch.zeros((CR, CC), dtype=torch.complex64, device=dev)
        self.T = [torch.zeros((CR, CC), dtype=torch.complex64, device=dev) for _ in range(3)]
        self.slot = [0, 1, 2]
        self.kinv = torch.as_tensor(np.asarray(tables["kinv"]), dtype=torch.float32, device=dev)
        self.edec = torch.as_tensor(np.asarray(tables["edec"]), dtype=torch.float32, device=dev)
        self.f0 = torch.as_tensor(np.asarray(tables["f0patch"]), dtype=torch.complex64, device=dev)
        self.W = torch.zeros((N, N), dtype=torch.complex64, device=dev)
        self.Z = torch.zeros((N, N), dtype=torch.complex64, device=dev)
        self.TH = torch.zeros((N, CCP), dtype=torch.complex64, device=dev)   # columns CC..CCP-1 stay zero
        self.q = None                                                # theta_phys() only
        self.stream = stream if stream is not None else torch.cuda.Stream()
        torch.cuda.synchronize()
        self.ws = wp.stream_from_torch(self.stream)
        v = (lambda t: wp.from_torch(torch.view_as_real(t), dtype=wp.vec2f))   # noqa: E731
        self.w_hi, self.w_lo, self.w_T = v(self.hi), v(self.lo), [v(t) for t in self.T]
        self.w_W, self.w_Z, self.w_TH, self.w_f0 = v(self.W), v(self.Z), v(self.TH), v(self.f0)
        self.w_kinv, self.w_edec = wp.from_torch(self.kinv), wp.from_torch(self.edec)
        s = self.stream.cuda_stream
        self.col_a = cb.create()                                     # W / Z: k_x = 0 .. KM, in place (tilerows')
        cb.make_plan_many(self.col_a, 1, [N], [N], N, 1, [N], N, 1, cb.Type.C2C, KM + 1)
        self.col_b = cb.create()                                     # W / Z: k_x = -KM .. -1
        cb.make_plan_many(self.col_b, 1, [N], [N], N, 1, [N], N, 1, cb.Type.C2C, KM)
        self.col_th = cb.create()                                    # theta: hi (N, CC) -> TH (N, CCP), out of place
        cb.make_plan_many(self.col_th, 1, [N], [N], CC, 1, [N], CCP, 1, cb.Type.C2C, CC)
        for h in (self.col_a, self.col_b, self.col_th):
            cb.set_stream(h, s)
        self.off_b = (N - KM) * 8
        self.ptr = {"W": self.W.data_ptr(), "Z": self.Z.data_ptr(), "hi": self.hi.data_ptr(), "TH": self.TH.data_ptr()}

    def pack(self, theta128, t_n, t_o):
        torch = self.torch
        th = np.asarray(theta128)
        outside = th.copy()
        outside[ROW_IDX, :CC] = 0
        if np.any(outside != 0):
            raise ValueError("state is not zero outside the compact kept block")
        thc = TS.to_compact(th)
        hi = thc.astype(np.complex64)
        self.hi.copy_(torch.as_tensor(to_fft_rows(hi, np.complex64)))
        self.lo.copy_(torch.as_tensor((thc - hi.astype(np.complex128)).astype(np.complex64)))
        self.slot = [0, 1, 2]
        self.T[0].copy_(torch.as_tensor(TS.to_compact(t_n).astype(np.complex64)))
        self.T[1].copy_(torch.as_tensor(TS.to_compact(t_o).astype(np.complex64)))
        torch.cuda.synchronize()

    def compact_hi(self):
        return self.hi[self.torch.as_tensor(ROW_IDX, device=self.hi.device)]

    def hi_outside_max(self):
        """max |hi| on the rows outside +-KM (must stay 0: nothing writes them)."""
        self.stream.synchronize()
        h = self.hi.cpu().numpy()
        h[ROW_IDX] = 0
        return float(np.max(np.abs(h)))

    def theta(self):
        self.stream.synchronize()
        hi = self.hi.cpu().numpy()[ROW_IDX]
        th = hi.astype(np.complex128) + self.lo.cpu().numpy().astype(np.complex128)
        return TS.from_compact(th, np.complex128)

    def _cols_th(self):
        self.cb.exec_c2c(self.col_th, self.ptr["hi"], self.ptr["TH"], INV)

    def _halve0(self):
        self.wp.launch(self.K["halve0"], dim=N, inputs=[self.w_TH], stream=self.ws)

    def theta_phys(self):
        """q_u = N^2 theta from the packed state by the in-tile path, and whether the out-of-place column pass left
        hi bit-identical: (q_u float32 numpy (N, N), hi_preserved)."""
        torch, wp = self.torch, self.wp
        if self.q is None:
            self.q = torch.zeros((N, N), dtype=torch.float32, device="cuda")
            self.w_q = wp.from_torch(self.q)
        self.stream.synchronize()
        before = self.hi.clone()                                     # on torch's default stream
        torch.cuda.synchronize()                                     # ... complete before the pass on self.stream
        self._cols_th()
        self._halve0()
        wp.launch_tiled(self.K["theta_rows"], dim=[N], inputs=[self.w_TH, self.w_q], block_dim=self.B, stream=self.ws)
        self.stream.synchronize()
        return self.q.cpu().numpy(), bool(torch.equal(before, self.hi))

    def ops(self):
        """The step as a list of (name, callable): the per-op split times these."""
        wp, cb, K, p, ws, B = self.wp, self.cb, self.K, self.ptr, self.ws, self.B
        return [
            ("build_w", lambda: wp.launch(K["build_w"], dim=(N, CK), inputs=[self.w_hi, self.w_kinv, self.w_W],
                                          stream=ws)),
            ("cufft_cols_w", lambda: (cb.exec_c2c(self.col_a, p["W"], p["W"], FWD),
                                      cb.exec_c2c(self.col_b, p["W"] + self.off_b, p["W"] + self.off_b, FWD))),
            ("cufft_cols_th", self._cols_th),
            ("halve0", self._halve0),
            ("tile_rows_ir", lambda: wp.launch_tiled(K["rows_ir"], dim=[N], inputs=[self.w_W, self.w_TH, self.w_Z],
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
