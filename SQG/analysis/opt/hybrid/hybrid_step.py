"""hybrid_step.py -- the A-SQG solver step as 3 cuFFT calls with fused LTO callbacks (nvmath-python) + 1 Warp kernel.

Same mathematics as seed 2's step (seed2_opt.make_step / lean32_precise.make_step). The data path changes:
  1. C2C forward, in place on p (N x N): its LOAD callback builds  w = s (-k_x + i k_y) conj(psi_hat)  on the fly from
     the compact theta_hat and kinv tables (seed2_opt.build_w's formula; k_x < 0 from the Hermitian mirror). No build_w
     pass, no w array. Result p = u - i v.
  2. C2R (N x nk -> N x N real): its LOAD callback reads the compact theta_hat (the input buffer is scratch, so the
     state is never overwritten: no input copy). Result q_u = N^2 theta (unnormalised).
  3. C2C forward, in place on p: its LOAD callback forms  Z = conj(p) * (q_u * s)  (s = 1/N^2, a power of two: exact),
     replacing the product pass and the two RFFTs of (u theta), (v theta); its STORE callback writes only the kept
     modes |k_x|, |k_y| <= KMAX into a compact square Zc (no full-plane write).
  4. One Warp kernel on the compact kept block: Hermitian split  A = (Z(k) + conj Z(-k))/2,  B = (Z(k) - conj Z(-k))/2i
     (= rfft2(u theta), rfft2(v theta)), tendency T = -(i k_x A + i k_y B) + F0, then the AB3 / IF-AB3 increment and the
     state update: float32 = lean32ek's compensated update (f32_update.compensated_update, component for component,
     compiled with fuse_fp=False so TwoSum / Fast2Sum stay error-free); float64 = lean_u1's (theta + inc) * decay.
State and history live on the compact block |k_y| <= KMAX (rows ky + KMAX), 0 <= k_x <= KMAX: the decay table carries
the 2/3 mask, so the state is exactly 0 outside it (research round 2, XR-1); `pack` checks that on entry.
Not bitwise to the JAX steps (different FFT decomposition and callback rounding): gated by tolerance (hybrid_probe.py).
Imports warp, torch, numba, nvmath inside the classes; module constants only at import.
"""
import numpy as np

N = 4096
NK = N // 2 + 1
KM = 1365                       # A-SQG KMAX = N/3 -> |k| <= 1365 kept
CR = 2 * KM + 1                 # compact rows: k_y = -KM .. KM  (row = k_y + KM)
CC = KM + 1                     # compact cols: k_x = 0 .. KM
B = CR * CC
PATCH = 8                       # forcing lives in |k_y| < 8, k_x < 8 (seed2_opt.PATCH)
ROW_IDX = np.concatenate([np.arange(N - KM, N), np.arange(0, KM + 1)])    # full-plane row of compact row r
FWD, INV = -1, 1                # cuFFT directions


def to_compact(a):
    """(N, NK) -> (CR, CC) numpy."""
    return np.ascontiguousarray(np.asarray(a)[ROW_IDX, :CC])


def from_compact(c, dtype):
    out = np.zeros((N, NK), dtype=dtype)
    out[ROW_IDX, :CC] = np.asarray(c)
    return out


# ---- the callbacks (hybrid_callbacks.py: numba device functions -> LTO-IR) -------------------------------------------
def compile_callbacks(fdt):
    """{name: LTO-IR bytes} for 'float32' or 'float64': w, q, z (load callbacks) and ze (the store callback)."""
    import nvmath
    import hybrid_callbacks as HC
    out = {}
    for name, (fn, elem, info) in HC.SETS[fdt].items():
        comp = nvmath.fft.compile_epilog if name == "ze" else nvmath.fft.compile_prolog
        out[name] = comp(fn, elem, info)
    return out


# ---- the Warp update kernels -----------------------------------------------------------------------------------------
def _kernels():
    import warp as wp
    opts = {"fuse_fp": False, "enable_backward": False}               # no FMA contraction: TwoSum / Fast2Sum exact

    @wp.kernel(module="unique", module_options=opts)
    def update32(Zc: wp.array2d(dtype=wp.vec2f), hi: wp.array2d(dtype=wp.vec2f), lo: wp.array2d(dtype=wp.vec2f),
                 tn: wp.array2d(dtype=wp.vec2f), to: wp.array2d(dtype=wp.vec2f), edec: wp.array2d(dtype=wp.float32),
                 f0: wp.array2d(dtype=wp.vec2f), tnew: wp.array2d(dtype=wp.vec2f),
                 c1: wp.float32, c2: wp.float32, c3: wp.float32, if_mode: int):
        r, c = wp.tid()
        z = Zc[r, c + KM]
        zm = Zc[2 * KM - r, KM - c]
        half = wp.float32(0.5)
        ar = (z[0] + zm[0]) * half                                   # A = rfft2(u theta)
        ai = (z[1] - zm[1]) * half
        br = (z[1] + zm[1]) * half                                   # B = rfft2(v theta)
        bi = -((z[0] - zm[0]) * half)
        kx = wp.float32(c)
        ky = wp.float32(r - KM)
        f0r = wp.float32(0.0)
        f0i = wp.float32(0.0)
        pr = r - (KM - PATCH)
        if pr >= 0 and pr < 2 * PATCH and c < PATCH:
            f = f0[pr, c]
            f0r = f[0]
            f0i = f[1]
        tr = -((-(kx * ai)) + (-(ky * bi))) + f0r                    # -(i k_x A + i k_y B) + F0
        ti = -((kx * ar) + (ky * br)) + f0i
        t_n = tn[r, c]
        t_o = to[r, c]
        e = edec[r, c]
        if if_mode == 0:
            ir = (c1 * tr + c2 * t_n[0]) + c3 * t_o[0]
            ii = (c1 * ti + c2 * t_n[1]) + c3 * t_o[1]
        else:                                                        # IF-AB3 (if_ab3.make_call_if32's float32 form)
            e2 = e * (wp.float32(2.0) - e)
            ir = (c1 * tr + c2 * (t_n[0] - e * t_n[0])) + c3 * (t_o[0] - e2 * t_o[0])
            ii = (c1 * ti + c2 * (t_n[1] - e * t_n[1])) + c3 * (t_o[1] - e2 * t_o[1])
        h = hi[r, c]
        l = lo[r, c]
        # f32_update.compensated_update, component by component
        yr = ir + l[0]
        ar_ = h[0] + yr
        bpr = ar_ - h[0]
        rr = (h[0] - (ar_ - bpr)) + (yr - bpr)
        ear = e * ar_
        ur = ar_ - ear
        r2r = (ar_ - ur) - ear
        yi = ii + l[1]
        ai_ = h[1] + yi
        bpi = ai_ - h[1]
        ri = (h[1] - (ai_ - bpi)) + (yi - bpi)
        eai = e * ai_
        ui = ai_ - eai
        r2i = (ai_ - ui) - eai
        hi[r, c] = wp.vec2f(ur, ui)
        lo[r, c] = wp.vec2f(r2r + (rr - e * rr), r2i + (ri - e * ri))
        tnew[r, c] = wp.vec2f(tr, ti)

    @wp.kernel(module="unique", module_options=opts)
    def update64(Zc: wp.array2d(dtype=wp.vec2d), th: wp.array2d(dtype=wp.vec2d),
                 tn: wp.array2d(dtype=wp.vec2d), to: wp.array2d(dtype=wp.vec2d), decay: wp.array2d(dtype=wp.float64),
                 f0: wp.array2d(dtype=wp.vec2d), tnew: wp.array2d(dtype=wp.vec2d),
                 c1: wp.float64, c2: wp.float64, c3: wp.float64, if_mode: int):
        r, c = wp.tid()
        z = Zc[r, c + KM]
        zm = Zc[2 * KM - r, KM - c]
        half = wp.float64(0.5)
        ar = (z[0] + zm[0]) * half
        ai = (z[1] - zm[1]) * half
        br = (z[1] + zm[1]) * half
        bi = -((z[0] - zm[0]) * half)
        kx = wp.float64(c)
        ky = wp.float64(r - KM)
        f0r = wp.float64(0.0)
        f0i = wp.float64(0.0)
        pr = r - (KM - PATCH)
        if pr >= 0 and pr < 2 * PATCH and c < PATCH:
            f = f0[pr, c]
            f0r = f[0]
            f0i = f[1]
        tr = -((-(kx * ai)) + (-(ky * bi))) + f0r
        ti = -((kx * ar) + (ky * br)) + f0i
        t_n = tn[r, c]
        t_o = to[r, c]
        d = decay[r, c]
        if if_mode == 0:                                             # lean_u1: qh + c1 T + c2 Tp + c3 Tpp, then * decay
            nr = ((th[r, c][0] + c1 * tr) + c2 * t_n[0]) + c3 * t_o[0]
            ni = ((th[r, c][1] + c1 * ti) + c2 * t_n[1]) + c3 * t_o[1]
        else:                                                        # IF-AB3 (if_ab3.make_advance_rt's 'if')
            nr = ((th[r, c][0] + c1 * tr) + c2 * (d * t_n[0])) + c3 * ((d * d) * t_o[0])
            ni = ((th[r, c][1] + c1 * ti) + c2 * (d * t_n[1])) + c3 * ((d * d) * t_o[1])
        th[r, c] = wp.vec2d(nr * d, ni * d)
        tnew[r, c] = wp.vec2d(tr, ti)

    return update32, update64


class Hybrid:
    """One SQG solver on the compact kept block. dtype 'float32' (lean32ek numerics: hi + lo state) or 'float64'
    (lean_u1 numerics). tables: kinv, edec (float32) or decay (float64), f0 patch (2 PATCH x PATCH, rows k_y = -8..7),
    all compact numpy arrays; dt the step."""

    def __init__(self, dtype, tables, dt, *, if_mode=False, kernels=None, callbacks=None, stream=None):
        import torch
        import warp as wp
        import nvmath
        from nvmath.bindings import cufft as cb
        self.torch, self.wp, self.cb = torch, wp, cb
        self.f32 = dtype == "float32"
        fdt = torch.float32 if self.f32 else torch.float64
        cdt = torch.complex64 if self.f32 else torch.complex128
        dev = "cuda"
        self.if_mode = int(bool(if_mode))
        self.dt = dt
        c = (23 / 12) * dt, (-16 / 12) * dt, (5 / 12) * dt            # pyqg_jax AB3 level-2 coefficients
        self.c = tuple((np.float32 if self.f32 else np.float64)(x) for x in c)
        # buffers
        self.U = torch.zeros(3 * B, dtype=fdt, device=dev)             # [theta_hat (re, im) | kinv]
        self.U[2 * B:] = torch.as_tensor(np.asarray(tables["kinv"]).reshape(-1), dtype=fdt, device=dev)
        self.th = torch.view_as_complex(self.U[:2 * B].view(CR, CC, 2))  # the (hi) state, a view into U
        self.lo = torch.zeros((CR, CC), dtype=cdt, device=dev) if self.f32 else None
        self.T = [torch.zeros((CR, CC), dtype=cdt, device=dev) for _ in range(3)]
        self.slot = [0, 1, 2]                                          # newest, previous, free
        self.damp = torch.as_tensor(np.asarray(tables["edec" if self.f32 else "decay"]), dtype=fdt, device=dev)
        self.f0 = torch.as_tensor(np.asarray(tables["f0patch"]), dtype=cdt, device=dev)
        self.Zc = torch.zeros(CR * CR, dtype=cdt, device=dev)
        self.p = torch.zeros((N, N), dtype=cdt, device=dev)
        self.qin = torch.zeros((N, NK), dtype=cdt, device=dev)         # C2R scratch input (read via the callback)
        self.q = torch.zeros((N, N), dtype=fdt, device=dev)
        # Warp views
        T2 = (lambda t: wp.from_torch(torch.view_as_real(t), dtype=wp.vec2f if self.f32 else wp.vec2d))  # noqa: E731
        self.wZ = T2(self.Zc.view(CR, CR))
        self.wth = T2(self.th)
        self.wlo = T2(self.lo) if self.f32 else None
        self.wT = [T2(t) for t in self.T]
        self.wdamp = wp.from_torch(self.damp)
        self.wf0 = T2(self.f0)
        k32, k64 = kernels if kernels is not None else _kernels()
        self.kernel = k32 if self.f32 else k64
        # cuFFT plans with callbacks (nvmath plans; executed below with cufft.xt_exec on fixed buffers)
        cbs = callbacks if callbacks is not None else compile_callbacks("float32" if self.f32 else "float64")
        # a dedicated non-default stream (CUDA-graph capture is not allowed on the legacy default stream)
        self.stream = stream if stream is not None else torch.cuda.Stream()
        torch.cuda.synchronize()
        self.ws = wp.stream_from_torch(self.stream)
        opt_c2c = {"fft_type": "C2C", "inplace": True}
        self.fw = nvmath.fft.FFT(self.p, axes=[0, 1], options=opt_c2c, execution="cuda", stream=self.stream)
        self.fw.plan(prolog={"ltoir": cbs["w"], "data": self.U.data_ptr()}, stream=self.stream)
        self.fq = nvmath.fft.FFT(self.qin, axes=[0, 1], options={"fft_type": "C2R", "last_axis_parity": "even",
                                                                "result_layout": "natural"},
                                 execution="cuda", stream=self.stream)
        self.fq.plan(prolog={"ltoir": cbs["q"], "data": self.U.data_ptr()}, stream=self.stream)
        self.fz = nvmath.fft.FFT(self.p, axes=[0, 1], options=opt_c2c, execution="cuda", stream=self.stream)
        self.fz.plan(prolog={"ltoir": cbs["z"], "data": self.q.data_ptr()},
                     epilog={"ltoir": cbs["ze"], "data": self.Zc.data_ptr()}, stream=self.stream)
        # one nvmath execute each: allocates and sets the cuFFT work areas (kept: release_workspace=False)
        self.fw.execute(direction="forward", stream=self.stream)
        self.fq.execute(stream=self.stream)
        self.fz.execute(direction="forward", stream=self.stream)
        self.layouts = {"w_in": self.fw.get_input_layout(), "q_in": self.fq.get_input_layout(),
                        "q_out": self.fq.get_output_layout(), "z_out": self.fz.get_output_layout()}
        for f in (self.fw, self.fq, self.fz):
            cb.set_stream(f.handle, self.stream.cuda_stream)
        self.handles = (self.fw.handle, self.fq.handle, self.fz.handle)
        self.ptr = (self.p.data_ptr(), self.qin.data_ptr(), self.q.data_ptr())

    # ---- state in / out -----------------------------------------------------------------------------------------------
    def pack(self, theta128, t_n, t_o):
        """Load a full-plane float64 state (numpy complex128 (N, NK)) and its AB3 history. hi = c64(theta), lo =
        c64(theta - hi) as lean32ek's entry (float32); theta itself (float64)."""
        torch = self.torch
        th = np.asarray(theta128)
        outside = th.copy()
        outside[ROW_IDX, :CC] = 0
        if np.any(outside != 0):
            raise ValueError("state is not zero outside the compact kept block")
        thc = to_compact(th)
        if self.f32:
            hi = thc.astype(np.complex64)
            self.th.copy_(torch.as_tensor(hi))
            self.lo.copy_(torch.as_tensor((thc - hi.astype(np.complex128)).astype(np.complex64)))
        else:
            self.th.copy_(torch.as_tensor(thc))
        cd = np.complex64 if self.f32 else np.complex128
        self.slot = [0, 1, 2]
        self.T[0].copy_(torch.as_tensor(to_compact(t_n).astype(cd)))
        self.T[1].copy_(torch.as_tensor(to_compact(t_o).astype(cd)))
        torch.cuda.synchronize()                                      # the copies ran on the default stream

    def theta(self):
        """The full-plane complex128 state (numpy): hi + lo summed in float64 (lean32ek's exit)."""
        self.stream.synchronize()
        th = self.th.cpu().numpy().astype(np.complex128)
        if self.f32:
            th = th + self.lo.cpu().numpy().astype(np.complex128)
        return from_compact(th, np.complex128)

    # ---- the step ---------------------------------------------------------------------------------------------------
    def step(self):
        cb, (hw, hq, hz), (pp, qi, qq) = self.cb, self.handles, self.ptr
        cb.xt_exec(hw, pp, pp, FWD)                                   # w (built on load) -> p = u - i v
        cb.xt_exec(hq, qi, qq, INV)                                   # theta_hat (read on load) -> q_u
        cb.xt_exec(hz, pp, pp, FWD)                                   # conj(p) theta -> Z_hat, kept modes -> Zc
        a, b, f = self.slot
        if self.f32:
            inputs = [self.wZ, self.wth, self.wlo, self.wT[a], self.wT[b], self.wdamp, self.wf0, self.wT[f],
                      self.c[0], self.c[1], self.c[2], self.if_mode]
        else:
            inputs = [self.wZ, self.wth, self.wT[a], self.wT[b], self.wdamp, self.wf0, self.wT[f],
                      self.c[0], self.c[1], self.c[2], self.if_mode]
        self.wp.launch(self.kernel, dim=(CR, CC), inputs=inputs, stream=self.ws)
        self.slot = [f, a, b]                                          # the new tendency is the newest

    def run(self, n):
        for _ in range(n):
            self.step()
