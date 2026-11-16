"""inrow_step.py -- the SQG solver step behind `tilerows_inrow_if<n>`: one float32 time step of the A-SQG model on the
GPU (Warp tile kernels + cuFFT on torch buffers), with theta's inverse real FFT folded into the row kernel ("in-tile
C2R", candidate 1 of warp/WARP_BUILTINS_AUDIT_2026-11-14.md). o1.2 runs it through o12run/tilerows_advance.py; the
probe (inrow_probe.py), the gate battery (tilegates/) and the precision probe (mixprec/) call it the same way.
Measured: 1.470 ms per IF-AB3 step on an A100 against 1.592 for tilerows_if25 (job 7853317); o1.2's 14-A100 schedule
48.9 h (job 7853365). Notes: SQG/polaris/polaris_sqg_notes.md Part XVII I. Predictions + gates: PREDICTIONS_inrow.md.

The model (SQG/fixed_ring/A-SQG_data.py, `inner`: "AB3 on advection + fixed forcing, then exact linear damping").
theta is real on an N x N doubly periodic grid (N = 4096). Spectra use the unnormalised FFT convention of the stored
state: theta_hat(k) = sum_x theta(x) e^{-i k.x}, theta(x) = N^-2 sum_k theta_hat(k) e^{i k.x}.
    d theta / dt = -div(u theta) + f0 - lambda(k) theta
    psi_hat = theta_hat / |k|,   u = -d psi / dy,   v = d psi / dx        (SQG inversion; div u = 0)
  - advection in flux form, computed pseudo-spectrally (products in physical space, derivatives in spectral space);
  - f0: A-SQG_data's fixed Hermitian ring forcing, inside the AB3 tendency (it lies in the (2P, P) low-mode patch);
  - lambda(k) = NU |k|^2 + MU / |k|^2 (k != 0): viscosity + large-scale drag, applied exactly as a decay factor;
  - dealiasing: circular 2/3 truncation |k| <= N / 3. The kept block here is the square |k_x|, |k_y| <= KM =
    floor(N / 3) = 1365; the decay table is 0 outside the circle, so those corner modes are zeroed every step.
One step of size h, with E = exp(-lambda h) and N_n the advection + forcing tendency at step n:
    AB3     theta_{n+1} = E [theta_n + h (23/12 N_n - 16/12 N_{n-1} + 5/12 N_{n-2})]          (if_mode False: lean32ek)
    IF-AB3  theta_{n+1} = E [theta_n + h (23/12 N_n - 16/12 E N_{n-1} + 5/12 E^2 N_{n-2})]   (if_mode True)

Buffers (float32 complex: torch.complex64 / wp.vec2f; CC = KM + 1, CR = CK = 2 KM + 1):
  hi    (N, CC)   theta_hat's float32 head in FFT row order (row = k_y mod N), columns k_x = 0..KM: the half plane,
                  since theta is real (theta_hat(-k) = conj theta_hat(k)). Rows with |k_y| > KM stay 0.
  lo    (CR, CC)  the float32 tail: theta_hat = hi + lo to ~48 bits (compact rows k_y + KM); see `update`
  T[3]  (CR, CC)  the AB3 history N_{n-1}, N_{n-2} and the slot the next update writes; `slot` rotates them
  W, Z  (N, N)    complex scratch over the full k_x range: W the velocity, Z the flux (u + i v) theta. Only the CK
                  kept columns (k_x = 0..KM and -KM..-1) are transformed; the others are dropped (dealiasing).
  TH    (N, CCP)  theta after the inverse column FFT, (y, k_x) for k_x = 0..KM; row pitch CCP = 1376 (128-B aligned)
The step, ops() in order (per-op medians of the non-IF step in job 7853317, ms):
  build_w        SIMT   W(k) = S (-k_x + i k_y) conj(psi_hat(k)), S = 1 / N^2                     0.148
  cufft_cols_w   cuFFT  forward column FFTs of W's kept columns: W(y, k_x)                         0.288
  cufft_cols_th  cuFFT  inverse column FFTs, hi -> TH(y, k_x), out of place (hi is the state)      0.129
  halve0         SIMT   TH[:, 0] /= 2, so the row inverse below equals a real (C2R) inverse        0.008
  tile_rows_ir   tile   per row y: N^2 theta = 2 Re IFFT(TH row); p = FFT(W row) = u - i v;
                        Z = conj(p) theta = (u + i v) theta; store FFT(Z) along the row            0.418
  cufft_cols_z   cuFFT  forward column FFTs of Z's kept columns: Z_hat = FFT(u theta) + i FFT(v theta)   0.291
  update         SIMT   split Z_hat into FFT(u theta), FFT(v theta); N_n = -i k . (them) + f0; (IF-)AB3;
                        exact damping; compensated (TwoSum / Fast2Sum errors carried in lo)        0.217
tilerows_step.TileRows made q_u = N^2 theta with expand (SIMT) + a 2-D cuFFT C2R (0.079 + 0.302 ms) and read it back
from memory. Here both ops are gone, for +0.115 ms in the row kernel plus theta's 0.129 ms column pass: 7% faster.
Identity (test_inrow_math.py, 2.4e-16): with G_0 = A_0 / 2 and G_k = A_k (1 <= k <= KM),
    2 Re sum_k G_k e^{2 pi i k x / N} = Re A_0 + 2 Re sum_{k>=1} A_k e^{2 pi i k x / N},
the Hermitian (C2R) inverse with Im A_0 dropped, as numpy's irfft does. The row kernel transforms theta first and
keeps only the real tile live during W's FFT (the warp probe's fused 3-FFT kernel, two complex tiles live, ran at
0.549 ms vs 0.302 for 2 FFTs: job 7840069). build_w, the W / Z column passes and update are tilerows_step's
arithmetic, re-indexed for the FFT-order hi.
"""
import numpy as np

import tilerows_step as TS

N, NK, KM, CR, CC, CK, P = TS.N, TS.NK, TS.KM, TS.CR, TS.CC, TS.CK, TS.P
ROW_IDX = TS.ROW_IDX                                                   # compact row r -> FFT row (k_y mod N)
CCP = (CC + 31) // 32 * 32                                             # TH row pitch: 1376 = 11,008 B, 128-B aligned
FWD, INV = -1, 1                                                       # cuFFT exponent signs: e^{-i...} / e^{+i...}
ORDERS = ("theta_first", "w_first")                                    # row-kernel FFT order; w_first is 8.8% slower


def to_fft_rows(compact, dtype):
    """(CR, CC) compact block -> (N, CC) FFT row order, zero rows between +-KM (the layout of InRow.hi)."""
    out = np.zeros((N, CC), dtype=dtype)
    out[ROW_IDX] = np.asarray(compact)
    return out


def kernels(dt, if_mode, order):
    """Compile the step's Warp kernels for step size dt.

    dt, the AB3 coefficients and S = 1 / N^2 are baked in as compile-time float32 constants; if_mode picks AB3 (False)
    or IF-AB3 (True) inside `update`; order picks the row kernel's FFT order (ORDERS). Every kernel is compiled with
    fuse_fp=False: the compensated update relies on error-free transformations (TwoSum / Fast2Sum) that FMA
    contraction would break, and the other kernels keep lean32ek's exact rounding sequence.

    Returns {"build_w", "halve0", "rows_ir", "theta_rows", "update"}: Warp kernels on float32 complex (wp.vec2f)
    arrays.
    """
    import warp as wp
    if order not in ORDERS:
        raise ValueError(f"order {order!r} not in {ORDERS}")
    opts = {"fuse_fp": False, "enable_backward": False}
    # float32 values held as Python floats (exact): Warp rejects numpy scalars as compile-time constants (job 7840360)
    C1, C2, C3 = (float(np.float32((23 / 12) * dt)), float(np.float32((-16 / 12) * dt)),
                  float(np.float32((5 / 12) * dt)))                  # AB3 weights times h, rounded once to float32
    IFM = int(bool(if_mode))
    S = float(np.float32(1.0 / (N * N)))                             # 1 / N^2 = 2^-24 exactly (N = 2^12)
    NN = N                                                           # the tile length, as a plain int for Warp

    @wp.kernel(module="unique", module_options=opts)
    def build_w(hi: wp.array2d(dtype=wp.vec2f), kinv: wp.array2d(dtype=wp.float32), W: wp.array2d(dtype=wp.vec2f)):
        """W(k) = S (-k_x + i k_y) conj(psi_hat(k)), psi_hat = kinv theta_hat, on the kept columns of W (FFT order).

        Why this form: the forward 2-D FFT of W is
            S conj(sum_k (-k_x - i k_y) psi_hat e^{i k.x}) = S conj(N^2 (u + i v)) = u - i v,
        with u = -d psi / dy, v = d psi / dx (d/dx <-> i k_x, so -k_x psi_hat <-> i d psi/dx = i v and
        -i k_y psi_hat <-> -d psi/dy = u). Building the
        CONJUGATE lets the row kernel use the same forward FFT for W and Z. One thread per (FFT row iy, kept column j);
        rows with |k_y| > KM are written as zero.
        """
        iy, j = wp.tid()                                             # j over the CK kept columns
        jx = j                                                       # kept index -> FFT column:
        if j > KM:
            jx = j + (N - CK)                                        #   j = KM+1..2KM -> k_x = -KM..-1 (cols N-KM..N-1)
        ky = iy                                                      # signed k_y of FFT row iy
        if iy >= N // 2:
            ky = iy - N
        kx = jx                                                      # the state holds k_x >= 0 only, so a negative
        kr = ky                                                      # column reads the mirror mode -k:
        if jx > N // 2:                                              # theta_hat(-k) = conj theta_hat(k) (theta real)
            kx = N - jx
            kr = -ky
        out = wp.vec2f(0.0, 0.0)
        if kr <= KM and kr >= -KM:                                   # outside the kept rows: 0 (dealiased)
            fr = kr                                                  # FFT row of k_y = kr
            if kr < 0:
                fr = kr + N
            t = hi[fr, kx]                                           # theta_hat (mirror: its conjugate), head only
            ki = kinv[kr + KM, kx]                                   # SQG inversion factor (1 / |k|)
            p_r = t[0] * ki                                          # P = psi_hat(k)  (mirror columns: conj psi_hat(k))
            p_i = t[1] * ki
            mi = S * wp.float32(ky)                                  # m = S (-k_x + i k_y): Im part S k_y
            if jx <= N // 2:
                mr = -(S * wp.float32(jx))                           # k_x = jx >= 0: Re m = -S k_x
                out = wp.vec2f(mr * p_r - mi * (-p_i), mr * (-p_i) + mi * p_r)   # m * conj(P)
            else:
                mr = S * wp.float32(N - jx)                          # k_x = jx - N < 0: Re m = -S k_x = S (N - jx)
                out = wp.vec2f(mr * p_r - mi * p_i, mr * p_i + mi * p_r)         # m * P, P already conj(psi_hat)
        W[iy, jx] = out

    @wp.kernel(module="unique", module_options=opts)
    def halve0(TH: wp.array2d(dtype=wp.vec2f)):
        """TH[:, 0] *= 1/2 (exact): G_0 = A_0 / 2, so rows_ir's 2 Re IFFT(G) is the C2R inverse (module docstring)."""
        i = wp.tid()
        TH[i, 0] = TH[i, 0] * wp.float32(0.5)

    @wp.func
    def re2(g: wp.vec2f):
        """2 Re g: the k_x >= 0 half-spectrum's complex inverse -> the real (C2R) inverse, q_u = N^2 theta."""
        return g[0] + g[0]                                           # q_u = 2 Re G (exact)

    @wp.func
    def zmul(a: wp.vec2f, q: wp.float32):                            # tilerows_step.zmul
        """conj(a) * (q S): with a = p = u - i v and q = q_u = N^2 theta, the flux (u + i v) theta in physical units."""
        qq = q * S                                                   # theta = q_u / N^2, exact
        r = a * wp.float32(0.0)                                      # a zero vec2f of a's type
        r[0] = a[0] * qq
        r[1] = (-a[1]) * qq                                          # conj(p) theta
        return r

    if order == "theta_first":
        @wp.kernel(module="unique", module_options=opts)
        def rows_ir(W: wp.array2d(dtype=wp.vec2f), TH: wp.array2d(dtype=wp.vec2f), Z: wp.array2d(dtype=wp.vec2f)):
            """One block per physical row y: Z(y, :) = FFT_x(conj(FFT_x(W(y, :))) theta(y, :)).

            theta(y, x) comes from TH's row by the in-tile C2R (q_u = 2 Re IFFT_x(G), G = TH(y, k_x >= 0), G_0 halved);
            FFT_x(W row) is p = u - i v (build_w); the product (u + i v) theta is transformed forward again so the
            column pass finishes its 2-D FFT. theta is done first so only its real tile is live during W's FFT.
            """
            y, _l = wp.tid()
            g = wp.tile_load(TH, shape=(1, NN), offset=(y, 0))       # k_x = 0..KM; columns >= CCP read as zero
            wp.tile_ifft(g)                                          # sum_k G_k e^{+2 pi i k x / N} (unnormalised)
            qr = wp.tile_map(re2, g)                                 # only the real tile stays live
            t = wp.tile_load(W, shape=(1, NN), offset=(y, 0))        # W(y, k_x), all N columns (non-kept are 0)
            wp.tile_fft(t)                                           # p row = (u - i v)(y, :)
            z = wp.tile_map(zmul, t, qr)                             # (u + i v) theta
            wp.tile_fft(z)                                           # forward along x: Z(y, k_x)
            wp.tile_store(Z, z, offset=(y, 0))
    else:
        @wp.kernel(module="unique", module_options=opts)
        def rows_ir(W: wp.array2d(dtype=wp.vec2f), TH: wp.array2d(dtype=wp.vec2f), Z: wp.array2d(dtype=wp.vec2f)):
            """rows_ir with W's FFT first: bit-identical, 8.8% slower (two complex tiles live; job 7853317)."""
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
        """One (IF-)AB3 step for compact mode (r, c) = (k_y + KM, k_x); see delta/f32_update.compensated_update.

        1. Hermitian split: u theta and v theta are real, so Z_hat = A + i B with A = FFT(u theta), B = FFT(v theta)
           recovers them as A = (Z_hat(k) + conj Z_hat(-k)) / 2, B = (Z_hat(k) - conj Z_hat(-k)) / (2 i).
        2. Tendency N_n = -(i k_x A + i k_y B) + f0 = FFT(-div(u theta)) + forcing (f0 only in the P x P patch).
        3. Increment inc = C1 N_n + C2 N_{n-1} + C3 N_{n-2} (AB3); IF-AB3 damps the history by E and E^2, written
           as t - e t and t - e2 t with e = edec = 1 - E, e2 = e (2 - e) = 1 - E^2: e keeps the small damping RATE
           to float32 relative precision (storing E rounds lambda h ~ 1e-7..5e-6 to a multiple of 6e-8;
           delta/f32_update.py).
        4. theta_hat = hi + lo; theta' = E (hi + lo + inc), compensated: TwoSum(hi, inc + lo) -> a + rr exactly;
           Fast2Sum(a, -e a) -> u + r2 exactly (|e a| <= |a|); hi' = u, lo' = r2 + E rr (the carried error decays
           too). Exact up to the rounding of the products e a and e rr.
        Writes hi, lo and tnew = N_n (the next steps' history).
        """
        r, c = wp.tid()
        iy = r - KM                                                  # FFT row of compact row r
        if r < KM:
            iy = r + (N - KM)
        z = Zh[iy, c]                                                # Z_hat(k)
        zm = Zh[(N - iy) % N, (N - c) % N]                           # Z_hat(-k)
        half = wp.float32(0.5)
        ar = (z[0] + zm[0]) * half                                   # A = (Z_hat(k) + conj Z_hat(-k)) / 2
        ai = (z[1] - zm[1]) * half                                   #   = FFT(u theta)
        br = (z[1] + zm[1]) * half                                   # B = (Z_hat(k) - conj Z_hat(-k)) / 2i
        bi = -((z[0] - zm[0]) * half)                                #   = FFT(v theta)
        kx = wp.float32(c)
        ky = wp.float32(r - KM)
        fr = wp.float32(0.0)                                         # forcing f0(k): nonzero only in the low-mode
        fi = wp.float32(0.0)                                         # patch k_y = -P..P-1, k_x = 0..P-1 (row k_y + P)
        pr = r - (KM - P)
        if pr >= 0 and pr < 2 * P and c < P:
            f = f0[pr, c]
            fr = f[0]
            fi = f[1]
        tr = -((-(kx * ai)) + (-(ky * bi))) + fr                     # N_n = -(i k_x A + i k_y B) + f0: real part
        ti = -((kx * ar) + (ky * br)) + fi                           #                                   imaginary part
        t_n = tn[r, c]                                               # N_{n-1}
        t_o = to[r, c]                                               # N_{n-2}
        e = edec[r, c]                                               # e = 1 - E = -expm1(-lambda h); 1 off the circle
        if IFM == 0:
            ir = (C1 * tr + C2 * t_n[0]) + C3 * t_o[0]               # AB3 increment
            ii = (C1 * ti + C2 * t_n[1]) + C3 * t_o[1]
        else:
            e2 = e * (wp.float32(2.0) - e)                           # 1 - E^2
            ir = (C1 * tr + C2 * (t_n[0] - e * t_n[0])) + C3 * (t_o[0] - e2 * t_o[0])   # IF-AB3: E N_{n-1}, E^2 N_{n-2}
            ii = (C1 * ti + C2 * (t_n[1] - e * t_n[1])) + C3 * (t_o[1] - e2 * t_o[1])
        h = hi[iy, c]                                                # theta_hat = hi + lo
        l = lo[r, c]
        yr = ir + l[0]                                               # y = inc + lo
        a_r = h[0] + yr                                              # TwoSum(hi, y): a + rr == hi + y exactly
        bp_r = a_r - h[0]
        rr = (h[0] - (a_r - bp_r)) + (yr - bp_r)
        ea_r = e * a_r
        u_r = a_r - ea_r                                             # u = E a (damping), Fast2Sum(a, -e a):
        r2_r = (a_r - u_r) - ea_r                                    #   u + r2 == a - e a exactly
        yi = ii + l[1]                                               # the same for the imaginary part
        a_i = h[1] + yi
        bp_i = a_i - h[1]
        ri = (h[1] - (a_i - bp_i)) + (yi - bp_i)
        ea_i = e * a_i
        u_i = a_i - ea_i
        r2_i = (a_i - u_i) - ea_i
        hi[iy, c] = wp.vec2f(u_r, u_i)                               # new head
        lo[r, c] = wp.vec2f(r2_r + (rr - e * rr), r2_i + (ri - e * ri))   # new tail: r2 + E rr
        tnew[r, c] = wp.vec2f(tr, ti)                                # N_n for the next two steps

    return {"build_w": build_w, "halve0": halve0, "rows_ir": rows_ir, "theta_rows": theta_rows, "update": update}


class InRow:
    """tilerows_step.TileRows with the in-tile C2R. Same interface (pack, theta, ops, step, run); self.hi is the
    (N, CC) FFT-row-order buffer (compact_hi() gives the (CR, CC) block).

    All work runs in order on one CUDA stream (self.stream): torch owns the buffers, Warp launches on the same stream
    (self.ws) and cuFFT is bound to it (set_stream), so no op needs a host sync. tables: "kinv" (compact SQG
    inversion factors), "edec" (compact 1 - E at step h; 1 outside |k| <= N/3) and "f0patch" (the (2P, P) forcing
    patch), all from o12run/tilerows_advance.make. dt is the step h; if_mode selects IF-AB3.
    """

    def __init__(self, tables, dt, *, if_mode=False, block_dim=256, order="theta_first", stream=None):
        import torch
        import warp as wp
        from nvmath.bindings import cufft as cb
        self.torch, self.wp, self.cb = torch, wp, cb
        dev = "cuda"
        self.B = block_dim                                           # threads per tile block (<= N/2; 256 is fastest)
        self.K = kernels(dt, if_mode, order)
        self.hi = torch.zeros((N, CC), dtype=torch.complex64, device=dev)
        self.lo = torch.zeros((CR, CC), dtype=torch.complex64, device=dev)
        self.T = [torch.zeros((CR, CC), dtype=torch.complex64, device=dev) for _ in range(3)]
        self.slot = [0, 1, 2]                                        # T indices: [N_{n-1}, N_{n-2}, next write]
        self.kinv = torch.as_tensor(np.asarray(tables["kinv"]), dtype=torch.float32, device=dev)
        self.edec = torch.as_tensor(np.asarray(tables["edec"]), dtype=torch.float32, device=dev)
        self.f0 = torch.as_tensor(np.asarray(tables["f0patch"]), dtype=torch.complex64, device=dev)
        self.W = torch.zeros((N, N), dtype=torch.complex64, device=dev)
        self.Z = torch.zeros((N, N), dtype=torch.complex64, device=dev)
        self.TH = torch.zeros((N, CCP), dtype=torch.complex64, device=dev)   # columns CC..CCP-1 stay zero
        self.q = None                                                # theta_phys() only
        self.stream = stream if stream is not None else torch.cuda.Stream()
        torch.cuda.synchronize()                                     # buffers initialised before the stream uses them
        self.ws = wp.stream_from_torch(self.stream)
        v = (lambda t: wp.from_torch(torch.view_as_real(t), dtype=wp.vec2f))   # noqa: E731  complex64 -> vec2f view
        self.w_hi, self.w_lo, self.w_T = v(self.hi), v(self.lo), [v(t) for t in self.T]
        self.w_W, self.w_Z, self.w_TH, self.w_f0 = v(self.W), v(self.Z), v(self.TH), v(self.f0)
        self.w_kinv, self.w_edec = wp.from_torch(self.kinv), wp.from_torch(self.edec)
        s = self.stream.cuda_stream
        # cuFFT plans: 1-D transforms of length N down columns (element stride N, adjacent columns 1 apart). Only the
        # kept columns are transformed: the dropped ones are the 2/3-rule truncation, so they cost nothing.
        self.col_a = cb.create()                                     # W / Z: k_x = 0 .. KM, in place (tilerows')
        cb.make_plan_many(self.col_a, 1, [N], [N], N, 1, [N], N, 1, cb.Type.C2C, KM + 1)
        self.col_b = cb.create()                                     # W / Z: k_x = -KM .. -1
        cb.make_plan_many(self.col_b, 1, [N], [N], N, 1, [N], N, 1, cb.Type.C2C, KM)
        self.col_th = cb.create()                                    # theta: hi (N, CC) -> TH (N, CCP), out of place
        cb.make_plan_many(self.col_th, 1, [N], [N], CC, 1, [N], CCP, 1, cb.Type.C2C, CC)
        for h in (self.col_a, self.col_b, self.col_th):
            cb.set_stream(h, s)
        self.off_b = (N - KM) * 8                                    # byte offset of column N - KM (complex64 = 8 B)
        self.ptr = {"W": self.W.data_ptr(), "Z": self.Z.data_ptr(), "hi": self.hi.data_ptr(), "TH": self.TH.data_ptr()}

    def pack(self, theta128, t_n, t_o):
        """Load a complex128 (N, NK) state theta_hat and its AB3 history (N_{n-1}, N_{n-2}): hi = float32(theta_hat)
        in FFT rows, lo = float32(theta_hat - hi), so hi + lo carries theta_hat to ~48 bits. Refuses a state that is
        nonzero outside the kept block (it would be silently truncated)."""
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
        """hi's kept rows as the (CR, CC) compact block (the layout of lo and T)."""
        return self.hi[self.torch.as_tensor(ROW_IDX, device=self.hi.device)]

    def hi_outside_max(self):
        """max |hi| on the rows outside +-KM (must stay 0: nothing writes them)."""
        self.stream.synchronize()
        h = self.hi.cpu().numpy()
        h[ROW_IDX] = 0
        return float(np.max(np.abs(h)))

    def theta(self):
        """The state as complex128 (N, NK) theta_hat = hi + lo (host copy; synchronises the stream)."""
        self.stream.synchronize()
        hi = self.hi.cpu().numpy()[ROW_IDX]
        th = hi.astype(np.complex128) + self.lo.cpu().numpy().astype(np.complex128)
        return TS.from_compact(th, np.complex128)

    def _cols_th(self):
        """theta's inverse column FFT: TH(y, k_x) = sum_{k_y} hi(k_y, k_x) e^{+i k_y y}, k_x = 0..KM, out of place
        (hi is the state and must survive)."""
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
        """The step as a list of (name, callable), in order (module docstring); the per-op split times these, and
        mixprec.InRowQ inserts its rounding between them. Each call only enqueues work on self.stream."""
        wp, cb, K, p, ws, B = self.wp, self.cb, self.K, self.ptr, self.ws, self.B
        return [
            ("build_w", lambda: wp.launch(K["build_w"], dim=(N, CK), inputs=[self.w_hi, self.w_kinv, self.w_W],
                                          stream=ws)),
            ("cufft_cols_w", lambda: (cb.exec_c2c(self.col_a, p["W"], p["W"], FWD),            # W(k) -> W(y, k_x)
                                      cb.exec_c2c(self.col_b, p["W"] + self.off_b, p["W"] + self.off_b, FWD))),
            ("cufft_cols_th", self._cols_th),
            ("halve0", self._halve0),
            ("tile_rows_ir", lambda: wp.launch_tiled(K["rows_ir"], dim=[N], inputs=[self.w_W, self.w_TH, self.w_Z],
                                                     block_dim=B, stream=ws)),
            ("cufft_cols_z", lambda: (cb.exec_c2c(self.col_a, p["Z"], p["Z"], FWD),            # Z(y, k_x) -> Z_hat(k)
                                      cb.exec_c2c(self.col_b, p["Z"] + self.off_b, p["Z"] + self.off_b, FWD))),
            ("update", self._update),
        ]

    def _update(self):
        """Launch update with N_{n-1} = T[a], N_{n-2} = T[b], writing N_n into the free slot f; then rotate, so the new
        tendency is next step's N_{n-1} and the oldest slot is reused."""
        a, b, f = self.slot
        self.wp.launch(self.K["update"], dim=(CR, CC), inputs=[self.w_Z, self.w_hi, self.w_lo, self.w_T[a],
                                                                self.w_T[b], self.w_edec, self.w_f0, self.w_T[f]],
                       stream=self.ws)
        self.slot = [f, a, b]

    def step(self):
        """One time step (enqueued; call self.stream.synchronize() before reading the state on the host)."""
        for _, op in self.ops():
            op()

    def run(self, n):
        """n time steps, enqueued back to back (the op list is built once)."""
        ops = [op for _, op in self.ops()]
        for _ in range(n):
            for op in ops:
                op()
