"""mixprec.py -- probe 1 of PREDICTIONS_mixprec.md (written first): emulated 16-bit STORAGE of the in-row step's
intermediate arrays, with fixed or adaptive per-column precision, to measure what the gates say before anyone builds
16-bit kernels. The user (2026-11-16) asked about adaptive mixed precision ("it chooses between types but avoids
error and has been applied in Cholesky factorization").

The solver is inrow_step.InRow (tilerows_inrow_if25) unchanged. InRowQ runs the same kernels in the same order and,
after each op that writes an intermediate to global memory, rounds the array's kept columns to 16 bits:
  W_spec  after build_w        W (k_y, k_x), spectral          W_mix  after cufft_cols_w   W (y, k_x), mixed
  TH      after halve0         theta (y, k_x), mixed           Z_mix  after tile_rows_ir   Z (y, k_x), mixed
  Z_spec  after cufft_cols_z   Z (k_y, k_x), spectral          T      after update         the newest tendency slot
The model is "16-bit storage, fp32 compute": each pass still computes in float32 (an optimistic bound for a real
16-bit FFT). The state (hi + lo, float32 with lean32ek's compensation) is never rounded. Formats:
  bf16  torch's float32 -> bfloat16 round-to-nearest-even (8 significant bits, float32's range): unit roundoff 2^-8.
  fp16  per-column power-of-two scale s = 2^(15 - e) (column max |x| < 2^e), float16 RNE, unscale: block floating
        point, unit roundoff 2^-11. (Unscaled fp16 would underflow: W carries 1 / N^2.)
Every array is column-major in k_x, so precision is chosen per column (k_x); the kept columns are k_x <= KM.
Policies (the scheme suffix: tilerows_inrow_if25@<policy>):
  none         no rounding: the op list is InRow's own (must reproduce tilerows_inrow_if25 bit for bit)
  bf16_all     every kept column bf16           fp16_all     every kept column fp16
  fp16_kx<K>   columns with |k_x| > K fp16, the rest float32 (K = 128: the scales the 256^2 frames do not resolve)
  ad<tau>      adaptive, per array per call: column c gets the lowest precision p in (bf16, fp16, fp32) with
               u_p^2 |X_c|^2 <= (tau u_32)^2 |X|^2 / n (n kept columns). Summed over columns, the added rounding
               error is then <= tau x float32's own (|X| u_32): the per-tile bucketing of adaptive-precision Cholesky.
Numpy versions of both formats (np_round) are the login-node reference for test_mixprec.py.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
for _p in ("tilerows", "inrow"):
    if os.path.join(OPT, _p) not in sys.path:
        sys.path.insert(0, os.path.join(OPT, _p))

U32, U = 2.0 ** -24, {"bf16": 2.0 ** -8, "fp16": 2.0 ** -11}
POINTS = ("W_spec", "W_mix", "TH", "Z_mix", "Z_spec", "T")
AFTER = {"build_w": "W_spec", "cufft_cols_w": "W_mix", "halve0": "TH", "tile_rows_ir": "Z_mix",
         "cufft_cols_z": "Z_spec", "update": "T"}
REGISTRY = {}                                                        # policy -> solver (tr_pairs.py reads summary())


def parse_policy(p):
    if p == "none":
        return {"kind": "none"}
    if p in ("bf16_all", "fp16_all"):
        return {"kind": "all", "fmt": p[:4]}
    if p.startswith("fp16_kx"):
        return {"kind": "kx", "fmt": "fp16", "K": int(p[len("fp16_kx"):])}
    if p.startswith("ad"):
        tau = float(p[2:])
        if not tau > 0:
            raise ValueError(f"policy {p!r}: tau must be > 0")
        return {"kind": "ad", "tau": tau}
    raise ValueError(f"unknown mixprec policy {p!r}")


def buckets(e, tau):
    """Adaptive precision per column from column energies e = |X_c|^2 (numpy or torch, 1-D): 2 = bf16, 1 = fp16,
    0 = float32. Lowest precision p with u_p^2 e_c <= (tau u32)^2 sum(e) / n."""
    n = e.shape[0]
    budget = (tau * U32) ** 2 * float(e.sum()) / n
    bf = e * U["bf16"] ** 2 <= budget
    fp = e * U["fp16"] ** 2 <= budget
    return bf * 1 + fp * 1                                           # bf implies fp: 2, 1 or 0


def np_round(x, fmt):
    """Reference rounding of a float32 array (..., m) column-wise (last axis = column): bf16 RNE by bit arithmetic;
    fp16 with the per-column power-of-two scale, numpy's float16 RNE."""
    x = np.asarray(x, dtype=np.float32)
    if fmt == "bf16":
        i = x.view(np.int32)
        lsb = (i >> 16) & 1
        return ((i + np.int32(0x7FFF) + lsb) & np.int32(-65536)).astype(np.int32).view(np.float32)
    axes = tuple(range(x.ndim - 1))
    amax = np.max(np.abs(x), axis=axes) if x.size else np.zeros(x.shape[-1], np.float32)
    _, e = np.frexp(amax)
    s = np.ldexp(np.float32(1.0), np.minimum(15 - e, 126)).astype(np.float32)
    return ((x * s).astype(np.float16).astype(np.float32) / s).astype(np.float32)


def torch_round_cols_(X, idx, fmt):
    """Round columns idx (long tensor) of the complex64 2-D tensor X in place (fmt bf16 | fp16, as np_round)."""
    import torch
    if idx.numel() == 0:
        return
    sub = torch.view_as_real(X.index_select(1, idx))                 # (R, m, 2) float32
    if fmt == "bf16":
        q = sub.to(torch.bfloat16).to(torch.float32)
    else:
        amax = torch.linalg.vector_norm(sub, ord=float("inf"), dim=(0, 2))
        _, e = torch.frexp(amax)
        s = torch.ldexp(torch.ones_like(amax), torch.clamp(15 - e, max=126))[None, :, None]
        q = (sub * s).to(torch.float16).to(torch.float32) / s
    X.index_copy_(1, idx, torch.view_as_complex(q.contiguous()))


def make_solver(tables, h, *, if_mode, policy, block_dim=256):
    """An inrow_step.InRow whose op list rounds the intermediates by `policy` (registered for summary())."""
    import torch
    import inrow_step as IRS
    pol = parse_policy(policy)
    N, KM, CC = IRS.N, IRS.KM, IRS.CC

    class InRowQ(IRS.InRow):
        def __init__(self):
            super().__init__(tables, h, if_mode=if_mode, block_dim=block_dim)
            dev = self.hi.device
            full = torch.cat([torch.arange(0, KM + 1), torch.arange(N - KM, N)]).to(dev)   # kept columns of (N, N)
            kx_full = torch.where(full <= KM, full, N - full)
            cc = torch.arange(CC, device=dev)
            self.lay = {"full": (full, kx_full), "cc": (cc, cc)}
            self.layout = {"W_spec": "full", "W_mix": "full", "TH": "cc", "Z_mix": "full", "Z_spec": "full", "T": "cc"}
            self.cnt = {a: [0, 0, 0, 0] for a in POINTS}                # bf16 cols, fp16 cols, all cols, calls
            if pol["kind"] in ("all", "kx"):
                self.static = {}
                for name, (idx, kx) in self.lay.items():
                    self.static[name] = idx if pol["kind"] == "all" else idx[kx > pol["K"]]

        def array(self, a):
            return {"W_spec": self.W, "W_mix": self.W, "TH": self.TH, "Z_mix": self.Z, "Z_spec": self.Z,
                    "T": self.T[self.slot[0]]}[a]                       # T: the slot update() just wrote

        def q(self, a):
            X = self.array(a)
            lay = self.layout[a]
            idx, _ = self.lay[lay]
            c = self.cnt[a]
            with torch.cuda.stream(self.stream):
                if pol["kind"] == "ad":
                    e = torch.linalg.vector_norm(X, dim=0).double().square()[idx]
                    b = buckets(e, pol["tau"])
                    sel_bf, sel_fp = idx[b == 2], idx[b == 1]               # host sync (data-dependent sizes)
                    torch_round_cols_(X, sel_bf, "bf16")
                    torch_round_cols_(X, sel_fp, "fp16")
                    c[0] += int(sel_bf.numel())
                    c[1] += int(sel_fp.numel())
                else:
                    sel = self.static[lay]
                    torch_round_cols_(X, sel, pol["fmt"])
                    c[0 if pol["fmt"] == "bf16" else 1] += int(sel.numel())
            c[2] += int(idx.numel())
            c[3] += 1

        def ops(self):
            base = IRS.InRow.ops(self)
            if pol["kind"] == "none":
                return base
            out = []
            for name, op in base:
                out.append((name, op))
                if name in AFTER:
                    out.append((f"q_{AFTER[name]}", (lambda a=AFTER[name]: self.q(a))))
            return out

    s = InRowQ()
    REGISTRY[policy] = s
    return s


def summary():
    """Per policy and array: the mean fractions of kept columns stored bf16 / fp16 / 16-bit, and the calls."""
    out = {}
    for policy, s in REGISTRY.items():
        out[policy] = {}
        for a, (nb, nf, n, calls) in s.cnt.items():
            out[policy][a] = {"bf16": nb / n if n else 0.0, "fp16": nf / n if n else 0.0,
                              "f16": (nb + nf) / n if n else 0.0, "calls": calls}
    return out


def projected_ms(fractions, step_ms):
    """Upper bound on the step saving if 16-bit storage halved the bytes of the stored fraction of every array
    (all of the step's traffic is taken as these arrays'): 0.5 x mean fraction x step."""
    f = [fractions[a]["f16"] for a in POINTS if a in fractions]
    return 0.5 * (sum(f) / len(f) if f else 0.0) * step_ms


if __name__ == "__main__":
    print(__doc__)
    print("u_bf16 / u32 = %g, u_fp16 / u32 = %g" % (U["bf16"] / U32, U["fp16"] / U32), math.log2(U["fp16"] / U32))
