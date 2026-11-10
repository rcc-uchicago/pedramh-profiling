"""dose_common.py -- what gate D's trainer, rollouts and analysis share (numpy only; torch helpers take torch as an
argument). Geometry and bands are analyze_taxonomy.py's, so the injected bias pattern is defined exactly as job
7824121's regression measured it.

- Bias pattern B_hat(x, y). On o1.2's sub-frame rfft2 spectra, per coarse band b:
    B_hat = irfft2(alpha_b * X + beta_b * (Y - X)),    X = rfft2(x), Y = rfft2(y)
  where (alpha_b, beta_b) are lean32ek's level-1 coefficients (bias_pattern_lean32ek.json). It is the band-level
  conditional mean of lean32ek's paired error at the 0.02 lag. A dose a trains on the target y + a * B_hat(x, y).
- Band energy E_b(x): band b's contribution to the per-pixel mean square of x (Parseval on the rfft half-plane, o1.2
  units). E_low is the k <= 4 part (bands 1-2 and 2-4).
- Segments: the stored frames of gen_frames.py, m<MM>_t<tttt.tt>.npy, frame spacing 40 solver steps = 2e-4. A 0.02
  pair is frames (k, k + LAG), LAG = 100.
- Fixed draws, the same for every arm: the val pairs (VAL_SEED) and the rollout initial conditions.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "taxonomy"))
import analyze_taxonomy as AT                                         # noqa: E402  (geometry, BANDS, band_of)

NC, HALF, LAG = AT.NC, AT.HALF, 100
BANDS, BNAME = AT.BANDS, AT.BNAME
VAL_SEED = 20261110
LOW_BANDS = (0, 1)                                                    # k 1-2 and 2-4: the k <= 4 "large scales"
PATTERN = os.path.join(HERE, "bias_pattern_lean32ek.json")


def geometry():
    G = AT.geometry()
    G["w"] = np.where((G["KX"] > 0) & (G["KX"] < HALF), 2.0, 1.0)  # rfft half-plane -> full-plane weight
    return G


def bias_maps(G, pattern=PATTERN):
    """Per-mode real coefficient maps (CX, CD) on the (256, 129) rfft grid; 0 outside every band (k = 0)."""
    with open(pattern) as fh:
        p = json.load(fh)
    if [list(b) for b in p["bands"]] != [list(b) for b in BANDS]:
        raise ValueError("bias pattern bands differ from analyze_taxonomy.BANDS")
    cx, cd = np.zeros(G["band"].shape), np.zeros(G["band"].shape)
    for b in range(len(BANDS)):
        m = G["band"] == b
        cx[m], cd[m] = p["alpha"][b], p["beta"][b]
    return cx, cd


def bias_hat_np(x, y, cx, cd):
    """B_hat for real fields x, y of shape (..., 256, 256), float64."""
    X, Y = np.fft.rfft2(x), np.fft.rfft2(y)
    return np.fft.irfft2(cx * X + cd * (Y - X), s=(NC, NC))


def band_energy_np(x, G):
    """(..., 256, 256) -> (..., 8): per-band contribution to the per-pixel mean square."""
    X = np.fft.rfft2(np.asarray(x, dtype=np.float64))
    p = G["w"] * np.abs(X) ** 2 / NC ** 4
    return np.stack([p[..., G["band"] == b].sum(-1) for b in range(len(BANDS))], -1)


def torch_tools(torch, G, device):
    """(bias_hat(x, y), band_energy(x)) on the GPU for (B, 1, 256, 256) float32 tensors."""
    cx, cd = bias_maps(G)
    CX = torch.as_tensor(cx, dtype=torch.float32, device=device)
    CD = torch.as_tensor(cd, dtype=torch.float32, device=device)
    W = torch.as_tensor(G["w"] / NC ** 4, dtype=torch.float32, device=device)
    onehot = torch.as_tensor(np.stack([(G["band"] == b) for b in range(len(BANDS))], -1), dtype=torch.float32,
                             device=device)                          # (256, 129, 8)

    def bias_hat(x, y):
        X, Y = torch.fft.rfft2(x), torch.fft.rfft2(y)
        return torch.fft.irfft2(CX * X + CD * (Y - X), s=(NC, NC))

    def band_energy(x):
        p = W * torch.abs(torch.fft.rfft2(x)) ** 2                    # (B, 1, 256, 129)
        return torch.einsum("bcij,ijk->bk", p, onehot)               # (B, 8)

    return bias_hat, band_energy


def list_segments(frames_dir, members):
    """[(name, member, t_c, path)] of complete segments for the members, sorted by (member, t_c)."""
    out = []
    for fn in sorted(glob.glob(os.path.join(frames_dir, "m*_t*.npy"))):
        if fn.endswith(".tmp.npy"):
            continue
        name = os.path.basename(fn)[:-4]
        m = re.match(r"m(\d\d)_t([0-9.]+)$", name)
        meta_fn = os.path.join(frames_dir, name + ".json")
        if not m or not os.path.exists(meta_fn):
            continue
        with open(meta_fn) as fh:
            if not json.load(fh).get("complete"):
                continue
        member, t_c = int(m.group(1)), float(m.group(2))
        if member in members:
            out.append((name, member, t_c, fn))
    return sorted(out, key=lambda r: (r[1], r[2]))


def open_segments(frames_dir, members):
    segs = list_segments(frames_dir, members)
    return segs, [np.load(s[3], mmap_mode="r") for s in segs]


def pair_index(arrays):
    """All 0.02 pairs: (segment index, start frame) arrays."""
    seg = np.concatenate([np.full(a.shape[0] - LAG, i) for i, a in enumerate(arrays)])
    k = np.concatenate([np.arange(a.shape[0] - LAG) for a in arrays])
    return seg, k


def val_pairs(arrays, n):
    """The fixed held-out pairs (the same for every arm): x, y float32 (n, 256, 256)."""
    seg, k = pair_index(arrays)
    pick = np.sort(np.random.default_rng(VAL_SEED).choice(len(seg), size=min(n, len(seg)), replace=False))
    x = np.stack([arrays[seg[i]][k[i]] for i in pick]).astype(np.float32)
    y = np.stack([arrays[seg[i]][k[i] + LAG] for i in pick]).astype(np.float32)
    return x, y, pick


def rollout_ics(arrays, n_ic):
    """n_ic initial frames spread evenly over the held-out segments (n_ic / len(arrays) per segment)."""
    per = max(1, n_ic // len(arrays))
    ics, tags = [], []
    for i, a in enumerate(arrays):
        for j in range(per):
            k = int(j * (a.shape[0] - 1) / per)
            ics.append(np.asarray(a[k], dtype=np.float32))
            tags.append((i, k))
    return np.stack(ics[:n_ic]), tags[:n_ic]
