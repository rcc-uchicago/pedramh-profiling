"""ConvNextUViT: ConvNeXt UNet with global self-attention at the bottleneck.

Compact rewrite of ML/ConvNextUnetV4.py + ML/ConvNextUViT.py, same architecture and same parameters
(ML checkpoints load with load_ml_state_dict; outputs identical). Differences to the ML files:
  - no gradient checkpointing inside the UNet stages (not needed at 256^2),
  - no padding for odd sizes: N must be a multiple of 2**(len(depths) + 1) (64 for 5 levels).

Shapes for N = 256, n_feat = 32, depths = [2, 2, 4, 4, 8]:
  input (B, 1, 256, 256)
  down1..down5 -> 32 x 128^2, 64 x 64^2, 128 x 32^2, 256 x 16^2, 512 x 8^2
  to_vec       -> 512 x 4^2  (16 tokens) -> attention blocks -> up0 -> 512 x 8^2
  up1..up5     -> back to 32 x 256^2 -> out -> (B, 1, 256, 256)
Padded convolutions use circular padding (periodic domain).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from torch.nn.attention import SDPBackend, sdpa_kernel


class CappedGELU(nn.Module):
    """GELU clamped from above at `cap`."""

    def __init__(self, cap=1.0):
        super().__init__()
        self.register_buffer("cap", torch.tensor(cap, dtype=torch.float32))

    def forward(self, x):
        return torch.clamp(F.gelu(x), max=self.cap)


class ConvNeXtBlock(nn.Module):
    """depthwise conv -> LayerNorm -> MLP (x4) -> LayerScale -> DropPath, plus a (1x1) skip."""

    def __init__(self, in_ch, latent_ch, out_ch, kernel_size=7, drop_path=0.0):
        super().__init__()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size // 2, groups=in_ch)
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        self.pw1 = nn.Linear(in_ch, 4 * latent_ch)
        self.act = CappedGELU()
        self.pw2 = nn.Linear(4 * latent_ch, out_ch)
        self.gamma = nn.Parameter(1e-6 * torch.ones(out_ch))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        h = self.dwconv(x).permute(0, 2, 3, 1)                   # NHWC
        h = self.pw2(self.act(self.pw1(self.norm(h)))) * self.gamma
        return self.skip(x) + self.drop_path(h.permute(0, 3, 1, 2))


class UnetDown(nn.Module):
    """ConvNeXt blocks (first one changes the width), then a stride-2 conv."""

    def __init__(self, in_ch, out_ch, kernel_size, num_blocks, drop_path=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(in_ch if i == 0 else out_ch, in_ch if i == 0 else out_ch, out_ch, kernel_size, drop_path)
            for i in range(num_blocks))
        self.downsample = nn.Conv2d(out_ch, out_ch, 2, stride=2, bias=False)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.downsample(x)


class UnetUp(nn.Module):
    """concat(x, skip) -> stride-2 transposed conv -> ConvNeXt blocks."""

    def __init__(self, in_ch, out_ch, kernel_size, num_blocks, drop_path=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(out_ch, out_ch, out_ch, kernel_size, drop_path) for _ in range(num_blocks))

    def forward(self, x, skip):
        x = self.up(torch.cat([x, skip], 1))
        for blk in self.blocks:
            x = blk(x)
        return x


class AxialRoPE(nn.Module):
    """2D rotary position embedding, periodic in the token grid (integer wavenumbers 1 .. S/2)."""

    def __init__(self, head_dim):
        super().__init__()
        assert head_dim % 4 == 0
        self.head_dim = head_dim
        self._cache = {}

    def _angles(self, S, device):
        n = self.head_dim // 4
        geom = torch.logspace(0, math.log10(max(S // 2, 1)), n, dtype=torch.float64).round().tolist()
        m, prev = [], 0
        for g in geom:                                           # n distinct integer wavenumbers
            prev = max(prev + 1, int(g))
            m.append(prev)
        freqs = 2 * math.pi * torch.tensor(m, device=device, dtype=torch.float64) / S
        return torch.arange(S, device=device, dtype=torch.float64)[:, None] * freqs[None]   # (S, n)

    def forward(self, x, S_h, S_w):
        key = (S_h, S_w, x.device, x.dtype)
        if key not in self._cache:
            n = self.head_dim // 4
            a_h, a_w = self._angles(S_h, x.device), self._angles(S_w, x.device)
            ang = torch.cat([a_h[:, None, :].expand(S_h, S_w, n), a_w[None, :, :].expand(S_h, S_w, n)], -1)
            ang = ang.reshape(S_h * S_w, -1)
            self._cache[key] = (ang.cos().to(x.dtype), ang.sin().to(x.dtype))
        cos, sin = self._cache[key]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class AttnBlock(nn.Module):
    """Pre-LN transformer block on a (B, C, S_h, S_w) map; LayerScale 1e-6 (identity at init)."""

    def __init__(self, dim, n_heads=8, mlp_ratio=4, drop_path=0.0):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = AxialRoPE(self.head_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.gamma1 = nn.Parameter(1e-6 * torch.ones(dim))
        self.gamma2 = nn.Parameter(1e-6 * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def attention(self, t, S_h, S_w):
        B, L, C = t.shape
        q, k, v = self.qkv(t).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, S_h, S_w), self.rope(k, S_h, S_w)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):   # fp32-capable kernels
            o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, L, C))

    def forward(self, x):
        B, C, S_h, S_w = x.shape
        t = x.flatten(2).transpose(1, 2)                          # (B, L, C) tokens
        t = t + self.drop_path(self.gamma1 * self.attention(self.norm1(t), S_h, S_w))
        t = t + self.drop_path(self.gamma2 * self.mlp(self.norm2(t)))
        return t.transpose(1, 2).reshape(B, C, S_h, S_w)


class ConvNextUViT(nn.Module):
    """residual=True : returns x + net(x)  (tendency model, big skip)
       residual=False: returns net(x)                  (full-field model)"""

    def __init__(self, in_ch=1, out_ch=1, n_feat=32, depths=(2, 2, 4, 4, 8), kernel_size=7, drop_path=0.1,
                 n_attn=4, n_heads=8, residual=True):
        super().__init__()
        L = len(depths)
        chs = [n_feat * 2 ** l for l in range(L)]
        self.n_levels, self.residual = L, residual

        self.proj_in = nn.Conv2d(in_ch, n_feat, 3, 1, 1)                       # full-resolution input skip
        self.init_conv = ConvNeXtBlock(in_ch, n_feat, n_feat, kernel_size, drop_path)
        for l in range(L):
            self.add_module(f"down{l + 1}", UnetDown(chs[l - 1] if l else n_feat, chs[l], kernel_size, depths[l]))
        self.to_vec = nn.Sequential(nn.Conv2d(chs[-1], chs[-1], 2, stride=2, bias=False), nn.GELU())
        self.mid = nn.Sequential(*[AttnBlock(chs[-1], n_heads, drop_path=drop_path) for _ in range(n_attn)])
        self.up0 = nn.Sequential(nn.ConvTranspose2d(chs[-1], chs[-1], 2, 2), nn.GroupNorm(8, chs[-1]), nn.ReLU())
        for m in range(1, L + 1):
            l = L - m
            self.add_module(f"up{m}", UnetUp(2 * chs[l], chs[l - 1] if l else n_feat, kernel_size, depths[l]))
        self.out = nn.Sequential(nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), nn.GroupNorm(8, n_feat), nn.ReLU(),
                                 nn.Conv2d(n_feat, out_ch, 3, 1, 1))

        for mod in self.modules():                                              # periodic domain
            if isinstance(mod, nn.Conv2d) and any(p > 0 for p in mod.padding):
                mod.padding_mode = "circular"

    def forward(self, x):
        assert x.shape[-1] % 2 ** (self.n_levels + 1) == 0 and x.shape[-2] % 2 ** (self.n_levels + 1) == 0
        x_in_proj = self.proj_in(x)
        h, downs = self.init_conv(x), []
        for l in range(self.n_levels):
            h = getattr(self, f"down{l + 1}")(h)
            downs.append(h)
        up = self.up0(self.mid(self.to_vec(h)))
        for m in range(1, self.n_levels + 1):
            up = getattr(self, f"up{m}")(up, downs[self.n_levels - m])
        out = self.out(torch.cat([up, x_in_proj], 1))
        return x + out if self.residual else out


def load_ml_state_dict(model, state_dict):
    """Load a checkpoint of ML/ConvNextUViT (its convs are wrapped in NormalLayer: '.layer.' in the keys)."""
    return model.load_state_dict({k.replace(".layer.", "."): v for k, v in state_dict.items()})
