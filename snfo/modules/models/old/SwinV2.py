import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.layers.unpatchify import PatchInterpolate2D

# ----------------------------------------------------------------------------
# Utility Functions

def window_partition(x: torch.Tensor, window_size: tuple[int, int]):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: tuple[int, int], img_size: tuple[int, int]
):
    """(num_windows * B, window_size[0], window_size[1], C) -> (B, H, W, C)"""
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(
        -1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def get_shift_window_mask(
    grid_size: tuple[int, int],
    window_size: tuple[int, int],
    shift_size: tuple[int, int],
):
    """Compute attention mask for shifted windows on a lat/lon grid.

    Longitude is periodic, so shifted boundary windows along that axis contain
    genuinely adjacent tokens and need no masking.  Only latitude boundaries
    require masking after the cyclic shift.

    Returns:
        attn_mask: (n_windows, 1, win_h*win_w, win_h*win_w) or None if no
        latitude shift is needed.
    """
    H, W = grid_size
    wh, ww = window_size
    sh, _sw = shift_size

    if sh == 0:
        return None

    img_mask = torch.zeros((1, H, W, 1))

    lat_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))

    cnt = 0
    for lat in lat_slices:
        img_mask[:, lat, :, :] = cnt
        cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # (n_win, wh, ww, 1)
    mask_windows = mask_windows.view(-1, wh * ww)  # (n_win, wh*ww)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )

    return attn_mask.unsqueeze(1)  # (n_win, 1, wh*ww, wh*ww)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000):
    """Sinusoidal timestep embeddings."""
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype) / half
    ).to(device=t.device)
    args = t[:, None].to(t.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    embedding = (
        embedding.reshape(embedding.shape[0], 2, -1).flip(1).reshape(*embedding.shape)
    )  # flip sin/cos as done with edm

    return embedding


def build_2d_rope_table(
    height: int, width: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute 2D RoPE cos/sin tables for a (height x width) spatial grid.

    Splits head_dim in half: first half encodes row position, second half
    encodes column position.  Returns cos and sin tensors of shape
    (height*width, half_dim) where half_dim = head_dim // 2.
    """
    half_dim = head_dim // 2
    row_dim = (half_dim + 1) // 2  # ceil division — extra dim goes to rows
    col_dim = half_dim - row_dim

    row_freqs = 1.0 / (
        10000.0 ** (torch.arange(0, row_dim).float() / max(row_dim, 1))
    )
    col_freqs = 1.0 / (
        10000.0 ** (torch.arange(0, col_dim).float() / max(col_dim, 1))
    )

    rows = torch.arange(height).float()
    cols = torch.arange(width).float()

    # (H, row_dim) and (W, col_dim) outer products
    row_phases = torch.outer(rows, row_freqs)
    col_phases = torch.outer(cols, col_freqs)

    # broadcast to (H, W, row_dim) and (H, W, col_dim), then concat
    row_phases = row_phases[:, None, :].expand(-1, width, -1)
    col_phases = col_phases[None, :, :].expand(height, -1, -1)

    phases = torch.cat([row_phases, col_phases], dim=-1)  # (H, W, half_dim)
    phases = phases.reshape(height * width, half_dim)

    return phases.cos(), phases.sin()  # each (N, half_dim)


def apply_rope_2d(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply 2D rotary embeddings to x.

    Args:
        x:   (B, heads, N, head_dim)
        cos: (N, half_dim)   — from build_2d_rope_table
        sin: (N, half_dim)
    """
    d2 = cos.shape[-1]  # half_dim = head_dim // 2
    x1 = x[..., :d2]
    x2 = x[..., d2 : 2 * d2]
    x_pass = x[..., 2 * d2 :]  # dims that bypass RoPE (0 or 1 leftover)

    cos = cos[None, None]  # (1, 1, N, half_dim)
    sin = sin[None, None]

    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if x_pass.shape[-1] > 0:
        rotated = torch.cat([rotated, x_pass], dim=-1)
    return rotated


# ----------------------------------------------------------------------------
# Swin Modules

class LatentEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=True)
        self.l2 = nn.Linear(dim, dim, bias=True)

    def forward(self, emb):
        return F.silu(self.l2(F.silu(self.l1(emb))))


class LogCPB(nn.Module):
    """Log-spaced Continuous Position Bias (SwinV2).

    A small MLP maps log-normalized relative (row, col) offsets to per-head
    bias values.  Only the *unique* relative coordinates are stored and fed
    through the MLP; an index buffer expands them to the full (N, N) matrix.
    """

    def __init__(self, height: int, width: int, num_heads: int, mlp_dim: int = 512):
        super().__init__()
        self.num_heads = num_heads

        self.mlp = nn.Sequential(
            nn.Linear(2, mlp_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, num_heads, bias=False),
        )

        # --- unique relative coordinate table ---
        rel_h = torch.arange(-(height - 1), height).float()  # (2H-1,)
        rel_w = torch.arange(-(width - 1), width).float()    # (2W-1,)
        rel_grid = torch.stack(
            torch.meshgrid(rel_h, rel_w, indexing="ij")
        )  # (2, 2H-1, 2W-1)
        rel_table = rel_grid.permute(1, 2, 0).contiguous()  # (2H-1, 2W-1, 2)

        # SwinV2 log-space transform: normalize to [-8, 8] then sign*log2(|x|+1)/log2(8)
        if height > 1:
            rel_table[:, :, 0] /= height - 1
        if width > 1:
            rel_table[:, :, 1] /= width - 1
        rel_table *= 8.0
        rel_table = (
            torch.sign(rel_table)
            * torch.log2(torch.abs(rel_table) + 1.0)
            / math.log2(8)
        )

        self.register_buffer(
            "relative_coords_table", rel_table.reshape(-1, 2)
        )  # ((2H-1)*(2W-1), 2)

        # --- pairwise index into the table ---
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # (2, H, W)
        coords_flat = coords.reshape(2, -1)  # (2, N)

        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel[0] += height - 1  # shift to non-negative
        rel[1] += width - 1
        rel_index = rel[0] * (2 * width - 1) + rel[1]  # (N, N)
        self.register_buffer("relative_position_index", rel_index.long())

    def forward(self) -> torch.Tensor:
        """Returns position bias of shape (1, num_heads, N, N)."""
        bias_table = self.mlp(self.relative_coords_table)  # (T, num_heads)
        N = self.relative_position_index.shape[0]
        bias = bias_table[self.relative_position_index.view(-1)].view(
            N, N, self.num_heads
        )
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)


class ModulatedNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps)
        self.modulation = nn.Linear(dim, dim * 2, bias=True)

    def forward(self, x, t):
        x = self.norm(x)  # b, n, d
        scale, shift = self.modulation(t).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    """SwiGLU FeedForward"""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = ModulatedNorm(dim)
        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, t):
        gate, up_proj = self.w1(x).chunk(2, dim=-1)
        x = self.w2(F.silu(gate) * up_proj)
        x = self.norm(x, t)  # new: post-norm
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.norm = ModulatedNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)

        self.scale = nn.Parameter(torch.log(10 * torch.ones(1, heads, 1, 1)))

    def forward(self, x, t, mask=None, rope=None, pos_bias=None):
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.heads)
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply 2D RoPE before QK normalization
        if rope is not None:
            cos, sin = rope
            q = apply_rope_2d(q, cos, sin)
            k = apply_rope_2d(k, cos, sin)

        q = (
            F.normalize(q, dim=-1)
            * torch.clamp(self.scale, max=math.log(1.0 / 0.01)).exp()
        )
        k = F.normalize(k, dim=-1)

        # Combine attention mask and position bias
        attn_bias = None
        if mask is not None and pos_bias is not None:
            attn_bias = mask + pos_bias
        elif mask is not None:
            attn_bias = mask
        elif pos_bias is not None:
            attn_bias = pos_bias

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=1.0)

        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.wo(x)
        x = self.norm(x, t)  # new: post-norm
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        heads,
        window_size,
        grid_size,
        shift_size,
        global_every: int = 3,
    ):
        super().__init__()

        self.window_size = window_size
        self.grid_size = grid_size
        self.shift_size = shift_size
        self.depth = depth

        assert grid_size[0] % window_size[0] == 0 and grid_size[1] % window_size[1] == 0, (
            f"grid_size {grid_size} must be divisible by window_size {window_size}"
        )

        head_dim = dim // heads
        mlp_dim = int(8 / 3.0 * dim)

        # Which layers use global (full-grid) attention
        self.is_global = [
            (global_every > 0 and (i + 1) % global_every == 0)
            for i in range(depth)
        ]

        # Attention + FF layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads, head_dim),
                        FeedForward(dim, mlp_dim),
                    ]
                )
                for _ in range(depth)
            ]
        )

        # Per-layer log-CPB (window-sized or grid-sized depending on layer type)
        self.cpb_layers = nn.ModuleList(
            [
                LogCPB(grid_size[0], grid_size[1], heads)
                if self.is_global[i]
                else LogCPB(window_size[0], window_size[1], heads)
                for i in range(depth)
            ]
        )

        # 2D RoPE tables (precomputed, shared across layers of same type)
        win_cos, win_sin = build_2d_rope_table(
            window_size[0], window_size[1], head_dim
        )
        self.register_buffer("win_rope_cos", win_cos)
        self.register_buffer("win_rope_sin", win_sin)

        grid_cos, grid_sin = build_2d_rope_table(
            grid_size[0], grid_size[1], head_dim
        )
        self.register_buffer("grid_rope_cos", grid_cos)
        self.register_buffer("grid_rope_sin", grid_sin)

        # Shift-window attention mask (only used by windowed layers)
        attn_mask = get_shift_window_mask(grid_size, window_size, shift_size)
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        sh, sw = self.shift_size
        do_shift: bool = any(self.shift_size)

        # t expanded for windowed layers (one copy per window)
        repeat_factor = (self.grid_size[0] // self.window_size[0]) * (
            self.grid_size[1] // self.window_size[1]
        )
        t_expanded = t.repeat_interleave(repeat_factor, dim=0)

        for i, (attn, ff) in enumerate(self.layers):
            xp = x
            pos_bias = self.cpb_layers[i]()

            if self.is_global[i]:
                # ---- global attention (full grid) ----
                x = attn(
                    x, t, mask=None,
                    rope=(self.grid_rope_cos, self.grid_rope_sin),
                    pos_bias=pos_bias,
                )
            else:
                # ---- windowed attention ----
                x = x.view(-1, self.grid_size[0], self.grid_size[1], x.shape[-1])
                B, h, w, d = x.shape

                use_shift = do_shift and i % 2 != 0

                if use_shift:
                    x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))

                x = window_partition(x, self.window_size)
                x = x.view(-1, self.window_size[0] * self.window_size[1], d)

                mask = None
                if use_shift and self.attn_mask is not None:
                    mask = self.attn_mask.repeat(B, 1, 1, 1)

                x = attn(
                    x, t_expanded, mask=mask,
                    rope=(self.win_rope_cos, self.win_rope_sin),
                    pos_bias=pos_bias,
                )

                x = x.view(-1, self.window_size[0], self.window_size[1], d)
                x = window_reverse(x, self.window_size, (h, w))

                if use_shift:
                    x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
                x = x.view(-1, h * w, d)

            x = xp + x
            x = x + ff(x, t)

        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, dim):
        super().__init__()
        self.patch_size = p1, p2 = patch_size
        self.emb = nn.Linear(in_channels * p1 * p2, dim)

    def forward(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        return self.emb(x)

# ----------------------------------------------------------------------------
# Swin Transformer Class


class SwinV2(nn.Module):
    def __init__(
        self,
        img_resolution,
        in_channels: int,
        out_channels: int,
        window_size,
        shift_size,
        patch_size,
        depth: int = 6,
        dim: int = 512,
        heads: int = 12,
        global_every: int = 3,
    ):
        super().__init__()

        image_height, image_width = img_resolution
        patch_height, patch_width = patch_size
        grid_size = gh, gw = (image_height // patch_height, image_width // patch_width)

        self.pos_embed = nn.Parameter(torch.randn(1, gh * gw, dim) * 0.02)
        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim)
        self.t_embed = LatentEmbedding(dim)

        self.transformer = SwinTransformer(
            depth,
            dim,
            heads,
            window_size,
            grid_size,
            shift_size,
            global_every=global_every,  
        )

        self.out_layer = PatchInterpolate2D(grid_size, patch_size,
                                            dim, out_channels)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-init AdaLN modulation layers so residual blocks start as identity
        for module in self.modules():
            if isinstance(module, ModulatedNorm):
                nn.init.constant_(module.modulation.weight, 0)
                nn.init.constant_(module.modulation.bias, 0)

        # at the end of _init_weights, after the ModulatedNorm loop:                                                                                                                                                                                                                    
        if isinstance(self.out_layer, PatchInterpolate2D):
            nn.init.constant_(self.out_layer.adaLN_shift_scale[-1].weight, 0)                                                                                                                                                                                                           
            nn.init.constant_(self.out_layer.adaLN_shift_scale[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        c_grid: torch.Tensor,
    ):  
        
        x = torch.cat([x, cond, c_grid], dim = 1)  # concat conditioning to noised state

        x = self.patch_embed(x)  # b, n, d
        x = x + self.pos_embed  

        if t.dim() == 0 or (t.dim() == 1 and t.size(0) == 1):
            t = t.repeat(x.size(0))

        t = self.t_embed(timestep_embedding(t, x.size(2)))  # b, d

        x = self.transformer(x, t)  # b, n, d
        
        x = self.out_layer(x, t)

        return x # b c h w