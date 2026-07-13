import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.layers.patchify import PatchEmbed
from modules.layers.cross_attention import CrossAttentionBlock
from modules.layers.unpatchify import Unpatchify
from modules.layers.positional_encoding import TimestepEmbedder


# ---------------------------------------------------------------------------
# 2-D axial RoPE helpers
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotate paired dims: [..., x1, x2] -> [..., -x2, x1]."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, freqs_cos, freqs_sin):
    """Apply RoPE rotation to x.

    x          : [..., n, d]
    freqs_cos  : [n, d] (or broadcastable)
    freqs_sin  : [n, d] (or broadcastable)
    """
    return x * freqs_cos + rotate_half(x) * freqs_sin


def build_rope_2d_cache(grid_h, grid_w, dim_head, device, theta=10000.0):
    """Pre-compute 2-D axial RoPE cosine/sine tables.

    The head dimension is split into two equal halves:
      - dims  0 .. dim_head//2 - 1   encode the height (row) axis.
      - dims  dim_head//2 .. dim_head - 1  encode the width (col) axis.

    Returns
    -------
    freqs_cos, freqs_sin : Tensor, each shape [grid_h * grid_w, dim_head]
    """
    assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2-D axial RoPE"
    quarter = dim_head // 4  # base frequencies per axis

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, quarter, dtype=torch.float32, device=device) / quarter)
    )

    h_pos = torch.arange(grid_h, dtype=torch.float32, device=device)
    w_pos = torch.arange(grid_w, dtype=torch.float32, device=device)

    h_angles = torch.outer(h_pos, inv_freq)  # [grid_h, quarter]
    w_angles = torch.outer(w_pos, inv_freq)  # [grid_w, quarter]

    # Duplicate so that rotate_half pairs see the same angle (standard RoPE).
    h_angles = torch.cat([h_angles, h_angles], dim=-1)  # [grid_h, dim_head//2]
    w_angles = torch.cat([w_angles, w_angles], dim=-1)  # [grid_w, dim_head//2]

    # Broadcast to every (row, col) position and flatten to a sequence.
    h_angles = h_angles.unsqueeze(1).expand(-1, grid_w, -1).reshape(-1, dim_head // 2)
    w_angles = w_angles.unsqueeze(0).expand(grid_h, -1, -1).reshape(-1, dim_head // 2)

    angles = torch.cat([h_angles, w_angles], dim=-1)  # [grid_h * grid_w, dim_head]
    return angles.cos(), angles.sin()


# ---------------------------------------------------------------------------
# DiT block with 2-D axial RoPE
# ---------------------------------------------------------------------------

class DiTBlockRoPE(nn.Module):
    """Self-attention DiT block with AdaLN-Zero conditioning and 2-D axial RoPE.

    Identical in structure to DiTBlock (SI_DiT.py) but applies rotary
    positional encoding to Q and K rather than relying on additive
    positional embeddings.

    Input/output shape: [b, n, dim].
    """

    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        dim_head = dim // num_heads

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim)
        self.scale = dim_head ** -0.5

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        # AdaLN-Zero: 6 * dim for (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t_emb, freqs_cos, freqs_sin):
        """
        Args:
            x         : [b, n, dim]
            t_emb     : [b, dim]  AdaLN conditioning (timestep embedding)
            freqs_cos : [n, dim_head]  pre-computed RoPE cosines
            freqs_sin : [n, dim_head]  pre-computed RoPE sines
        """
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)  # [b, 1, 6*dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Self-attention with AdaLN modulation
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa

        b, n, c = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, b, heads, n, dim_head]
        q, k, v = qkv.unbind(0)

        # Apply 2-D RoPE to queries and keys; broadcast over batch and heads.
        fc = freqs_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, n, dim_head]
        fs = freqs_sin.unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, fc, fs)
        k = apply_rope(k, fc, fs)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(b, n, c)
        h = self.attn_out(h)

        x = x + gate_msa * h

        # FFN with AdaLN modulation
        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h

        return x


class SuperResDiT(nn.Module):
    """
    DiT-based super-resolution decoder.

    Takes a low-resolution state (b, c_lr, h, w) and high-resolution prior
    history (b, c_hr, H, W). The low-res state is patchified with patch_size_lr,
    the high-res history is patchified with patch_size_hr (typically larger, e.g. 8).

    Low-res tokens go through self-attention blocks interleaved with cross-attention
    to the high-res history tokens, then are unpatchified to produce the high-res output.

    Args:
        in_dim_lr: Number of channels in low-resolution input.
        in_dim_hr: Number of channels in high-resolution history.
        out_dim: Number of output channels at high resolution.
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        num_blocks: Number of self-attention + cross-attention block pairs.
        patch_size_lr: Patch size for the low-resolution input.
        patch_size_hr: Patch size for the high-resolution history.
        h_lr, w_lr: Spatial dimensions of the low-resolution input.
        h_hr, w_hr: Spatial dimensions of the high-resolution output/history.
        dropout: Dropout rate.
        output_mode: "unpatchify" to reshape channels into spatial dims,
                     "channel_upsample" to project channel dim and reshape.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 dim,
                 num_heads,
                 num_blocks,
                 patch_size_lr=2,
                 patch_size_hr=8,
                 h_lr=45,
                 w_lr=90,
                 h_hr=180,
                 w_hr=360,
                 dropout=0.,
                 output_mode="unpatchify",
                 ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.patch_size_lr = patch_size_lr
        self.patch_size_hr = patch_size_hr
        self.h_lr = h_lr
        self.w_lr = w_lr
        self.h_hr = h_hr
        self.w_hr = w_hr
        self.out_dim = out_dim
        self.output_mode = output_mode

        # Grid sizes after patchification
        self.grid_h_lr = h_lr // patch_size_lr
        self.grid_w_lr = w_lr // patch_size_lr
        self.grid_h_hr = h_hr // patch_size_hr
        self.grid_w_hr = w_hr // patch_size_hr

        # Patch embeddings
        self.patch_embed_lr = PatchEmbed(patch_size=patch_size_lr,
                                         in_chans=in_dim,
                                         hidden_size=dim,
                                         flatten=True)

        self.patch_embed_hr = PatchEmbed(patch_size=patch_size_hr,
                                         in_chans=in_dim,
                                         hidden_size=dim,
                                         flatten=True)

        # Learnable positional embedding for the HR context (cross-attention keys/values).
        # LR tokens rely on RoPE instead of an additive embedding.
        num_patches_hr = self.grid_h_hr * self.grid_w_hr
        self.pos_embed_hr = nn.Parameter(torch.zeros(1, num_patches_hr, dim))
        nn.init.trunc_normal_(self.pos_embed_hr, std=0.02)

        # 2-D axial RoPE cache for LR tokens (registered as a buffer so it moves
        # to the correct device automatically and is not treated as a parameter).
        dim_head = dim // num_heads
        rope_cos, rope_sin = build_rope_2d_cache(
            self.grid_h_lr, self.grid_w_lr, dim_head, device=torch.device('cpu')
        )
        self.register_buffer('rope_cos_lr', rope_cos)  # [n_lr, dim_head]
        self.register_buffer('rope_sin_lr', rope_sin)  # [n_lr, dim_head]

        # Conditioning embedder (scalar timestep or just a bias)
        self.cond_embed = TimestepEmbedder(dim)

        # Self-attention (RoPE) + cross-attention blocks
        self.sa_blocks = nn.ModuleList()
        self.ca_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.sa_blocks.append(DiTBlockRoPE(dim, num_heads, dropout=dropout))
            self.ca_blocks.append(CrossAttentionBlock(num_heads, dim, dropout=dropout))

        # Output head
        if output_mode == "unpatchify":
            # The lr patch grid must match the hr output after unpatchify.
            # Compute the upscale factor per spatial dim.
            upscale_h = h_hr // self.grid_h_lr
            upscale_w = w_hr // self.grid_w_lr
            assert upscale_h == upscale_w, \
                f"Upscale factors must match: got {upscale_h} x {upscale_w}"
            up_patch = upscale_h
            self.unpatchify = Unpatchify(
                grid_size=(self.grid_h_lr, self.grid_w_lr),
                patch_size=(up_patch, up_patch),
                in_dim=dim,
                out_dim=out_dim,
                cond_dim=dim,
            )
        elif output_mode == "channel_upsample":
            # Project channels to cover the spatial upscale, then reshape.
            self.upscale_h = h_hr // h_lr
            self.upscale_w = w_hr // w_lr
            out_tokens = out_dim * (self.upscale_h * patch_size_lr) * (self.upscale_w * patch_size_lr)
            self.out_proj = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, out_tokens),
            )
        else:
            raise ValueError(f"Unknown output_mode: {output_mode}")

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Re-init HR pos embed after apply (LR uses RoPE — no learnable embed to restore).
        nn.init.trunc_normal_(self.pos_embed_hr, std=0.02)

    def forward(self, x_lr, x_hr, t=None):
        """
        Args:
            x_lr: Low-resolution state, shape (b, c_lr, h, w).
            x_hr: High-resolution prior history, shape (b, c_hr, H, W).
            t: Optional scalar conditioning, shape (b, 1). Defaults to ones.

        Returns:
            output: High-resolution state, shape (b, c_out, H, W).
        """
        batch_size = x_lr.shape[0]

        # Convert to channel-last for PatchEmbed: (b, c, h, w) -> (b, h, w, c)
        x_lr = x_lr.permute(0, 2, 3, 1)
        x_hr = x_hr.permute(0, 2, 3, 1)

        # Patchify: (b, h, w, c) -> (b, num_patches, dim)
        x_lr = self.patch_embed_lr(x_lr)  # (b, grid_h_lr * grid_w_lr, dim)
        x_hr = self.patch_embed_hr(x_hr)  # (b, grid_h_hr * grid_w_hr, dim)

        # LR tokens: positional information comes from RoPE (applied inside each SA block).
        # HR context tokens: additive learnable positional embedding.
        x_hr = x_hr + self.pos_embed_hr

        if t is None:
            t = torch.ones((batch_size, 1), device=x_lr.device)  # (b, 1)

        c = self.cond_embed(t)  # (b, dim)

        # Transformer blocks: RoPE self-attention on LR, cross-attention to HR context
        for i in range(self.num_blocks):
            x_lr = self.sa_blocks[i](x_lr, c, self.rope_cos_lr, self.rope_sin_lr)
            x_lr = self.ca_blocks[i](x_lr, x_hr)

        # Output
        if self.output_mode == "unpatchify":
            # x_lr: (b, grid_h_lr * grid_w_lr, dim)
            out = self.unpatchify(x_lr, c)  # (b, H, W, out_dim)
            out = out.permute(0, 3, 1, 2)  # (b, out_dim, H, W)
        elif self.output_mode == "channel_upsample":
            # x_lr: (b, grid_h_lr * grid_w_lr, dim)
            out = self.out_proj(x_lr)  # (b, grid_h_lr * grid_w_lr, out_dim * up_h * p * up_w * p)
            out = rearrange(out,
                            'b (gh gw) (c uh ph uw pw) -> b c (gh uh ph) (gw uw pw)',
                            gh=self.grid_h_lr, gw=self.grid_w_lr,
                            uh=self.upscale_h, uw=self.upscale_w,
                            ph=self.patch_size_lr, pw=self.patch_size_lr,
                            c=self.out_dim)

        return out
