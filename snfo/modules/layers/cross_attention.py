import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.layers.old.fa_basics import MLP
from modules.layers.positional_encoding import apply_2d_rotary_pos_emb


class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional 2D RoPE on q and k.

    Uses F.scaled_dot_product_attention. Queries come from `x`, keys/values
    from `context`. If RoPE frequencies are provided they must correspond to
    the same grid for both the query and context token sequences (i.e. the
    context is expected to live on the same patch grid as the queries).
    """

    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context,
                rope_cos_lat=None, rope_sin_lat=None,
                rope_cos_lon=None, rope_sin_lon=None):
        b, nq, _ = x.shape
        nk = context.shape[1]
        h, d = self.heads, self.dim_head

        q = self.to_q(x).reshape(b, nq, h, d).transpose(1, 2)   # [b, h, nq, d]
        k = self.to_k(context).reshape(b, nk, h, d).transpose(1, 2)
        v = self.to_v(context).reshape(b, nk, h, d).transpose(1, 2)

        if rope_cos_lat is not None:
            q = apply_2d_rotary_pos_emb(q, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)
            k = apply_2d_rotary_pos_emb(k, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + MLP residual block (no time conditioning).

    Kept for non-DiT callers. For DiT use the AdaLN-gated cross-attention
    block defined alongside the model instead.
    """

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout=0.0,
            mlp_ratio=4,
    ):
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)
        self.Attn = CrossAttention(hidden_dim, hidden_dim,
                                   heads=num_heads,
                                   dim_head=hidden_dim // num_heads,
                                   dropout=dropout)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, expansion_ratio=mlp_ratio)

    def forward(self, q, kv, reshape=False,
                rope_cos_lat=None, rope_sin_lat=None,
                rope_cos_lon=None, rope_sin_lon=None):
        if reshape:
            _, h, w, _ = q.shape
            q = rearrange(q, 'b h w c -> b (h w) c')
            kv = rearrange(kv, 'b h w c -> b (h w) c')

        fx = self.Attn(self.ln_q(q), self.ln_kv(kv),
                       rope_cos_lat, rope_sin_lat,
                       rope_cos_lon, rope_sin_lon) + q
        fx = self.mlp(self.ln_2(fx)) + fx

        if reshape:
            fx = rearrange(fx, 'b (h w) c -> b h w c', h=h, w=w)

        return fx
