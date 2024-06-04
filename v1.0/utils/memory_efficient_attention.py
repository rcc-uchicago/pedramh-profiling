import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
#import xformers.ops as xops
from torch.nn.attention import SDPBackend, sdpa_kernel

def memory_efficient_attention_torch(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None
) -> torch.Tensor:
    L = Q.shape[-2]
    S = K.shape[-2]
    scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
    if attn_mask is not None or is_causal:
        attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
        assert attn_mask is not None
        attn_mask = (
                torch.masked_fill(
                    attn_mask,
                    (attn_mask.bitwise_not()),
                    -float('inf'),
                ) if attn_mask.dtype == torch.bool
                else attn_mask
        )
        # attn_mask = (
        #         attn_mask.masked_fill(
        #             attn_mask.bitwise_not()
        #             # not attn_mask,
        #             -float('inf')
        #         ) if attn_mask.dtype==torch.bool else attn_mask
        # )
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1)
    else:
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * scale_factor), dim=-1)
    attn_weight = F.dropout(attn_weight, dropout_p)
    return attn_weight @ V

class MemEffAttentionTorch(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = head_dim**-0.5
        self.scale = 1.0 / math.sqrt(head_dim)
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        total_params = sum(p.numel() for p in self.qkv.parameters())
        print(f"Number of parameters: {total_params}")
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(0, 3, 2, 1, 4)
        q, k, v = torch.unbind(qkv, 2)
        x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                #attn_mask=attn_bias,
                dropout_p=self.attn_drop,
                scale=self.scale)
        """
        x = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=self.attn_drop,
                scale=self.scale
            )
        """
        return self.proj_drop(self.proj(x.permute(0, 2, 1, 3).reshape([B, N, C])))
