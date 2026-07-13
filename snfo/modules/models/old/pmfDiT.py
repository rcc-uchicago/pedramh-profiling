import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class TorchLinear(nn.Module):
    """A linear layer similar to torch.nn.Linear."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_init="scaled_variance",
        init_constant=1.0,
        bias_init="zeros",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_init = weight_init
        self.init_constant = init_constant
        self.bias_init = bias_init

        if self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.in_features)
            weight_initializer = partial(nn.init.normal_, std=std)
        elif self.weight_init == "zeros":
            weight_initializer = nn.init.zeros_
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        if self.bias_init == "zeros":
            bias_initializer = nn.init.zeros_
        else:
            raise ValueError(f"Invalid bias_init: {self.bias_init}")

        self._flax_linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
        )
        weight_initializer(self._flax_linear.weight)
        if self.bias:
            bias_initializer(self._flax_linear.bias)

    def forward(self, x):
        return self._flax_linear(x)


class TorchEmbedding(nn.Module):
    """A embedding layer similar to torch.nn.Embedding."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        weight_init="scaled_variance",
        init_constant=1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_init = weight_init
        self.init_constant = init_constant

        if self.weight_init is None:
            std = 0.02
        elif self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.embedding_dim)
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        self._flax_embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
        )
        nn.init.normal_(self._flax_embedding.weight, std=std)

    def forward(self, x):
        return self._flax_embedding(x)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.dim))

    def _norm(self, x):
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps)

    def forward(self, x):
        output = self._norm(x).to(x.dtype)
        return output * self.weight


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP."""

    def __init__(
        self,
        in_features,
        hidden_features,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        init_kwargs = dict(
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        self.w1 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w3 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w2 = TorchLinear(self.hidden_features, self.in_features, **init_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TimestepEmbedder(nn.Module):
    """Embeds a scalar timestep (or scalar conditioning) into a vector."""

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        weight_init="scaled_variance",
        init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.weight_init = weight_init
        self.init_constant = init_constant

        init_kwargs = dict(
            out_features=self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
            bias_init="zeros",
        )
        self.mlp = nn.Sequential(
            TorchLinear(self.frequency_embedding_size, **init_kwargs),
            nn.SiLU(),
            TorchLinear(self.hidden_size, **init_kwargs),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        args = t[:, None].to(torch.float32) * freqs[None].to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], axis=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class BottleneckPatchEmbedder(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self, input_size, initial_patch_size, pca_channels, in_channels, hidden_size, bias=True
    ):
        super().__init__()
        self.input_size = input_size
        self.initial_patch_size = initial_patch_size
        self.in_channels = in_channels
        self.pca_channels = pca_channels
        self.hidden_size = hidden_size
        self.bias = bias

        self.patch_size = (self.initial_patch_size[0], self.initial_patch_size[1])
        self.img_size = self.input_size
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            self.img_size
        )

        self.flatten = True
        self.proj1 = nn.Conv2d(
            self.in_channels,
            self.pca_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.bias,
        )
        self.proj2 = nn.Conv2d(
            self.pca_channels,
            self.hidden_size,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=self.bias,
        )

        # init proj1 weights like nn.Linear, instead of nn.Conv2d
        kh = kw = self.patch_size[0]
        fan_in = kh * kw * self.in_channels
        fan_out = self.pca_channels
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj1.weight, -limit, limit)

        # init proj2 weights like nn.Linear, instead of nn.Conv2d
        fan_in = self.pca_channels
        fan_out = self.hidden_size
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj2.weight, -limit, limit)
        if self.bias:
            nn.init.zeros_(self.proj1.bias)
            nn.init.zeros_(self.proj2.bias)
    
    def _init_img_size(self, img_size):
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def forward(self, x):
        B, C, H, W = x.shape  # (2, 32, 32, 4)
        x = self.proj2(self.proj1(x))  # (B, H/p, W/p, hidden_c)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, x.shape[1])  # NCHW -> NLC
        return x

def unsqueeze(t, dim):
    """Adds a new axis to a tensor at the given position."""
    return t.unsqueeze(dim)


#################################################################################
#                   Modern Transformer Components with Vec Gates               #
#################################################################################


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and QK RMS norm."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        init_kwargs = dict(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)

        self.head_dim = self.hidden_size // self.num_heads

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_freqs):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)

        # manually implement attention to match JAX implementation
        query = q / math.sqrt(self.head_dim)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, k)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        attn = attn.reshape(batch, seq_len, self.hidden_size)

        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized vector gates on residuals."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=8/3,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        self.norm1 = RMSNorm(self.hidden_size)
        self.attn = RoPEAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )
        self.norm2 = RMSNorm(self.hidden_size)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        # round mlp hidden dim to multiple of 8
        if hidden_size > 1024: # only for HSDP code
            mlp_hidden_dim = (mlp_hidden_dim + 7) // 8 * 8
        self.mlp = SwiGLUMlp(
            self.hidden_size,
            mlp_hidden_dim,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )

        self.attn_scale = nn.Parameter(torch.zeros(self.hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, x, rope_freqs):
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero init weights."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.norm = RMSNorm(self.hidden_size)
        self.linear = TorchLinear(
            self.hidden_size,
            self.patch_size[0] * self.patch_size[1] * self.out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def forward(self, x):
        return self.linear(self.norm(x))


#################################################################################
#                improved MeanFlow DiT with In-context Conditioning             #
#################################################################################


class pmfDiT(nn.Module):
    """
    MeanFlow improved Transformer (pmfDiT).
    A shared backbone processes the first (depth - aux_head_depth) layers.
    Two heads of equal depth (aux_head_depth) branch off afterwards.
    """

    def __init__(
        self,
        input_size = [180, 360],
        patch_size = [6, 6],
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: float = 8 / 3,
        aux_head_depth: int = 8,
        num_time_tokens: int = 4,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
    ):
        """
        Set up the pmfDiT model components.
         - Patch embedder for input images.
         - Embedders for time, omega, cfg intervals, and class labels.
         - Learnable tokens for conditioning.
         - Transformer blocks with shared backbone and dual heads.
         - Final projection layers for u and v outputs.
        """
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.aux_head_depth = aux_head_depth

        self.num_time_tokens = num_time_tokens

        self.token_init_constant = token_init_constant
        self.embedding_init_constant = embedding_init_constant
        self.weight_init_constant = weight_init_constant

        self.out_channels = out_channels

        self.x_embedder = BottleneckPatchEmbedder(
            self.input_size,
            self.patch_size,
            256, # pca channels. 256 for H/G
            self.in_channels,
            self.hidden_size,
            bias=True,
        )

        embed_kwargs = dict(
            hidden_size=self.hidden_size,
            weight_init="scaled_variance",
            init_constant=self.embedding_init_constant,
        )

        self.h_embedder = TimestepEmbedder(**embed_kwargs)

        token_initializer = partial(
            nn.init.normal_, std=self.token_init_constant / math.sqrt(self.hidden_size)
        )
        self.time_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_time_tokens, self.hidden_size))
        )

        total_tokens = (
            self.x_embedder.num_patches
            + self.num_time_tokens
        )
        self.prefix_tokens =self.num_time_tokens
        self.head_dim = self.hidden_size // self.num_heads
        self.register_buffer("rope_freqs", precompute_rope_freqs(self.head_dim, self.x_embedder.grid_size))
        self.pos_embed = nn.Parameter(
            nn.init.normal_(torch.empty(1, total_tokens, self.hidden_size), std=0.02)
        )

        head_depth = self.aux_head_depth
        shared_depth = self.depth - head_depth

        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=self.weight_init_constant,
        )

        self.shared_blocks = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(shared_depth)]
        )
        self.u_heads = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(head_depth)]
        )

        # We don't need the v heads during evaluation
        self.v_heads = nn.ModuleList(
            [
                TransformerBlock(**block_kwargs)
                for _ in range(head_depth)
            ]
        )

        self.u_final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )
        self.v_final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        ) 

    def unpatchify(self, x):
        c = self.out_channels
        p_h, p_w = self.x_embedder.patch_size
        h, w = self.x_embedder.grid_size

        x = x.reshape((x.shape[0], h, w, p_h, p_w, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        images = x.reshape((x.shape[0], c, h * p_h, w * p_w))
        return images

    def _build_sequence(self, x, h):
        """
        Build the input token sequence for the transformer.
        1. Embed the input image patches.
        2. Embed the conditioning information (time, omega, cfg, class labels).
        3. Prepend the conditioning tokens to the patch embeddings.

        Args:
            x: Input images
            h: timestep
            w: CFG scale
            t_min, t_max: CFG interval
            y: Class labels

        Returns:
            seq: Token sequence for the transformer
        """

        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)

        time_tokens = self.time_tokens + unsqueeze(h_embed, 1)

        seq = torch.cat(
            [
                time_tokens,
                x_embed,
            ],
            axis=1,
        )
        seq = seq + self.pos_embed

        return seq

    def forward(self, x, t, h, cond):
        """
        Forward pass of the pmfDiT model.
        Returns the predicted u and v components.

        Args:
            x: Input images
            t, h: time steps

        Returns:
            u: Average velocity field
            v: Instantaneous velocity field
        """
        x = torch.cat((x, cond), dim=1) # b c h w'

        seq = self._build_sequence(x, h)

        for block in self.shared_blocks:
            seq = block(seq, self.rope_freqs)

        u_seq = v_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq, self.rope_freqs)

        for block in self.v_heads:
            v_seq = block(v_seq, self.rope_freqs)

        u_tokens = u_seq[:, self.prefix_tokens :]
        v_tokens = v_seq[:, self.prefix_tokens :]

        u = self.unpatchify(self.u_final_layer(u_tokens))
        v = self.unpatchify(self.v_final_layer(v_tokens))

        return u, v

#################################################################################
#                           Rotary Position Helpers                             #
#################################################################################


def precompute_rope_freqs(dim: int, grid_size: tuple, theta: float = 10000.0):
    dim = dim // 2 # for 2d rotary embeddings
    H, W = grid_size
    seq_len = H * W
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions_h = torch.arange(H, dtype=torch.float32)
    positions_w = torch.arange(W, dtype=torch.float32)
    freqs_h = torch.einsum('i,j->ij', positions_h, freqs)
    freqs_w = torch.einsum('i,j->ij', positions_w, freqs)
    freqs = torch.concatenate([torch.tile(freqs_h[:, None, :], (1, W, 1)), torch.tile(freqs_w[None, :, :], (H, 1, 1))], axis=-1)  # (H, W, 2D)
    real = torch.cos(freqs).reshape(seq_len, dim)
    imag = torch.sin(freqs).reshape(seq_len, dim)
    return torch.complex(real, imag)


def apply_rotary_pos_emb(x, freqs_cis):
    # Convert last dimension to complex: (B, S, D) -> (B, S, D//2) where each element is complex
    x_float = x.to(torch.float32)
    x_complex = torch.view_as_complex(x_float.reshape(*x_float.shape[:-1], -1, 2).contiguous())
    
    freqs_cis = unsqueeze(unsqueeze(freqs_cis, 0), 2)
    T = freqs_cis.shape[1]
    
    # Only apply rotation to last T tokens (image patches), preserve prefix tokens
    x_rotated = x_complex.clone()
    x_rotated[:, -T:, :] = x_complex[:, -T:, :] * freqs_cis
    
    # Convert back to real representation
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.to(x.dtype)