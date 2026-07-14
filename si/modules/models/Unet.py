import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.layers.conv import SphereConv2d, DCUpsample, DCDownsample, nonlinearity
from modules.layers.positional_encoding import TimestepEmbedder


class ResBlock(nn.Module):
    """Residual block with SphereConv2d and timestep conditioning via AdaGN."""

    def __init__(self, in_channels, out_channels, t_emb_dim, dropout=0.1, num_groups=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6)
        self.conv1 = SphereConv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Timestep projection -> scale and shift for AdaGN
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels * 2),
        )

        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = SphereConv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Zero-init the last conv so residual block starts as identity
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        """
        Args:
            x: [b, c, h, w]
            t_emb: [b, t_emb_dim]
        """
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        # AdaGN: scale and shift after norm2
        scale_shift = self.t_proj(t_emb)[:, :, None, None]  # [b, 2*out_channels, 1, 1]
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention with flash attention (scaled_dot_product_attention)."""

    def __init__(self, channels, num_heads=8, num_groups=16):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Zero-init output projection
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Args:
            x: [b, c, h, w]
        Returns:
            [b, c, h, w]
        """
        b, c, h, w = x.shape
        nh = self.num_heads
        head_dim = c // nh

        qkv = self.qkv(self.norm(x))  # [b, 3*c, h, w]
        qkv = rearrange(qkv, 'b (three nh hd) h w -> three b nh (h w) hd', three=3, nh=nh, hd=head_dim)
        q, k, v = qkv.unbind(0)  # each [b, nh, h*w, head_dim]

        out = F.scaled_dot_product_attention(q, k, v)  # [b, nh, h*w, head_dim]
        out = rearrange(out, 'b nh (h w) hd -> b (nh hd) h w', h=h, w=w)

        return x + self.proj(out)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels=256,
        channel_mult=(1, 2, 4),
        num_res_blocks=3,
        attn_levels=(2),
        num_heads=8,
        dropout=0.0,
        t_emb_dim=256,
        num_groups=16,
    ):
        """
        Args:
            in_channels: channels of x_noised + cond concatenated
            out_channels: output prediction channels
            model_channels: base channel width
            channel_mult: per-level channel multipliers
            num_res_blocks: residual blocks per level
            attn_levels: which levels get self-attention (0-indexed)
            num_heads: attention heads
            dropout: dropout rate
            t_emb_dim: timestep embedding dimension
            num_groups: groups for GroupNorm
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_mult = channel_mult
        self.num_levels = len(channel_mult)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(t_emb_dim)

        # Input projection
        self.input_conv = SphereConv2d(in_channels, model_channels, kernel_size=(3, 3), padding=(1, 1))

        # ── Encoder ──
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = model_channels
        enc_channels = [ch]  # track channels for skip connections

        for level in range(self.num_levels):
            out_ch = model_channels * channel_mult[level]
            use_attn = level in attn_levels

            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                block_in = ch if i == 0 else out_ch
                level_blocks.append(ResBlock(block_in, out_ch, t_emb_dim, dropout, num_groups))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch, num_heads, num_groups))
                ch = out_ch
                enc_channels.append(ch)

            self.enc_blocks.append(level_blocks)

            # Downsample (except at the last level)
            if level < self.num_levels - 1:
                self.downsamples.append(DCDownsample(ch, ch))
                enc_channels.append(ch)
            else:
                self.downsamples.append(None)

        # ── Bottleneck ──
        self.mid_block1 = ResBlock(ch, ch, t_emb_dim, dropout, num_groups)
        self.mid_attn = AttentionBlock(ch, num_heads, num_groups)
        self.mid_block2 = ResBlock(ch, ch, t_emb_dim, dropout, num_groups)

        # ── Decoder ──
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(self.num_levels)):
            out_ch = model_channels * channel_mult[level]
            use_attn = level in attn_levels

            level_blocks = nn.ModuleList()
            # +1 res block in decoder for the skip connection at each level
            for i in range(num_res_blocks + 1):
                skip_ch = enc_channels.pop()
                block_in = ch + skip_ch
                level_blocks.append(ResBlock(block_in, out_ch, t_emb_dim, dropout, num_groups))
                if use_attn:
                    level_blocks.append(AttentionBlock(out_ch, num_heads, num_groups))
                ch = out_ch

            self.dec_blocks.append(level_blocks)

            # Upsample (except at level 0)
            if level > 0:
                self.upsamples.append(DCUpsample(ch, ch))
            else:
                self.upsamples.append(None)

        # Output
        self.out_norm = nn.GroupNorm(num_groups=num_groups, num_channels=ch, eps=1e-6)
        self.out_conv = SphereConv2d(ch, out_channels, kernel_size=(3, 3), padding=(1, 1))
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.initialize_weights()

    def initialize_weights(self):
        """Xavier-uniform for linear/conv layers, standard init for norms.
        Preserves the zero-inits already applied to output-critical layers."""
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, SphereConv2d)):
                # Skip layers that were deliberately zero-initialized
                if name.endswith(('out_conv', 'conv2', 'proj')):
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_noised, cond, t):
        """
        Args:
            x_noised: [b, c, h, w] — noised interpolant I_t
            cond: [b, c, h, w] — conditioning (current state / low-res)
            t: [b, 1] — diffusion timestep

        Returns:
            [b, out_channels, h, w] — predicted target
        """
        # Timestep embedding
        t_emb = self.t_embedder(t)  # [b, t_emb_dim]

        # Concatenate noised input and conditioning
        x = torch.cat([x_noised, cond], dim=1)
        x = self.input_conv(x)

        # ── Encoder ──
        skips = [x]
        for level in range(self.num_levels):
            for block in self.enc_blocks[level]:
                if isinstance(block, ResBlock):
                    x = block(x, t_emb)
                    skips.append(x)
                else:
                    x = block(x)

            if self.downsamples[level] is not None:
                x = self.downsamples[level](x)
                skips.append(x)

        # ── Bottleneck ──
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # ── Decoder ──
        for level_idx, level in enumerate(reversed(range(self.num_levels))):
            for block in self.dec_blocks[level_idx]:
                if isinstance(block, ResBlock):
                    skip = skips.pop()
                    x = torch.cat([x, skip], dim=1)
                    x = block(x, t_emb)
                else:
                    x = block(x)

            if self.upsamples[level_idx] is not None:
                x = self.upsamples[level_idx](x)

        # Output
        x = self.out_norm(x)
        x = nonlinearity(x)
        x = self.out_conv(x)

        return x
