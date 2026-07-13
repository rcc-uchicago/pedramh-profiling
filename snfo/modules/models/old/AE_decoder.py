import torch
import torch.nn as nn
from einops import rearrange
from modules.models.AE_simple import ResnetBlock, Upsample, DCUpsample, make_attn
from modules.layers.patchify import PatchEmbed

TYPE = "group"

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, 3D, or 4D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels, num_groups=16, type=TYPE):
    if type == "layer":
        return torch.nn.LayerNorm(in_channels, eps=1e-6)
    elif type == "group":
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        raise ValueError(f"unknown normalization type {type}")


class DecoderHistory(nn.Module):
    def __init__(self,
                 out_channels, # output channel dim
                 hidden_channels, # width of network
                 z_channels, # input latent dim
                 ch_mult=(1,2,4), 
                 num_res_blocks = 4,
                 resolution = (180, 360), 
                 attn_resolutions = [32], 
                 dropout=0.0, 
                 tanh_out=False,
                 dim=2,
                 padding_mode='zeros',
                 upsample_type = 'avg',
                 resamp_with_conv = True,
                 kernel_size=3,
                 padding=1,
                 num_out_blocks=0,
                 ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.tanh_out = tanh_out
        attn_type = "vanilla"
        self.dim = dim
        self.nsurface = 6
        self.ndiagnostic = 9
        self.nlevels = 26

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.hidden_channels*ch_mult[self.num_resolutions-1]

        curr_res = resolution[-1] // 2**(self.num_resolutions-1)

        # z to block_in
        self.conv_in = conv_nd(dim,
                                z_channels,
                                block_in//2,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding,
                                padding_mode=padding_mode)

        self.conv_in_history = PatchEmbed(patch_size=4,
                                        in_chans=z_channels,
                                        hidden_size=block_in//2,
                                        norm_layer=None,
                                        flatten=False)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode,
                                       padding=padding,
                                       kernel_size=kernel_size)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode,
                                       kernel_size=kernel_size,
                                       padding=padding,)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.hidden_channels*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout,
                                         dim=dim,
                                         padding_mode=padding_mode,
                                         padding=padding,
                                         kernel_size=kernel_size))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dim=dim))
                    print(f"added attn at res {curr_res}")
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if upsample_type == 'dc':
                    up.upsample = DCUpsample(block_in, 
                                            block_in)
                else:
                    up.upsample = Upsample(block_in, resamp_with_conv, dim=dim)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        self.num_out_blocks = num_out_blocks
        if num_out_blocks > 0:
            self.out_blocks = nn.ModuleList()
            for i in range(num_out_blocks):
                self.out_blocks.append(ResnetBlock(in_channels=block_in,
                                                out_channels=block_in,
                                                dropout=dropout,
                                                dim=dim,
                                                padding_mode=padding_mode,
                                                padding=padding,
                                                kernel_size=kernel_size))

        self.norm_out = Normalize(block_in) 
        self.conv_out = conv_nd(dim,
                        block_in,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        padding_mode=padding_mode)
        
        # Apply He Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Applies He (Kaiming) initialization to Conv2d and Linear layers.
        Initializes normalization layers (LayerNorm, BatchNorm) with scale 1 and bias 0.
        """
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def assemble_input(self, surface, multilevel, diagnostic):
        multilevel = rearrange(
            multilevel, "b l h w c -> b h w (l c)"
        )
        out = torch.cat((surface, diagnostic, multilevel), dim=-1) # b h w c
        out = rearrange(
            out, "b h w c -> b c h w"
        )

        return out
    
    def disassemble_input(self, x):
        x = rearrange(
            x, "b c h w -> b h w c"
        )

        surface = x[..., : self.nsurface]
        diagnostic = x[..., self.nsurface : self.nsurface + self.ndiagnostic]
        multilevel = x[..., self.nsurface + self.ndiagnostic :]

        multilevel = rearrange(
            multilevel,
            "b h w (l c) -> b l h w c",
            l=self.nlevels,
        )

        return surface, multilevel, diagnostic

    def forward(self, surface_history, multilevel_history, diagnostic_history,
                z_surface, z_history, z_diagnostic) -> torch.Tensor:

        # surface in shape b nlat nlon c 
        # multilevel in shape b nlevel nlat nlon c
        # diagnostic in shape b nlat nlon c

        z = self.assemble_input(z_surface, z_history, z_diagnostic) # b c zlat zlon
        history = self.assemble_input(surface_history, multilevel_history, diagnostic_history) # b c nlat nlon

        # z to block_in
        h = self.conv_in(z)
        history = self.conv_in_history(history, reshape=False) 

        h = torch.cat((h, history), dim=1)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.num_out_blocks > 0:
            for i in range(len(self.out_blocks)):
                h = self.out_blocks[i](h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w

        surface_out, multilevel_out, diagnostic_out = self.disassemble_input(h)

        return surface_out, multilevel_out, diagnostic_out
    

class ChannelLayerNorm(nn.Module):
    """
    Layer Normalization over third-last channel dimension.
    """

    def __init__(
        self, n_channels: int, eps: float = 1e-5, elementwise_affine: bool = False
    ):
        super(ChannelLayerNorm, self).__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))
        else:
            self.weight = None
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.constant_(self.weight, 1.0)
            torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(
                f"Expected at least 3D input with channel at dim=-3, got shape {tuple(x.shape)}"
            )
        if x.size(-3) != self.n_channels:
            raise ValueError(
                f"Channel dimension mismatch: got C={x.size(-3)}, expected {self.n_channels}"
            )

        # Compute per-pixel mean/var across channels without transposing
        mean = x.mean(dim=-3, keepdim=True)
        var = x.var(dim=-3, keepdim=True, unbiased=False)
        inv_std = torch.rsqrt(var + self.eps)
        y = (x - mean) * inv_std

        if self.weight is not None and self.bias is not None:
            # Broadcast [C] over [N, C, *spatial]
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = y * self.weight.view(*shape) + self.bias.view(*shape)
        return y


class ConditionalLayerNorm(nn.Module):
    """
    Conditional Layer Normalization as described in "AdaSpeech: Adaptive
    Text to Speech for Custom Voice" https://arxiv.org/abs/2103.00993.

    Assumes that the input has shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        embed_dim: int,
        n_channels: int,
        epsilon: float = 1e-5,
        elementwise_affine: bool = False,
    ):
        super(ConditionalLayerNorm, self).__init__()
        self.n_channels = n_channels
        self.epsilon = epsilon

        self.W_scale: nn.Linear | None = nn.Linear(
            embed_dim, self.n_channels
        )
        self.W_bias: nn.Linear | None = nn.Linear(
            embed_dim, self.n_channels
        )

        self.norm = ChannelLayerNorm(
            self.n_channels,
            eps=epsilon,
            elementwise_affine=elementwise_affine,
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)

        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        context
    ) -> torch.Tensor:
        """
        Conditional Layer Normalization

        This is a modified version of LayerNorm that allows the scale and bias to be
        conditioned on a context embedding.

        Args:
            x: The input tensor to normalize, of shape
                (batch_size, channels, height, width).
            context: The context to condition on.

        Returns:
            The normalized tensor, of shape (batch_size, channels, height, width).
        """

        scale: torch.Tensor = (
            self.W_scale(context).unsqueeze(-1).unsqueeze(-1)
        )

        bias: torch.Tensor = (
            self.W_bias(context).unsqueeze(-1).unsqueeze(-1)
        )

        x_norm: torch.Tensor = self.norm(x)
        return_value = x_norm * scale + bias
        return return_value
    
class CondResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, cond_channels=None, conv_shortcut=False,
                 dropout=0, dim=2, padding_mode='zeros', kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        cond_channels = in_channels if cond_channels is None else cond_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.dim = dim

        self.norm1 = ConditionalLayerNorm(cond_channels, in_channels, elementwise_affine=True)
        self.conv1 = conv_nd(dim,
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                            padding_mode=padding_mode)

        self.norm2 = ConditionalLayerNorm(cond_channels, out_channels, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_nd(dim,
                            out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                            padding_mode=padding_mode)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_nd(dim,
                                            in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding,
                                            padding_mode=padding_mode)
            else:
                self.nin_shortcut = conv_nd(dim,
                                            in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)

    def forward(self, x, c):

        h = x
        h = self.norm1(h, c)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h, c)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class StochasticDecoderHistory(nn.Module):
    """
    Decoder with stochastic forcing via conditional layer norm (FGN-style).
    Mirrors DecoderHistory but replaces ResnetBlock with CondResnetBlock,
    injecting a low-dimensional Gaussian noise vector via adaptive layer norm.
    """
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 z_channels,
                 noise_dim=64,
                 ch_mult=(1, 2, 2),
                 num_res_blocks=4,
                 resolution=(180, 360),
                 attn_resolutions=[32],
                 dropout=0.0,
                 tanh_out=False,
                 dim=2,
                 padding_mode='zeros',
                 upsample_type='avg',
                 resamp_with_conv=True,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.tanh_out = tanh_out
        attn_type = "vanilla"
        self.dim = dim
        self.nsurface = 6
        self.ndiagnostic = 9
        self.nlevels = 26
        self.noise_dim = noise_dim

        # Noise embedding: maps z ~ N(0,I) to conditioning dimension
        block_in = self.hidden_channels * ch_mult[self.num_resolutions - 1]
        self.noise_embed = nn.Sequential(
            nn.Linear(noise_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        curr_res = resolution[-1] // 2 ** (self.num_resolutions - 1)

        # z to block_in
        self.conv_in = conv_nd(dim,
                               z_channels,
                               block_in // 2,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding,
                               padding_mode=padding_mode)

        self.conv_in_history = PatchEmbed(patch_size=4,
                                          in_chans=z_channels,
                                          hidden_size=block_in // 2,
                                          norm_layer=None,
                                          flatten=False)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CondResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           cond_channels=hidden_channels,
                                           dropout=dropout,
                                           dim=dim,
                                           padding_mode=padding_mode,
                                           padding=padding,
                                           kernel_size=kernel_size)
        self.mid.block_2 = CondResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           cond_channels=hidden_channels,
                                           dropout=dropout,
                                           dim=dim,
                                           padding_mode=padding_mode,
                                           kernel_size=kernel_size,
                                           padding=padding)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.hidden_channels * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(CondResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             cond_channels=hidden_channels,
                                             dropout=dropout,
                                             dim=dim,
                                             padding_mode=padding_mode,
                                             padding=padding,
                                             kernel_size=kernel_size))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dim=dim))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if upsample_type == 'dc':
                    up.upsample = DCUpsample(block_in, block_in)
                else:
                    up.upsample = Upsample(block_in, resamp_with_conv, dim=dim)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = conv_nd(dim,
                                block_in,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding,
                                padding_mode=padding_mode)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def assemble_input(self, surface, multilevel, diagnostic):
        multilevel = rearrange(multilevel, "b l h w c -> b h w (l c)")
        out = torch.cat((surface, diagnostic, multilevel), dim=-1)
        out = rearrange(out, "b h w c -> b c h w")
        return out

    def disassemble_input(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        surface = x[..., :self.nsurface]
        diagnostic = x[..., self.nsurface:self.nsurface + self.ndiagnostic]
        multilevel = x[..., self.nsurface + self.ndiagnostic:]
        multilevel = rearrange(multilevel, "b h w (l c) -> b l h w c", l=self.nlevels)
        return surface, multilevel, diagnostic

    def forward(self, surface_history, multilevel_history, diagnostic_history,
                z_surface, z_history, z_diagnostic, noise=None):
        """
        Args:
            surface_history, multilevel_history, diagnostic_history: full-res history
            z_surface, z_history, z_diagnostic: bilinearly downsampled (4x) inputs
            noise: (B, noise_dim) Gaussian noise vector. If None, sampled internally.
        """
        z = self.assemble_input(z_surface, z_history, z_diagnostic)
        history = self.assemble_input(surface_history, multilevel_history, diagnostic_history)

        # Sample noise if not provided
        if noise is None:
            noise = torch.randn(z.shape[0], self.noise_dim, device=z.device, dtype=z.dtype)

        # Noise conditioning vector (shared across all CondResnetBlocks)
        c = self.noise_embed(noise)  # (B, hidden_channels)

        # z to block_in
        h = self.conv_in(z)
        history = self.conv_in_history(history, reshape=False)
        h = torch.cat((h, history), dim=1)

        # middle
        h = self.mid.block_1(h, c)
        h = self.mid.block_2(h, c)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, c)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        surface_out, multilevel_out, diagnostic_out = self.disassemble_input(h)
        return surface_out, multilevel_out, diagnostic_out
