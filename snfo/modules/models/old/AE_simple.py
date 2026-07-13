from typing import Any
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

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


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=2):
        super().__init__()
        self.with_conv = with_conv
        self.dim = dim
        if self.with_conv:
            self.conv = conv_nd(dim,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, x):
        if len(x.shape) == 6: # interpolate doesn't support 6D
            x = torch.kron(x, torch.ones(2, 2, 2, 2, device=x.device))  # upsample w/ kronecker product
        elif self.dim == 3:
            x = torch.nn.functional.interpolate(x, scale_factor=(1,2,2), mode="trilinear") # do not upsample levels
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=2):
        super().__init__()
        self.with_conv = with_conv
        self.dim=dim
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = conv_nd(dim,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=0)

    def forward(self, x):
        if self.with_conv:
            if self.dim == 3:
                pad = (0,1,0,1,0,1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                x = self.conv(x)
            elif self.dim == 4:
                pad = (0,1,0,1,0,1,0,1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                x = self.conv(x)
            else:
                pad = (0,1,0,1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                x = self.conv(x)
        else:
            if self.dim == 3:
                x = torch.nn.functional.avg_pool3d(x, kernel_size=(1, 2, 2)) # do not downsample levels
            else:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
class DCUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(
                hidden_states, scale_factor=self.factor, mode=self.interpolation_mode
            )
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states
    
class DCDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = in_channels * self.factor**2 // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, dim=2, padding_mode='zeros', kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.dim = dim

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_nd(dim,
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                            padding_mode=padding_mode)

        self.norm2 = Normalize(out_channels)
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

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels, dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim

        self.norm = Normalize(in_channels)
        self.q = conv_nd(dim,
                        in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.k = conv_nd(dim,
                        in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.v = conv_nd(dim,
                        in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        self.proj_out = conv_nd(dim,
                                in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        if self.dim == 4:
            b, c, t, d, h, w = q.shape
            num_tokens = h*w*d*t
        elif self.dim == 3:
            b,c,d,h,w = q.shape
            num_tokens = h*w*d
        else:
            b,c,h,w = q.shape
            num_tokens = h*w

        q = q.reshape(b,c,num_tokens)
        q = q.permute(0,2,1)   # b,hwd,c
        k = k.reshape(b,c,num_tokens) # b,c,hwd
        w_ = torch.bmm(q,k)     # b,hwd,hwd    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,num_tokens)
        w_ = w_.permute(0,2,1)   # b,hwd,hwd (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hwd (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        if self.dim == 4:
            h_ = h_.reshape(b,c,t,d,h,w)
        elif self.dim == 3: 
            h_ = h_.reshape(b,c,d,h,w)
        else:
            h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla", dim=2):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    if attn_type == "vanilla":
        return AttnBlock(in_channels, dim=dim)
    elif attn_type == "none":
        return nn.Identity(in_channels)

class Encoder(nn.Module):
    def __init__(self,
                 in_channels, # input channel dim
                 hidden_channels, # width of network
                 z_channels, # output latent dim
                 ch_mult=(1,2,4), 
                 num_res_blocks = 2,
                 resolution = (180, 360), 
                 attn_resolutions = [32], 
                 dropout=0.0, 
                 double_z=False, 
                 tanh_out=False,
                 dim=2,
                 padding_mode='zeros',
                 downsample_type = 'avg',
                 use_attn=False,
                 saturate=True,
                 resamp_with_conv = True,
                 separate_embedders=False,
                 kernel_size=3,
                 padding=1):
        
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        attn_type = "vanilla"
        self.tanh_out = tanh_out
        self.dim = dim 
        self.ndiagnostic = 9
        self.nlevels = 26
        self.nsurface = 6

        if separate_embedders:
            self.conv_surface = conv_nd(2,
                                    6,
                                    self.hidden_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode=padding_mode)
            self.conv_diag = conv_nd(2,
                                    9,
                                    self.hidden_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode=padding_mode)
            self.conv_multilevel = conv_nd(dim,
                                    9,
                                    self.hidden_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    padding_mode=padding_mode)
        else:
            self.conv_in = conv_nd(dim,
                                    in_channels,
                                    self.hidden_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    padding_mode=padding_mode)

        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)

        curr_res = resolution[-1]
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.hidden_channels*in_ch_mult[i_level]
            block_out = self.hidden_channels*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout,
                                         dim=dim,
                                         padding_mode=padding_mode,
                                         kernel_size=kernel_size,
                                         padding=padding))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dim=dim))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                if downsample_type == 'dc':
                    down.downsample = DCDownsample(block_in, 
                                                  block_in)
                else:
                    down.downsample = Downsample(block_in, resamp_with_conv, dim=dim)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode,
                                       kernel_size=kernel_size,
                                       padding=padding)
        if use_attn:
            self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dim=dim)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode,
                                       kernel_size=kernel_size,
                                       padding=padding)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_nd(dim,
                                block_in,
                                2*z_channels if double_z else z_channels,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding,
                                padding_mode=padding_mode)

        # Apply He Initialization
        self.apply(self._init_weights)
        self.saturate_latent = saturate

    def _init_weights(self, m):
        """
        Applies He (Kaiming) initialization to Conv2d and Linear layers.
        Initializes normalization layers (LayerNorm, BatchNorm) with scale 1 and bias 0.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def saturate(self, x, B=5.0):
        x = x /torch.sqrt(1 + x**2/B**2)
        return x

    def forward(self, surface, multilevel, diagnostic=None) -> torch.Tensor:
        # surface in shape b nlat nlon c 
        # multilevel in shape b nlevel nlat nlon c
        # diagnostic in shape b nlat nlon c

        surface = rearrange(surface, 'b nlat nlon c -> b c nlat nlon')
        # flatten levels to channels. This is because we are purely compressing in lat/lon dimensions
        multilevel = rearrange(multilevel, 'b nlevel nlat nlon c -> b (c nlevel) nlat nlon')

        if diagnostic is not None:
            diagnostic = rearrange(diagnostic, 'b nlat nlon c -> b c nlat nlon')
            x = torch.cat([surface, diagnostic, multilevel], dim=1) # b c nlat nlon
        else:
            x = torch.cat([surface, multilevel], dim=1) # b c nlat nlon

        # downsampling  
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w

        if self.tanh_out:
            h = torch.tanh(h) # clamp between -1 and 1. 

        if self.saturate_latent:
            h = self.saturate(h, B=5.0)

        if diagnostic is not None:
            z_surface = h[:, :self.nsurface, :, :] # b n_surface zlat zlon
            z_diagnostic = h[:, self.nsurface:self.nsurface + self.ndiagnostic, :, :] # b n_diagnostic zlat zlon
            z_multilevel = h[:, self.nsurface + self.ndiagnostic:, :, :] # b (n_multilevel * nlevel) zlat zlon
            z_diagnostic = rearrange(z_diagnostic, 'b c zlat zlon -> b zlat zlon c')
        else:
            z_surface = h[:, :self.nsurface, :, :] # b n_surface zlat zlon
            z_multilevel = h[:, self.nsurface:, :, :] # b (n_multilevel * nlevel) zlat zlon
            z_diagnostic = None

        z_multilevel = rearrange(z_multilevel, 'b (c nlevel) zlat zlon -> b nlevel zlat zlon c', nlevel=self.nlevels)
        z_surface = rearrange(z_surface, 'b c zlat zlon -> b zlat zlon c')

        return z_surface, z_multilevel,  z_diagnostic
    
class Encoder3D(Encoder):
    def __init__(self,
                 in_channels, # input channel dim
                 hidden_channels, # width of network
                 z_channels, # output latent dim
                 ch_mult=(1,2,4), 
                 num_res_blocks = 2,
                 resolution = (180, 360, 30), 
                 attn_resolutions = [32], 
                 dropout=0.0, 
                 double_z=False, 
                 tanh_out=False,
                 dim=3,
                 padding_mode='zeros',
                 downsample_type = 'avg',
                 use_attn=False,
                 saturate=True,
                 resamp_with_conv = False,
                 kernel_size=3,
                 padding=1):
        
        super().__init__(in_channels,
                         hidden_channels,
                         z_channels,
                         ch_mult,
                         num_res_blocks,
                         resolution,
                         attn_resolutions,
                         dropout,
                         double_z,
                         tanh_out,
                         dim,
                         padding_mode,
                         downsample_type,
                         use_attn,
                         saturate,
                         resamp_with_conv,
                         separate_embedders=True,
                         kernel_size=kernel_size,
                         padding=padding)
        

    def forward(self, surface, multilevel, diagnostic) -> torch.Tensor:
        # surface in shape b nlat nlon c 
        # multilevel in shape b nlevel nlat nlon c
        # diagnostic in shape b nlat nlon c
        n_levels = multilevel.shape[1]

        surface = rearrange(surface, 'b nlat nlon c -> b c nlat nlon')
        diagnostic = rearrange(diagnostic, 'b nlat nlon c -> b c nlat nlon')
        # flatten levels to channels. This is because we are purely compressing in lat/lon dimensions
        multilevel = rearrange(multilevel, 'b nlevel nlat nlon c -> b c nlevel nlat nlon')

        surface = self.conv_surface(surface).unsqueeze(2) # b hidden 1 nlat nlon
        diagnostic = self.conv_diag(diagnostic).unsqueeze(2) # b hidden 1 nlat nlon
        multilevel = self.conv_multilevel(multilevel) # b hidden nlevel nlat nlon

        x = torch.cat([surface, diagnostic, multilevel], dim=2) # b hidden (nlevel + 2) nlat nlon

        # downsampling  
        hs = [x]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w

        if self.tanh_out:
            h = torch.tanh(h) # clamp between -1 and 1. 

        if self.saturate_latent:
            h = self.saturate(h, B=5.0)

        z_surface = h[:, :, 0] # b hidden zlat zlon
        z_diagnostic = h[:, :, 1] # b hidden zlat zlon
        z_multilevel = h[:, :, 2:] # b hidden nlevel zlat zlon

        z_multilevel = rearrange(z_multilevel, 'b c nlevel zlat zlon -> b nlevel zlat zlon c', nlevel=n_levels)
        z_surface = rearrange(z_surface, 'b c zlat zlon -> b zlat zlon c')
        z_diagnostic = rearrange(z_diagnostic, 'b c zlat zlon -> b zlat zlon c')

        return z_surface, z_multilevel,  z_diagnostic
      

class Decoder(nn.Module):
    def __init__(self,
                 out_channels, # output channel dim
                 hidden_channels, # width of network
                 z_channels, # input latent dim
                 ch_mult=(1,2,4), 
                 num_res_blocks = 2,
                 resolution = (180, 360), 
                 attn_resolutions = [32], 
                 dropout=0.0, 
                 double_z=True, 
                 tanh_out=False,
                 dim=2,
                 padding_mode='zeros',
                 upsample_type = 'avg',
                 use_attn=False,
                 resamp_with_conv = True,
                 separate_embedders=False,
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

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.hidden_channels*ch_mult[self.num_resolutions-1]

        curr_res = resolution[-1] // 2**(self.num_resolutions-1)

        # z to block_in
        self.conv_in = conv_nd(dim,
                                z_channels,
                                block_in,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding,
                                padding_mode=padding_mode)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode,
                                       padding=padding,
                                       kernel_size=kernel_size)
        if use_attn:
            self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dim=dim)
        else:
            self.mid.attn_1 = nn.Identity()
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

        # end
        self.norm_out = Normalize(block_in)
        if separate_embedders:
            self.surface_out = conv_nd(2,
                                       block_in,
                                    out_channels=6,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode=padding_mode)
            self.diagnostic_out = conv_nd(2,
                                       block_in,
                                    out_channels=9,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode=padding_mode)
            self.multilevel_out = conv_nd(dim,
                                    block_in,
                                    out_channels=9,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    padding_mode=padding_mode)
        else:
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

    def forward(self, surface, multilevel, diagnostic=None) -> torch.Tensor:
        # surface in shape b nlat nlon c 
        # multilevel in shape b nlevel nlat nlon c
        # diagnostic in shape b nlat nlon c

        surface = rearrange(surface, 'b nlat nlon c -> b c nlat nlon')
        # flatten levels to channels. This is because we are purely compressing in lat/lon dimensions
        multilevel = rearrange(multilevel, 'b nlevel nlat nlon c -> b (c nlevel) nlat nlon')

        if diagnostic is not None:
            diagnostic = rearrange(diagnostic, 'b nlat nlon c -> b c nlat nlon')
            z = torch.cat([surface, diagnostic, multilevel], dim=1) # b c nlat nlon
        else:
            z = torch.cat([surface, multilevel], dim=1) # b c nlat nlon

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w

        if self.tanh_out:
            h = torch.tanh(h)

        z_surface = h[:, :self.nsurface, :, :] # b n_surface zlat zlon
        z_diagnostic = h[:, self.nsurface:self.nsurface + self.ndiagnostic, :, :] # b n_diagnostic zlat zlon
        z_multilevel = h[:, self.nsurface + self.ndiagnostic:, :, :] # b (n_multilevel * nlevel) zlat zlon

        z_multilevel = rearrange(z_multilevel, 'b (c nlevel) zlat zlon -> b nlevel zlat zlon c', nlevel=self.nlevels)
        z_surface = rearrange(z_surface, 'b c zlat zlon -> b zlat zlon c')
        z_diagnostic = rearrange(z_diagnostic, 'b c zlat zlon -> b zlat zlon c')

        return z_surface, z_multilevel,  z_diagnostic
    

class Decoder3D(Decoder):
    def __init__(self,
                 out_channels, # output channel dim
                 hidden_channels, # width of network
                 z_channels, # input latent dim
                 ch_mult=(1,2,4), 
                 num_res_blocks = 2,
                 resolution = (180, 360, 30), 
                 attn_resolutions = [32], 
                 dropout=0.0, 
                 double_z=True, 
                 tanh_out=False,
                 dim=3,
                 padding_mode='zeros',
                 upsample_type = 'avg',
                 use_attn=False,
                 resamp_with_conv = False,
                 kernel_size=3,
                 padding=1):
        super().__init__(out_channels,
                         hidden_channels,
                         z_channels,
                         ch_mult,
                         num_res_blocks,
                         resolution,
                         attn_resolutions,
                         dropout,
                         double_z,
                         tanh_out,
                         dim,
                         padding_mode,
                         upsample_type,
                         use_attn,
                         resamp_with_conv,
                         separate_embedders=True,
                         kernel_size=kernel_size,
                         padding=padding)
        
    def forward(self, surface, multilevel, diagnostic) -> torch.Tensor:
        # surface in shape b nlat nlon c 
        # multilevel in shape b nlevel nlat nlon c
        # diagnostic in shape b nlat nlon c
        
        n_surface = surface.shape[-1]
        n_diagnostic = diagnostic.shape[-1]
        n_levels = multilevel.shape[1]

        surface = rearrange(surface, 'b nlat nlon c -> b c nlat nlon').unsqueeze(2) # b c 1 nlat nlon
        diagnostic = rearrange(diagnostic, 'b nlat nlon c -> b c nlat nlon').unsqueeze(2) # b c 1 nlat nlon
        multilevel = rearrange(multilevel, 'b nlevel nlat nlon c -> b c nlevel nlat nlon')

        z = torch.cat([surface, diagnostic, multilevel], dim=2) # b c (nlevel+2) nlat nlon

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)

        z_surface = h[:, :, 0] # b c zlat zlon
        z_diagnostic = h[:, :, 1] # b c zlat zlon
        z_multilevel = h[:, :, 2:] # b c nlevel zlat zlon

        z_surface = self.surface_out(z_surface) # b n_surface zlat zlon
        z_diagnostic = self.diagnostic_out(z_diagnostic)
        z_multilevel = self.multilevel_out(z_multilevel)

        z_multilevel = rearrange(z_multilevel, 'b c nlevel zlat zlon -> b nlevel zlat zlon c', nlevel=n_levels)
        z_surface = rearrange(z_surface, 'b c zlat zlon -> b zlat zlon c')
        z_diagnostic = rearrange(z_diagnostic, 'b c zlat zlon -> b zlat zlon c')

        return z_surface, z_multilevel,  z_diagnostic

class DecoderHistory(nn.Module):
    def __init__(self,
                 out_channels, # output channel dim
                 hidden_channels, # width of network
                 z_channels, # input latent dim
                 z_channels_2,
                 ch_mult=(1,2,2), 
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
                 num_out_blocks=8,
                 depthwise_out=False
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
                                block_in,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding,
                                padding_mode=padding_mode)

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
        
        self.conv_in_history = conv_nd(dim,
                        z_channels_2,
                        block_in,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        padding_mode=padding_mode)

        self.out_blocks = nn.ModuleList()
        for _ in range(num_out_blocks):
            self.out_blocks.append(ResnetBlock(in_channels=block_in*2,
                                                  out_channels=block_in*2,
                                                  dropout=dropout,
                                                  dim=dim,
                                                  padding_mode=padding_mode,
                                                  padding=padding,
                                                  kernel_size=kernel_size,))

        self.norm_out = Normalize(block_in*2) 
        self.conv_out = conv_nd(dim,
                        block_in*2,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        padding_mode=padding_mode)
        
        self.depthwise_out = depthwise_out
        if depthwise_out:
            self.depth_out = conv_nd(dim,
                                    out_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    padding_mode=padding_mode,
                                    groups = out_channels)

            
        
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
        history = self.conv_in_history(history) 

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

        
        h = torch.cat([h, history], dim=1)

        for block in self.out_blocks:
            h = block(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w

        if self.depthwise_out:
            h = self.depth_out(h)

        surface_out, multilevel_out, diagnostic_out = self.disassemble_input(h)

        return surface_out, multilevel_out, diagnostic_out