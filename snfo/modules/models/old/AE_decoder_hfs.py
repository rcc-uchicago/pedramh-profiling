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

class featscale2(nn.Module):
    def __init__(self, patch_size,channels):
        super(featscale2, self).__init__()
        self.patch_size = patch_size
        self.lambda1 = nn.Parameter(torch.ones(1,channels,1,1,1))
        self.lambda2 = nn.Parameter(torch.ones(1,channels,1,1,1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Reshape into patches of shape (batch_size, channels, num_patches, patch_size, patch_size)
        X_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        num_patches = (height//self.patch_size)* (width//self.patch_size)
        X_patches = X_patches.reshape(batch_size, channels, num_patches, self.patch_size, self.patch_size)
        X_mean_patch = X_patches.mean(dim=2)
        X_mean_expanded = X_mean_patch.unsqueeze(2).expand(-1, -1, num_patches, -1, -1)
        
        #Generate X_d and X_h
        X_d = X_mean_expanded
        X_h = X_patches - X_d

        # #Combine X_d and X_h
        X = X_patches + self.lambda1*X_d + self.lambda2*X_h
        X = X.reshape(batch_size, channels, height//self.patch_size, width//self.patch_size, self.patch_size, self.patch_size)
        X = X.permute(0,1,2,4,3,5).reshape(batch_size, channels, height, width)
        return X

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
        init_patch_size=1

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
        self.fs_bottleneck = featscale2(patch_size=init_patch_size, channels=block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout,
                                       dim=dim,
                                       padding_mode=padding_mode,
                                       kernel_size=kernel_size,
                                       padding=padding,)
        self.fs_bottleneck2 = featscale2(patch_size=init_patch_size, channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        self.feat_up = nn.ModuleList()
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
                init_patch_size = 2*init_patch_size
            self.up.insert(0, up) # prepend to get consistent order
            self.feat_up.insert(0, featscale2(patch_size=init_patch_size, channels=block_in))

        self.num_out_blocks = num_out_blocks
        if num_out_blocks > 0:
            self.out_blocks = nn.ModuleList()
            self.out_scale = nn.ModuleList()
            for i in range(num_out_blocks):
                self.out_blocks.append(ResnetBlock(in_channels=block_in,
                                                out_channels=block_in,
                                                dropout=dropout,
                                                dim=dim,
                                                padding_mode=padding_mode,
                                                padding=padding,
                                                kernel_size=kernel_size))
                self.out_scale.append(featscale2(patch_size=init_patch_size, channels=block_in))

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
        h = self.fs_bottleneck(h)
        h = self.mid.block_2(h)
        h = self.fs_bottleneck2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            h = self.feat_up[i_level](h)

        if self.num_out_blocks > 0:
            for i in range(len(self.out_blocks)):
                h = self.out_blocks[i](h)
                h = self.out_scale[i](h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w

        surface_out, multilevel_out, diagnostic_out = self.disassemble_input(h)

        return surface_out, multilevel_out, diagnostic_out
