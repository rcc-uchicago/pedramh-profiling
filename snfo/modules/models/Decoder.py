import torch
import torch.nn as nn
from modules.layers.conv import ResnetBlock, SphereConv2d, DCUpsample

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels, num_groups=16, type="group"):
    if type == "layer":
        return torch.nn.LayerNorm(in_channels, eps=1e-6)
    elif type == "group":
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        raise ValueError(f"unknown normalization type {type}")
    
class DecoderCNN(nn.Module):
    def __init__(self,
                 out_channels, # output channel dim
                 hidden_channels, # width of network
                 z_channels, # input latent dim
                 ch_mult=(1,2,4), 
                 num_res_blocks = 4,
                 num_out_blocks=0,
                 dropout = 0.1
                 ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        init_patch_size = 1

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.hidden_channels*ch_mult[self.num_resolutions-1]

        self.conv_in = SphereConv2d(in_channels=z_channels,
                                    out_channels=block_in,
                                    kernel_size=(3, 3), padding = (1, 1))

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = self.hidden_channels*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                       out_channels=block_out,
                                       dropout=dropout))
                block_in = block_out

            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = DCUpsample(block_in, 
                                            block_in)
                init_patch_size = 2*init_patch_size
            self.up.insert(0, up) # prepend to get consistent order

        self.num_out_blocks = num_out_blocks
        if num_out_blocks > 0:
            self.out_blocks = nn.ModuleList()
            for i in range(num_out_blocks):
                self.out_blocks.append(ResnetBlock(in_channels=block_in,
                                       out_channels=block_out,
                                       dropout=dropout))

        self.norm_out = Normalize(block_in) 
        self.conv_out = SphereConv2d(in_channels=block_in,
                                    out_channels=out_channels,
                                    kernel_size=(3, 3), padding = (1, 1))
        
        # Apply He Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Applies He (Kaiming) initialization to Conv2d and Linear layers.
        Initializes normalization layers (LayerNorm, BatchNorm) with scale 1 and bias 0.
        """
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear, SphereConv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z) -> torch.Tensor:

        # x in shape b c nlat nlon

        # z to block_in
        h = self.conv_in(z)
        
        # middle
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.num_out_blocks > 0:
            for i in range(len(self.out_blocks)):
                h = self.out_blocks[i](h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w

        return h
