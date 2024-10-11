import torch
from torch import nn


class PatchRecovery2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]


class PatchRecovery3D(nn.Module):
    """
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]
    

# borrowed from 
#https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

class SubPixelConvICNR_2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        assert patch_size[0] == patch_size[1], 'mismatch'
        self.pad_zero = nn.ZeroPad2d((0, 0, 1, 1))
        self.pad_circular = nn.CircularPad2d((1, 1, 0, 0))
        self.conv = nn.Conv2d(in_chans, out_chans*patch_size[0]**2, kernel_size=3, stride=1, padding=0, bias=0)
        self.pixelshuffle = nn.PixelShuffle(patch_size[0])
        weight = ICNR(self.conv.weight, 
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[0])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x):
        x_padded = self.pad_zero(self.pad_circular(x))
        output = self.conv(x_padded)
        #print(output.shape)
        
        output = self.pixelshuffle(output)
        #print(output.shape)
        
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]


class SubPixelConvICNR_3D(nn.Module):
    """
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans, padded_front = False):
        super().__init__()
        self.img_size = img_size
        self.padded_front = padded_front
        assert patch_size[1] == patch_size[2], 'mismatch'
        self.pad_zero = nn.ZeroPad3d((0, 0, 1, 1, 0, 0))
        self.pad_circular = nn.CircularPad3d((1, 1, 0, 0, 0, 0))
        self.conv = nn.Conv2d(in_chans//2, out_chans*patch_size[1]**2, kernel_size=3, stride=1, padding=0, bias=0)
        self.pixelshuffle = nn.PixelShuffle(patch_size[1])
        weight = ICNR(self.conv.weight, 
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[1])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x: torch.Tensor):
        # first, split in dimension
        # print(x.shape)
        x_padded = self.pad_zero(self.pad_circular(x))
        x_padded = x_padded.reshape(x_padded.shape[0], x_padded.shape[1]//2, 2, *x_padded.shape[2:])
        x_padded = x_padded.flatten(2, 3)
        if not self.padded_front:
            x_padded = x_padded[:, :, 0:self.img_size[0]]
        else:
            x_padded = x_padded[:, :, 1:self.img_size[0]+1] # to make 13 vertical dims
        x_padded = x_padded.movedim(-3, 1).flatten(0, 1)
        output = self.conv(x_padded)
        output = self.pixelshuffle(output)
        output = output.reshape(-1, self.img_size[0], *output.shape[1:]).movedim(1, -3)
        
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]
    
class PatchRecovery5(nn.Module):
    ''' true upsampling with 3D conv
    '''
    def __init__(self, 
                 input_dim=None,
                 dim=192,
                 downfactor=4,
                 hidden_dim=96,
                 output_dim=69,
                 n_level_variables=5):
        # input dim equals input_dim*z since we will be flattening stuff ?
        super().__init__()
        self.downfactor = downfactor
        if input_dim is None:
            input_dim = 8*dim
        
        self.input_conv = nn.Conv2d(input_dim, 14*hidden_dim, kernel_size=1, stride=1, padding=0)
        self.interp = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)

        self.head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1), # kernel size 3 for interactions and smoothing
            nn.GELU(),
        )
        if downfactor == 4:
            self.head2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6, affine=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1), # kernel size 3 for interactions and smoothing
            nn.GELU())
        
        self.proj_surface = nn.Conv2d(hidden_dim, 4, kernel_size=1, stride=1, padding=0)
        self.proj_level = nn.Conv3d(hidden_dim, n_level_variables, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # recover enough levels
        bs = x.shape[0]
        x = x.flatten(1, 2) # put levels in the channel dim
        x = self.input_conv(x)
        x = x.reshape((bs, 14, -1, *x.shape[-2:])).flatten(0, 1) # put levels back
        x = self.interp(x)
        x = x.reshape(bs, 14, -1, *x.shape[-2:]).movedim(1, 2)
        x = self.head(x)
        if self.downfactor == 4:
            x = x.reshape((bs, 14, -1, *x.shape[-2:])).flatten(0, 1) # put levels back
            x = self.interp(x)
            x = x.reshape(bs, 14, -1, *x.shape[-2:]).movedim(1, 2)
            x = self.head2(x)

        output_surface = self.proj_surface(x[:, :, 0])
        output_level = self.proj_level(x[:, :, 1:])

        return output_level, output_surface.unsqueeze(-3)
