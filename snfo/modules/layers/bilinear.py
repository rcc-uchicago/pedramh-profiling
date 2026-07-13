from typing import Any
import torch
from einops import rearrange
import torch.nn.functional as F
from common.utils import assemble_input

class BilinearDownsample():
    def __init__(self,
                 downsample_factor = 4):
        super().__init__()
        self.downsample_factor = downsample_factor

    def __call__(self, x) -> Any:
        return self.forward(x)
    
    def forward(self, x) -> torch.Tensor:

        x = F.interpolate(x, scale_factor=1/self.downsample_factor, mode='bilinear', align_corners=False)

        return x 
    
class BilinearEncoder():
    def __init__(self,
                 downsample_factor = 4):
        super().__init__()
        self.downsample_factor = downsample_factor

    def __call__(self, surface, multilevel, diagnostic, assemble=False) -> Any:

        z_surface, z_multilevel, z_diagnostic = self.forward(surface, multilevel, diagnostic)

        if assemble:
            z = assemble_input(z_surface, z_multilevel, z_diagnostic)
            return z
        else:
            return z_surface, z_multilevel, z_diagnostic
    
    def forward(self, surface, multilevel, diagnostic=None) -> torch.Tensor:
        # surface in shape b c nlat nlon
        # multilevel in shape b c nlevel nlat nlon
        # diagnostic in shape b c nlat nlon

        nlevels = multilevel.shape[2]

        multilevel = rearrange(multilevel, 'b c nlevel nlat nlon -> b (c nlevel) nlat nlon')

        surface = F.interpolate(surface, scale_factor=1/self.downsample_factor, mode='bilinear', align_corners=False)
        multilevel = F.interpolate(multilevel, scale_factor=1/self.downsample_factor, mode='bilinear', align_corners=False)

        multilevel = rearrange(multilevel, 'b (c nlevel) zlat zlon -> b c nlevel zlat zlon', nlevel=nlevels)

        if diagnostic is not None:
            diagnostic = F.interpolate(diagnostic, scale_factor=1/self.downsample_factor, mode='bilinear', align_corners=False)

        return surface, multilevel, diagnostic
    
class BilinearDecoder():
    def __init__(self,
                 downsample_factor = 4):
        super().__init__()
        self.downsample_factor = downsample_factor

    def __call__(self, surface, multilevel, diagnostic) -> Any:
        return self.forward(surface, multilevel, diagnostic)
    
    def forward(self, surface, multilevel, diagnostic=None) -> torch.Tensor:
        # surface in shape b c nlat nlon
        # multilevel in shape b c nlevel nlat nlon
        # diagnostic in shape b c nlat nlon
        nlevels = multilevel.shape[2]

        multilevel = rearrange(multilevel, 'b c nlevel nlat nlon -> b (c nlevel) nlat nlon')

        surface = F.interpolate(surface, scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)
        multilevel = F.interpolate(multilevel, scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)

        multilevel = rearrange(multilevel, 'b (c nlevel) zlat zlon -> b c nlevel zlat zlon', nlevel=nlevels)

        if diagnostic is not None:
            diagnostic = F.interpolate(diagnostic, scale_factor=self.downsample_factor, mode='bilinear', align_corners=False)

        return surface, multilevel, diagnostic