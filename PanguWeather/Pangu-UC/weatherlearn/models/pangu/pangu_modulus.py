import torch
import numpy as np

import modulus
from dataclasses import dataclass

from .utils.patch_embed import PatchEmbed2D, PatchEmbed3D
from .utils.patch_recovery import PatchRecovery2D, PatchRecovery3D
from .pangu import UpSample, DownSample, BasicLayer
@dataclass
class PanguPlasimModulusMetaData(modulus.ModelMetaData):
    name: str = "PanguPlasimModulus"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True

class PanguPlasimModulus(modulus.Module):
    """
    A general implementation of the Pangu-Weather model for `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(self, embed_dim=192, horizontal_resolution = (64, 128), num_levels = 10, num_atmo_vars = 5,
                 num_surface_vars = 4, num_boundary_vars = 3, patch_size = (2,4,4),
                 num_heads=(6, 12, 12, 6), window_size=(2, 6, 12), depths = (2, 6, 6, 2), drop_path = None,
                 updown_scale_factor = 2, predict_delta = True):
        super(PanguPlasimModulus, self).__init__(meta=PanguPlasimModulusMetaData())
        if not drop_path:
            drop_path = np.append(np.linspace(0, 0.2, np.sum(depths[:2])),
                np.linspace(0.2, 0, np.sum(depths[2:]))).tolist()
        self.num_surface_vars = num_surface_vars
        self.num_atmo_vars = num_atmo_vars
        self.num_boundary_vars = num_boundary_vars
        atmo_resolution = tuple([num_levels]) + horizontal_resolution
        depths_cumsum = np.cumsum(depths).astype(int)
        self.predict_delta = predict_delta
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        self.patchembed2d = PatchEmbed2D(
            img_size=horizontal_resolution,
            patch_size=patch_size[1:],
            in_chans=num_surface_vars + num_boundary_vars,  # add
            embed_dim=embed_dim,
        )
        self.patchembed3d = PatchEmbed3D(
            img_size=atmo_resolution,
            patch_size=patch_size,
            in_chans=num_atmo_vars,
            embed_dim=embed_dim
        )
        EST_input_resolution = (self.patchembed3d.output_size[0]+1, self.patchembed3d.output_size[1],
                                self.patchembed3d.output_size[2])
        downscale_resolution = (self.patchembed3d.output_size[0]+1,
                                (self.patchembed2d.output_size[0] - self.patchembed2d.output_size[0] % updown_scale_factor) \
                                // updown_scale_factor + self.patchembed2d.output_size[0] % updown_scale_factor,
                                (self.patchembed2d.output_size[1] - self.patchembed2d.output_size[1] % updown_scale_factor) \
                                // updown_scale_factor + self.patchembed2d.output_size[1] % updown_scale_factor)

        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=EST_input_resolution,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:depths_cumsum[0]]
        )
        self.downsample = DownSample(in_dim=embed_dim,
                                     input_resolution=EST_input_resolution,
                                     output_resolution=downscale_resolution, downsample_factor=updown_scale_factor)
        self.layer2 = BasicLayer(
            dim=embed_dim * updown_scale_factor,
            input_resolution=downscale_resolution,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[depths_cumsum[0]:depths_cumsum[1]]
        )
        self.layer3 = BasicLayer(
            dim=embed_dim * updown_scale_factor,
            input_resolution=downscale_resolution,
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[depths_cumsum[1]:depths_cumsum[2]]
        )
        self.upsample = UpSample(embed_dim * updown_scale_factor, embed_dim,
                                 downscale_resolution, (self.patchembed3d.output_size[0]+1,
                                                       self.patchembed3d.output_size[1],
                                                       self.patchembed3d.output_size[2]))
        self.layer4 = BasicLayer(
            dim=embed_dim,
            input_resolution=EST_input_resolution,
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[depths_cumsum[2]:]
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = PatchRecovery2D(horizontal_resolution, patch_size[1:], 2 * embed_dim, num_surface_vars)
        self.patchrecovery3d = PatchRecovery3D(atmo_resolution, patch_size, 2 * embed_dim, num_atmo_vars)

    def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        if len(constant_boundary.size()) == 3:
            constant_boundary = constant_boundary.unsqueeze(0)
        surface = torch.concat([surface_in, constant_boundary, varying_boundary], dim=1)
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air_in)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        x = self.layer1(x)

        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        if self.predict_delta:
            output_surface_delta = output[:, :, 0, :, :]
            output_upper_air_delta = output[:, :, 1:, :, :]

            output_surface_delta = self.patchrecovery2d(output_surface_delta)
            output_upper_air_delta = self.patchrecovery3d(output_upper_air_delta)
            
            output_surface = surface_in + output_surface_delta
            output_upper_air = upper_air_in + output_upper_air_delta
        else:
            output_surface = output[:, :, 0, :, :]
            output_upper_air = output[:, :, 1:, :, :]

            output_surface = self.patchrecovery2d(output_surface)
            output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air
