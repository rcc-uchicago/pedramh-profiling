import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from modules.layers.positional_encoding import TimestepEmbedder

from modules.layers.spherical_harmonics import SphericalHarmonicsPE

from modules.layers.old.factorized_attention import FADiTBlockS2
from modules.layers.unpatchify import SubPixelConvICNR_2D, Unpatchify
from modules.layers.patchify import PatchEmbed
from modules.layers.cross_attention import CrossAttentionBlock

class ClimaDiT(nn.Module):

    def __init__(self, 
                 in_dim,
                 out_dim,
                 dim,
                 num_heads,
                 num_blocks,
                 num_out_blocks = 1,
                 patch_size=4,
                 z_patch_size=1,
                 nlat = 180,
                 nlon = 360,
                 dropout = 0,
                 unpatch="subpixel",
                 nsurface=6,
                 ndiagnostic=9,
                 nlevels=26
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.num_heads = num_heads
        self.num_fa_blocks = num_blocks
        self.num_ca_blocks = num_blocks
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.nlat = nlat
        self.nlon = nlon
        self.unpatch = unpatch
        self.dropout = dropout
        self.nsurface = nsurface
        self.ndiagnostic = ndiagnostic
        self.nlevels = nlevels
        self.z_patch_size = z_patch_size

        self.grid_x = self.nlat // self.patch_size
        self.grid_y = self.nlon // self.patch_size

        self.with_poles = False

        self.z_embedder = nn.Sequential(
            Rearrange('b ny nx c -> b c ny nx'),
            nn.Conv2d(self.in_dim,
                      self.dim,
                      kernel_size=self.z_patch_size, stride=self.z_patch_size, padding=0),
            nn.SiLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0),
            Rearrange('b c ny nx -> b ny nx c')
        )

        # input embedding
        self.patch_embed = PatchEmbed(patch_size=self.patch_size,
                                      in_chans=self.in_dim,
                                      hidden_size=self.dim,
                                      flatten=False)

        # positional embedding
        l_max = 20
        self.pe_embed = SphericalHarmonicsPE(l_max, self.dim, self.dim,
                                             use_mlp=True)
        self.pe2patch = PatchEmbed(patch_size=self.patch_size,
                                   in_chans=self.dim,
                                   hidden_size=self.dim,
                                   flatten=False)
        
        self.cond_embedder = TimestepEmbedder(self.dim)

        fa_blocks = []
        for _ in range(self.num_fa_blocks):
            fa_blocks.append(FADiTBlockS2(self.dim,
                                       self.dim // self.num_heads,
                                       self.num_heads,
                                       self.dim,
                                       self.dim,
                                       self.dim,
                                       use_softmax=True,
                                       depth_dropout=self.dropout))
            
        ca_blocks = []
        for _ in range(self.num_ca_blocks):
            ca_blocks.append(CrossAttentionBlock(self.num_heads,
                                                 self.dim,))
            
        self.fa_blocks = nn.ModuleList(fa_blocks)
        self.ca_blocks = nn.ModuleList(ca_blocks)
            
        if self.unpatch == "subpixel":
            self.unpatchify_layer = SubPixelConvICNR_2D(img_size=(self.nlat, self.nlon), 
                                                        patch_size=(self.patch_size, self.patch_size),
                                                        in_chans=self.dim,
                                                        out_chans=self.dim,
                                                        cond_dim=self.dim,
                                                        num_lat=self.nlat)
        elif self.unpatch == "vanilla":
            self.unpatchify_layer = Unpatchify(grid_size=(self.grid_x, self.grid_y),
                                               patch_size=(self.patch_size, self.patch_size),
                                                in_dim=self.dim,
                                                out_dim=self.dim,
                                                cond_dim=self.dim)
        else:
            raise ValueError("unpatch type not supported")
        
        self.out_fa_blocks = nn.ModuleList()
        self.num_out_blocks = num_out_blocks

        for _ in range(num_out_blocks):
            self.out_fa_blocks.append(FADiTBlockS2(self.dim,
                        self.dim // self.num_heads,
                        self.num_heads,
                        self.dim,
                        self.dim,
                        self.dim,
                        use_softmax=True,
                        depth_dropout=self.dropout))
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.out_dim))
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    @torch.no_grad()
    def get_grid(self, nlat, nlon, device):
        # create lat, lon grid
        if self.with_poles:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat).to(device)
        else:
            # assume equiangular grid
            lat_end = (nlat - 1) * (2 * math.pi / nlon) / 2
            lat = torch.linspace(-lat_end, lat_end, nlat).to(device)

        lon = torch.linspace(0, 2 * math.pi - (2 * math.pi / nlon), nlon).to(device)
        latlon = torch.stack(torch.meshgrid(lat, lon), dim=-1)
        return latlon, lat, lon
    
    def assemble_input(self, surface, multilevel, diagnostic=None):
        multilevel = rearrange(
            multilevel, "b l h w c -> b h w (l c)"
        )

        if diagnostic is None:
            out = torch.cat((surface, multilevel), dim=-1) # b h w c
        else:
            out = torch.cat((surface, diagnostic, multilevel), dim=-1) # b h w c

        return out
    
    def disassemble_input(self, x, use_diagnostic=True):

        if use_diagnostic:
            surface = x[..., : self.nsurface]
            diagnostic = x[..., self.nsurface : self.nsurface + self.ndiagnostic]
            multilevel = x[..., self.nsurface + self.ndiagnostic :]
        else:
            surface = x[..., : self.nsurface]
            multilevel = x[..., self.nsurface :]
            diagnostic = None

        multilevel = rearrange(
            multilevel,
            "b h w (l c) -> b l h w c",
            l=self.nlevels,
        )

        return surface, multilevel, diagnostic

    def forward(self, surface_history, multilevel_history, diagnostic_history = None,
                z_surface=None, z_history=None, z_diagnostic=None, t=None):

        x = self.assemble_input(surface_history, multilevel_history, diagnostic_history) # b h w c
        z = self.assemble_input(z_surface, z_history, z_diagnostic) # b h w c

        batch_size = x.size(0)
        nlat, nlon, = x.size(1), x.size(2)
        nlat_grid = nlat // self.patch_size
        nlon_grid = nlon // self.patch_size
        _, lat, lon = self.get_grid(nlat, nlon, x.device)
        _, lat_grid, lon_grid = self.get_grid(nlat_grid, nlon_grid, x.device)

        # n x n distance matrix
        lat_grid_diff = lat_grid.unsqueeze(0) - lat_grid.unsqueeze(1)
        lon_grid_diff = lon_grid.unsqueeze(0) - lon_grid.unsqueeze(1)

        lat_diff = lat.unsqueeze(0) - lat.unsqueeze(1)
        lon_diff = lon.unsqueeze(0) - lon.unsqueeze(1)

        # patchify x
        x = self.patch_embed(x) # [b, nlat, nlon, c] -> [b, nlat//p, nlon//p, dim]

        # patchify latent
        z = self.z_embedder(z) # [b, h, w, c] 

        # patchify pos embed, lat from 0 to pi, lon from -pi to pi
        sphere_pe = self.pe_embed(lat + math.pi/2, lon - math.pi).expand(batch_size, -1, -1, -1) # [b, nlat, nlon, dim]
        sphere_pe = self.pe2patch(sphere_pe) # [b, nlat//p, nlon//p, dim]

        x = x + sphere_pe   # [b, nlat//p, nlon//p, dim]
        z = z + sphere_pe   # [b, nlat//p, nlon//p, dim]

        if t is None:
            t = torch.ones(batch_size, 1, device=x.device)

        c = self.cond_embedder(t)

        for l in range(self.num_blocks):
            z = self.fa_blocks[l](z, lat_grid, lat_grid_diff, lon_grid_diff, c)
            z = self.ca_blocks[l](z, x, reshape=True)

        # flatten x after factorized attention
        z = rearrange(x, 'b ny nx c -> b (ny nx) c') # [b, nlat//p * nlon//p, dim]

        z = self.unpatchify_layer(z, c) # [b, h, w, out_dim]

        for l in range(self.num_out_blocks):
            z = self.out_fa_blocks[l](z, lat, lat_diff, lon_diff, c)

        z = self.out_proj(z) # [b, h, w, out_dim]

        surface, multilevel, diagnostic = self.disassemble_input(z, use_diagnostic=(diagnostic_history is not None))

        return surface, multilevel, diagnostic
    

class ClimaSiT(nn.Module):

    def __init__(self, 
                 in_dim,
                 out_dim,
                 dim,
                 num_heads,
                 num_blocks,
                 nlat = 180,
                 nlon = 360,
                 dropout = 0,
                 nsurface=6,
                 ndiagnostic=9,
                 nlevels=26
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.num_heads = num_heads
        self.num_fa_blocks = num_blocks
        self.num_ca_blocks = num_blocks
        self.num_blocks = num_blocks
        self.nlat = nlat
        self.nlon = nlon
        self.dropout = dropout
        self.nsurface = nsurface
        self.ndiagnostic = ndiagnostic
        self.nlevels = nlevels

        self.grid_x = self.nlat 
        self.grid_y = self.nlon 

        self.with_poles = False

        # input embedding
        self.x_embed = PatchEmbed(patch_size=1,
                                      in_chans=self.in_dim,
                                      hidden_size=self.dim,
                                      flatten=False)

        # positional embedding
        l_max = 20
        self.pe_embed = SphericalHarmonicsPE(l_max, self.dim, self.dim,
                                             use_mlp=True)
        self.pe2patch = PatchEmbed(patch_size=1,
                                   in_chans=self.dim,
                                   hidden_size=self.dim,
                                   flatten=False)
        
        self.cond_embedder = TimestepEmbedder(self.dim)

        fa_blocks = []
        for _ in range(self.num_fa_blocks):
            fa_blocks.append(FADiTBlockS2(self.dim,
                                       self.dim // self.num_heads,
                                       self.num_heads,
                                       self.dim,
                                       self.dim,
                                       self.dim,
                                       use_softmax=True,
                                       depth_dropout=self.dropout))
            
        self.fa_blocks = nn.ModuleList(fa_blocks)
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.out_dim))
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    @torch.no_grad()
    def get_grid(self, nlat, nlon, device):
        # create lat, lon grid
        if self.with_poles:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat).to(device)
        else:
            # assume equiangular grid
            lat_end = (nlat - 1) * (2 * math.pi / nlon) / 2
            lat = torch.linspace(-lat_end, lat_end, nlat).to(device)

        lon = torch.linspace(0, 2 * math.pi - (2 * math.pi / nlon), nlon).to(device)
        latlon = torch.stack(torch.meshgrid(lat, lon), dim=-1)
        return latlon, lat, lon

    def forward(self, x, t=None):
        # x in shape b nx ny c

        batch_size = x.size(0)
        nlat, nlon, = x.size(1), x.size(2)
        _, lat, lon = self.get_grid(nlat, nlon, x.device)

        lat_diff = lat.unsqueeze(0) - lat.unsqueeze(1)
        lon_diff = lon.unsqueeze(0) - lon.unsqueeze(1)

        # patchify x
        x = self.x_embed(x) # [b, nlat, nlon, c] -> [b, nlat, nlon, dim]

        # patchify pos embed, lat from 0 to pi, lon from -pi to pi
        sphere_pe = self.pe_embed(lat + math.pi/2, lon - math.pi).expand(batch_size, -1, -1, -1) # [b, nlat, nlon, dim]
        sphere_pe = self.pe2patch(sphere_pe) # [b, nlat, nlon, dim]

        x = x + sphere_pe   # [b, nlat, nlon, dim]

        if t is None:
            t = torch.ones(batch_size, 1, device=x.device)

        c = self.cond_embedder(t)

        for l in range(self.num_blocks):
            x = self.fa_blocks[l](x, lat, lat_diff, lon_diff, c)

        x = self.out_proj(x) # [b, h, w, out_dim]

        return x