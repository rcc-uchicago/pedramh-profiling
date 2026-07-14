import torch
import torch.nn as nn
from einops import rearrange
from timm.layers.mlp import SwiGLU
from common.utils import disassemble_input, disassemble_prognostic_forcing, assemble_input

from modules.layers.old.arches_layers import (
    CondBasicLayer,
    DCDownSample,
    LinVert,
    Mlp,
    DCUpSample,
    ICNR_init,
    TimestepEmbedder
)

class WeatherEncodeDecodeLayer(nn.Module):
    """
    gathers layers for the encoder and decoder
    """

    def __init__(
        self,
        emb_dim=192,
        out_emb_dim=2 * 192,  
        patch_size=(2, 2, 2),
        surface_ch=6,
        level_ch=9,
        forcing_ch=3,
        invariant_ch=2,
        diagnostic_ch=9,
        encode_noise=True
    ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.surface_ch = surface_ch
        self.level_ch = level_ch
        self.forcing_ch = forcing_ch
        self.invariant_ch = invariant_ch
        self.diagnostic_ch = diagnostic_ch
        self.encode_noise = encode_noise

        surface_ch_in = surface_ch + forcing_ch + invariant_ch
        level_ch_in = level_ch 
        diag_ch_in = diagnostic_ch

        if self.encode_noise:
            surface_ch_in += surface_ch
            level_ch_in += level_ch
            diag_ch_in += diagnostic_ch

        self.level_proj = nn.Conv3d(
            level_ch_in, emb_dim, kernel_size=patch_size, stride=patch_size
        )
        self.surface_proj = nn.Conv2d(
            surface_ch_in, emb_dim, kernel_size=patch_size[1:], stride=patch_size[1:]
        )

        self.diagnostic_proj = nn.Conv2d(
            diag_ch_in, emb_dim, kernel_size=patch_size[1:], stride=patch_size[1:]
        )

        self.surface_deconv = nn.Conv2d(
            out_emb_dim,
            surface_ch * patch_size[-1] ** 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=0,
        )
        self.level_patch = patch_size[0]
        self.level_deconv = nn.Conv2d(
            out_emb_dim // self.level_patch,
            level_ch * patch_size[-1] ** 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=0,
        )

        self.diag_deconv = nn.Conv2d(
            out_emb_dim,
            diagnostic_ch * patch_size[-1] ** 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=0,
        )

        self.pixelshuffle = nn.PixelShuffle(patch_size[-1])
        # Apply He Initialization
        self.apply(self._init_weights)
        ICNR_init(
            self.surface_deconv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size[-1],
        )
        ICNR_init(
            self.level_deconv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size[-1],
        )
        ICNR_init(
            self.diag_deconv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size[-1],
        )

    def _init_weights(self, m):
        """
        Applies He (Kaiming) initialization to Conv2d and Linear layers.
        Initializes normalization layers (LayerNorm, BatchNorm) with scale 1 and bias 0.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x, cond):
        """
        x is the noised state
        cond is the conditioning state
        """

        surface_noised, multi_noised, diag_noised = disassemble_input(x)
        surface, multilevel, diagnostic, forcing, invariants = disassemble_prognostic_forcing(cond)

        surface = rearrange(surface, "b nlat nlon c -> b c nlat nlon")
        multilevel = rearrange(
            multilevel, "b nlevel nlat nlon c -> b c nlevel nlat nlon"
        )
        forcing = rearrange(forcing, "b nlat nlon c -> b c nlat nlon")
        invariants = rearrange(invariants, "b nlat nlon c -> b c nlat nlon")
        diagnostic = rearrange(diagnostic, "b nlat nlon c -> b c nlat nlon")
        
        surface = torch.cat([surface, forcing, invariants], dim=1) # b (surface_ch + forcing_ch + invariant_ch) nlat nlon
        
        if self.encode_noise:
            surface_noised = rearrange(surface_noised, "b nlat nlon c -> b c nlat nlon")
            diag_noised = rearrange(diag_noised, "b nlat nlon c -> b c nlat nlon")
            multi_noised = rearrange(multi_noised, "b nlevel nlat nlon c -> b c nlevel nlat nlon")

            surface = torch.cat([surface, surface_noised], dim=1) # b (surface_ch + forcing_ch + invariant_ch + surface_noised_ch + diagnostic_ch) nlat nlon
            multilevel = torch.cat([multilevel, multi_noised], dim=1) # b (level_ch + multi_noised_ch) nlevel nlat nlon
            diagnostic = torch.cat([diagnostic, diag_noised], dim=1) # b (diagnostic_ch + diag_noised_ch) nlat nlon

        # patchify
        surface = self.surface_proj(surface) # b emb_dim zlat zlon
        level = self.level_proj(multilevel) # b emb_dim zlevel zlat zlon
        diagnostic = self.diagnostic_proj(diagnostic) # b emb_dim zlat zlon

        x = torch.concat([surface.unsqueeze(2), diagnostic.unsqueeze(2), level], dim=2) # b emb_dim (2 + zlevel) zlat zlon
        return x

    def decode(self, x):
        # x: b, emb_dim, zlevel+1, zlat, zlon
        b = x.shape[0]

        surface, diagnostic, level = x[:, :, 0], x[:, :, 1], x[:, :, 2:]

        output_surface = self.surface_deconv(surface) # b, surface_ch * r^2, zlat, zlon
        output_surface = self.pixelshuffle(output_surface) # b, surface_ch, lat, lon
        
        output_diagnostic = self.diag_deconv(diagnostic) # b, diagnostic_ch * r^2, zlat, zlon
        output_diagnostic = self.pixelshuffle(output_diagnostic) # b, diagnostic_ch, lat, lon

        # do channel to level expansion
        output_level = rearrange(level, 'b (c p) zlevel zlat zlon -> b c (p zlevel) zlat zlon', p = self.level_patch)

        # lump levels into batch dim for deconv
        output_level = rearrange(output_level, "b c zlevel zlat zlon -> (b zlevel) c zlat zlon")
        output_level = self.level_deconv(output_level) # b*zlevel, level_ch * r^2, zlat, zlon
        output_level = self.pixelshuffle(output_level) # b*zlevel, level_ch, lat, lon
        output_level = rearrange(output_level, "(b zlevel) c zlat zlon -> b c zlevel zlat zlon", b=b) # b, level_ch, zlevel, lat, lon

        output_surface = rearrange(output_surface, "b c nlat nlon -> b nlat nlon c")
        output_level = rearrange(output_level, "b c nlevel nlat nlon -> b nlevel nlat nlon c")
        output_diagnostic = rearrange(output_diagnostic, "b c nlat nlon -> b nlat nlon c")

        return output_surface, output_level, output_diagnostic


class ArchesDiT(nn.Module):
    def __init__(
        self,
        tensor_size=(15, 90, 180),
        patch_size=(2, 2, 2),
        emb_dim=128,
        cond_dim=None,  # dim of the conditioning
        num_heads=(4, 8, 8, 4),
        window_size=(1, 5, 10),
        depth_multiplier=1,
        dropout=0.0,
        mlp_ratio=4.0,
        use_skip=True,
        first_interaction_layer="linear",
        mlp_layer="swiglu",
        surface_ch=6,
        level_ch=9,
        forcing_ch=3,
        invariant_ch=2,
        diagnostic_ch=9,
        **kwargs,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.first_interaction_layer = first_interaction_layer

        if cond_dim is None:
            cond_dim = emb_dim

        self.encode_decode = WeatherEncodeDecodeLayer(emb_dim=emb_dim,
                                                      out_emb_dim=2*emb_dim,# 2x since skip connection
                                                      patch_size=patch_size,
                                                      surface_ch=surface_ch,
                                                      level_ch=level_ch,
                                                      forcing_ch=forcing_ch,
                                                      invariant_ch=invariant_ch,
                                                      diagnostic_ch=diagnostic_ch,) 
        self.zdim = tensor_size[0]
        
        self.layer1_shape = tensor_size[1:]

        self.layer2_shape = (self.layer1_shape[0] // 2, self.layer1_shape[1] // 2)

        if first_interaction_layer == "linear":
            self.interaction_layer = LinVert(in_features=emb_dim,
                                             n_cols = self.zdim)

        layer_args = dict(
            cond_dim=cond_dim,
            window_size=window_size,
            act_layer=nn.GELU,
            drop=dropout,
            mlp_layer=Mlp,
            mlp_ratio=mlp_ratio,
        )

        if mlp_layer == "swiglu":
            layer_args["mlp_ratio"] = mlp_ratio * 2 / 3
            layer_args["mlp_layer"] = SwiGLU

        self.layer1 = CondBasicLayer(
            dim=emb_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2 * depth_multiplier,
            num_heads=num_heads[0],
            **layer_args,
            **kwargs,
        )
        self.downsample = DCDownSample(
            in_dim=emb_dim,
            out_dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer1_shape),
            output_resolution=(self.zdim, *self.layer2_shape),
        )
        self.layer2 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6 * depth_multiplier,
            num_heads=num_heads[1],
            **layer_args,
            **kwargs,
        )
        self.layer3 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6 * depth_multiplier,
            num_heads=num_heads[2],
            **layer_args,
            **kwargs,
        )
        self.upsample = DCUpSample(
            emb_dim * 2, emb_dim, (self.zdim, *self.layer2_shape), (self.zdim, *self.layer1_shape)
        )
        out_dim = emb_dim if not self.use_skip else 2 * emb_dim
        self.layer4 = CondBasicLayer(
            dim=out_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2 * depth_multiplier,
            num_heads=num_heads[3],
            **layer_args,
            **kwargs,
        )

        self.cond_embedders = nn.ModuleList([
            TimestepEmbedder(cond_dim),
            TimestepEmbedder(cond_dim),
            TimestepEmbedder(cond_dim),
        ])

        # Apply He Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Applies He (Kaiming) initialization to Conv2d and Linear layers.
        Initializes normalization layers (LayerNorm, BatchNorm) with scale 1 and bias 0.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, cond, cond_emb):
        # x is noised, prognostic state in shape (b, c, lat, lon)
        # cond is the conditioning state (prognostic + forcing) in shape (b, c, lat, lon)
        # cond_emb in shape (b, 3)
        
        cond_emb = [emb(cond_emb[:, i]) for i, emb in enumerate(self.cond_embedders)]
        cond_emb = torch.stack(cond_emb, dim=0) # 3, b, cond_dim
        cond_emb = torch.sum(cond_emb, dim=0) # b, cond_dim
        
        x = self.encode_decode.encode(x, cond) 

        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2) # B, N, C

        if self.first_interaction_layer:
            x = self.interaction_layer(x)

        x = self.layer1(x, cond_emb)

        skip = x
        x = self.downsample(x)

        x = self.layer2(x, cond_emb)

        x = self.layer3(x, cond_emb)

        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.concat([x, skip], dim=-1)
        x = self.layer4(x, cond_emb)

        output = x
        output = output.transpose(1, 2).reshape(output.shape[0], -1, self.zdim, *self.layer1_shape)

        output_surface, output_level, output_diagnostic = self.encode_decode.decode(output)

        y = assemble_input(output_surface, output_level, output_diagnostic)

        return y