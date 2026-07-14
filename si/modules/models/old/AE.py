import torch
import torch.nn as nn
from einops import rearrange

from modules.layers.old.dc_layers import SphereConv2d, LayerNorm2d, LayerNorm3d, \
    PixelShuffleUpSampleLayer, PixelUnshuffleDownSampleLayer, ChannelAveragingDownSampleLayer, ChannelDuplicatingUpSampleLayer

def conv(conv_type, **kwargs):
    if conv_type == '2d':
        return nn.Conv2d(**kwargs)
    elif conv_type == '3d':
        return nn.Conv3d(**kwargs)
    elif conv_type == 'spherical':
        return SphereConv2d(**kwargs)
    else:
        raise ValueError(f"Unsupported conv_type: {conv_type}")

class DCDownBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        conv_type = '2d'
    ) -> None:
        super().__init__()

        self.conv_block = PixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=factor, conv_type=conv_type
        )
        self.shortcut_block = ChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.conv_block(x) + self.shortcut_block(x)
    
class AvgDownBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        conv_type = '2d'
    ) -> None:
        super().__init__()

        self.conv = conv(conv_type,
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=3, 
                        padding=1)
        self.downsample = nn.AvgPool2d(kernel_size=factor, stride=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.downsample(x)
        return x
    
class AvgUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        conv_type = '2d'
    ) -> None:
        super().__init__()

        self.conv = conv(conv_type,
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=3, 
                        padding=1)
        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.upsample(x)
        return x
    
class DownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: int=26,
        factor: int = 2,
        with_conv=True,
    ) -> None:
        super().__init__()

        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels * n_levels,
                                out_channels=out_channels * n_levels,
                                kernel_size=3,
                                padding=1,
                                groups=in_channels * n_levels) # depthwise conv
        self.downsample = nn.AvgPool2d(kernel_size=factor, stride=factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in shape b c nlevel nlat nlon
        nlevel = x.shape[2]

        x = rearrange(x, 'b c nlevel nlat nlon -> b (c nlevel) nlat nlon')

        if self.with_conv:
            x = self.conv(x)

        x = self.downsample(x) # b (c nlevel) nlat//factor nlon//factor

        x = rearrange(x, 'b (c nlevel) nlat nlon -> b c nlevel nlat nlon', nlevel=nlevel)
        return x
    
class UpBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: int=26,
        factor: int = 2,
        with_conv=True,
    ) -> None:
        super().__init__()

        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels=in_channels * n_levels,
                                out_channels=out_channels * n_levels,
                                kernel_size=3,
                                padding=1,
                                groups=out_channels * n_levels) # depthwise conv
        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in shape b c nlevel nlat nlon
        nlevel = x.shape[2]

        x = rearrange(x, 'b c nlevel nlat nlon -> b (c nlevel) nlat nlon')

        if self.with_conv:
            x = self.conv(x)
            
        x = self.upsample(x) # b (c nlevel) nlat*factor nlon*factor

        x = rearrange(x, 'b (c nlevel) nlat nlon -> b c nlevel nlat nlon', nlevel=nlevel)
        return x

class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        conv_type = '2d'
    ) -> None:
        super().__init__()
        self.conv_block = PixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=factor, conv_type=conv_type
        )
        self.shortcut_block = ChannelDuplicatingUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=factor)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.conv_block(x) + self.shortcut_block(x)

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type = '2d'
    ) -> None:
        super().__init__()

        self.nonlinearity = nn.GELU()
        self.conv1 = conv(conv_type,
                          in_channels = in_channels, 
                          out_channels = in_channels, 
                          kernel_size=3, 
                          padding=1)
        self.conv2 = conv(conv_type,
                          in_channels=in_channels, 
                          out_channels = out_channels,
                          kernel_size=3, 
                          padding=1, 
                          bias=False)
        if conv_type == '3d':
            self.norm = LayerNorm3d(out_channels)
        else:
            self.norm = LayerNorm2d(out_channels)

    def forward(self, x) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x + residual
    
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels = (256, 256, 512, 512),
        blocks_per_layer = (2, 2, 2, 2),
        conv_type = "2d",
        saturate=False,
        downsample_type = "DC"
    ):
        super().__init__()

        self.saturate_latent = saturate

        num_layers = len(hidden_channels)
        latent_channels = in_channels

        self.conv_in = conv(
            conv_type,
            in_channels = in_channels,
            out_channels = hidden_channels[0],
            kernel_size = 3,
            padding = 1
        )
        
        self.down_layers = nn.ModuleList()
        for i, (out_channel, num_blocks) in enumerate(
            zip(hidden_channels, blocks_per_layer)
        ):
            for _ in range(num_blocks):
                block = ResBlock(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    conv_type=conv_type
                )
                self.down_layers.append(block)

            if i < num_layers - 1: # no downsample on last layer
                if downsample_type == "DC":
                    downsample_block = DCDownBlock2d(
                        in_channels=out_channel,
                        out_channels=hidden_channels[i + 1],
                        conv_type=conv_type
                    )
                elif downsample_type == "avg":
                    downsample_block = AvgDownBlock2d(in_channels=out_channel,
                                                      out_channels=hidden_channels[i + 1],
                                                     conv_type=conv_type)
                else:
                    raise ValueError(f"Unsupported downsample_type: {downsample_type}")
                self.down_layers.append(downsample_block)

        self.conv_out = conv(
            conv_type=conv_type,
            in_channels = hidden_channels[-1], 
            out_channels = latent_channels, 
            kernel_size=3,
            padding=1
        )

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
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def saturate(self, x, B=5.0):
        x = x /torch.sqrt(1 + x**2/B**2)
        return x

    def forward(self, surface, multilevel, diagnostic) -> torch.Tensor:
        # surface in shape b nlat nlon c 
        # multilevel in shape b nlevel nlat nlon c
        # diagnostic in shape b nlat nlon c
        n_surface = surface.shape[-1]
        n_diagnostic = diagnostic.shape[-1]
        n_levels = multilevel.shape[1]

        surface = rearrange(surface, 'b nlat nlon c -> b c nlat nlon')
        diagnostic = rearrange(diagnostic, 'b nlat nlon c -> b c nlat nlon')
        # flatten levels to channels. This is because we are purely compressing in lat/lon dimensions
        multilevel = rearrange(multilevel, 'b nlevel nlat nlon c -> b (c nlevel) nlat nlon')

        x = torch.cat([surface, diagnostic, multilevel], dim=1) # b c nlat nlon

        x = self.conv_in(x) # b hidden_dim nlat nlon

        for down_block in self.down_layers:
            x = down_block(x) 

        x = self.conv_out(x) # b latent_dim zlat zlon

        if self.saturate_latent:
            x = self.saturate(x)

        z_surface = x[:, :n_surface, :, :] # b n_surface zlat zlon
        z_diagnostic = x[:, n_surface:n_surface + n_diagnostic, :, :] # b n_diagnostic zlat zlon
        z_multilevel = x[:, n_surface + n_diagnostic:, :, :] # b (n_multilevel * nlevel) zlat zlon

        z_multilevel = rearrange(z_multilevel, 'b (c nlevel) zlat zlon -> b nlevel zlat zlon c', nlevel=n_levels)
        z_surface = rearrange(z_surface, 'b c zlat zlon -> b zlat zlon c')
        z_diagnostic = rearrange(z_diagnostic, 'b c zlat zlon -> b zlat zlon c')

        return z_surface, z_multilevel,  z_diagnostic
    
class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels = (512, 512, 256, 256),
        blocks_per_layer = (2, 2, 2, 2),
        conv_type = "2d",
        upsample_type = "DC"
    ):
        super().__init__()

        num_layers = len(hidden_channels)
        latent_channels = in_channels

        self.conv_in = conv(
            conv_type,
            in_channels = latent_channels,
            out_channels = hidden_channels[0],
            kernel_size=3,
            padding=1
        )
        
        self.up_layers = nn.ModuleList()
        for i, (out_channel, num_blocks) in enumerate(
            zip(hidden_channels, blocks_per_layer)
        ):
            for _ in range(num_blocks):
                block = ResBlock(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    conv_type=conv_type
                )
                self.up_layers.append(block)

            if i < num_layers - 1: # no upsample on last layer
                if upsample_type == "DC":
                    upsample_block = DCUpBlock2d(
                        in_channels=out_channel,
                        out_channels=hidden_channels[i + 1],
                        conv_type=conv_type
                    )
                elif upsample_type == "avg":
                    upsample_block = AvgUpBlock2d(in_channels=out_channel,
                                                    out_channels=hidden_channels[i + 1],
                                                     conv_type=conv_type)
                else:
                    raise ValueError(f"Unsupported upsample_type: {upsample_type}")
                
                self.up_layers.append(upsample_block)

        self.conv_out = conv(
            conv_type=conv_type,
            in_channels = hidden_channels[-1], 
            out_channels = in_channels, 
            kernel_size=3,
            padding=1)
        
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
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, surface, multilevel, diagnostic) -> torch.Tensor:
        # surface in shape b zlat zlon c 
        # multilevel in shape b nlevel zlat zlon c
        # diagnostic in shape b zlat zlon c

        n_surface = surface.shape[-1]
        n_diagnostic = diagnostic.shape[-1]
        n_levels = multilevel.shape[1]

        surface = rearrange(surface, 'b zlat zlon c -> b c zlat zlon')
        diagnostic = rearrange(diagnostic, 'b zlat zlon c -> b c zlat zlon')
        # flatten levels to channels. This is because we are purely compressing in lat/lon dimensions
        multilevel = rearrange(multilevel, 'b nlevel zlat zlon c -> b (c nlevel) zlat zlon')

        x = torch.cat([surface, diagnostic, multilevel], dim=1) # b c zlat zlon

        x = self.conv_in(x) # b hidden_dim zlat zlon

        for up_block in self.up_layers:
            x = up_block(x) 

        x = self.conv_out(x) # b in_channels nlat nlon

        z_surface = x[:, :n_surface, :, :] # b n_surface nlat nlon
        z_diagnostic = x[:, n_surface:n_surface + n_diagnostic, :, :] # b n_diagnostic nlat nlon
        z_multilevel = x[:, n_surface + n_diagnostic:, :, :] # b (n_multilevel * nlevel) nlat nlon

        z_multilevel = rearrange(z_multilevel, 'b (c nlevel) nlat nlon -> b nlevel nlat nlon c', nlevel=n_levels)
        z_surface = rearrange(z_surface, 'b c nlat nlon -> b nlat nlon c')
        z_diagnostic = rearrange(z_diagnostic, 'b c nlat nlon -> b nlat nlon c')

        return z_surface, z_multilevel, z_diagnostic

class Encoder3D(nn.Module):
    def __init__(
        self,
        surface_channels,
        multilevel_channels,
        diagnostic_channels,
        nlevels=26,
        hidden_channels = (64, 128, 256),
        blocks_per_layer = (2, 2, 2),
        conv_type = "3d"
    ):
        super().__init__()

        num_layers = len(hidden_channels)

        self.surface_in = conv(
            conv_type='2d',
            in_channels = surface_channels,
            out_channels = hidden_channels[0],
            kernel_size = 3,
            padding = 1
        )

        self.diagnostic_in = conv(
            conv_type='2d',
            in_channels =  diagnostic_channels,
            out_channels = hidden_channels[0],
            kernel_size = 3,
            padding = 1
        )

        self.multilevel_in = conv(
            conv_type,
            in_channels = multilevel_channels,
            out_channels = hidden_channels[0],
            kernel_size = 3,
            padding = 1
        )
        
        self.down_layers = nn.ModuleList()
        for i, (out_channel, num_blocks) in enumerate(
            zip(hidden_channels, blocks_per_layer)
        ):
            for _ in range(num_blocks):
                block = ResBlock(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    conv_type=conv_type
                )
                self.down_layers.append(block)

            if i < num_layers - 1: # no downsample on last layer
                downsample_block = DownBlock3d(
                    in_channels=out_channel,
                    out_channels=hidden_channels[i + 1],
                    n_levels=nlevels+2, # account for surface + diagnostic levels
                )
                self.down_layers.append(downsample_block)

        self.multilevel_out = conv(
            conv_type=conv_type,
            in_channels = hidden_channels[-1], 
            out_channels = multilevel_channels, 
            kernel_size=3,
            padding=1)
        
        self.surface_out = conv(
            conv_type='2d',
            in_channels = hidden_channels[-1], 
            out_channels = surface_channels, 
            kernel_size=3,
            padding=1)

        self.diagnostic_out = conv(
            conv_type='2d',
            in_channels = hidden_channels[-1],
            out_channels = diagnostic_channels, 
            kernel_size=3,
            padding=1)

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
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, surface, multilevel, diagnostic) -> torch.Tensor:
        # surface in shape b nlat nlon c 
        # multilevel in shape b nlevel nlat nlon c
        # diagnostic in shape b nlat nlon c

        surface = rearrange(surface, 'b nlat nlon c -> b c nlat nlon')
        diagnostic = rearrange(diagnostic, 'b nlat nlon c -> b c nlat nlon')
        # flatten levels to channels. This is because we are purely compressing in lat/lon dimensions
        multilevel = rearrange(multilevel, 'b nlevel nlat nlon c -> b c nlevel nlat nlon')

        surface = self.surface_in(surface).unsqueeze(2) # b hidden_dim 1 nlat nlon
        diagnostic = self.diagnostic_in(diagnostic).unsqueeze(2) # b hidden_dim 1 nlat nlon
        multilevel = self.multilevel_in(multilevel) # b hidden_dim nlevel nlat nlon

        x = torch.cat([surface, diagnostic, multilevel], dim=2) # b hidden_dim nlevel+2 nlat nlon

        for down_block in self.down_layers:
            x = down_block(x) 

        z_surface = x[:, :, 0, :, :] # b hidden_dim zlat zlon
        z_diagnostic = x[:, :, 1, :, :] # b hidden_dim zlat zlon
        z_multilevel = x[:, :, 2:, :] # b hidden_dim nlevel zlat zlon

        z_surface = self.surface_out(z_surface) # b n_surface zlat zlon
        z_diagnostic = self.diagnostic_out(z_diagnostic) # b n_diagnostic zlat zlon
        z_multilevel = self.multilevel_out(z_multilevel) # b n_multi nlevel zlat zlon

        z_multilevel = rearrange(z_multilevel, 'b c nlevel zlat zlon -> b nlevel zlat zlon c')
        z_surface = rearrange(z_surface, 'b c zlat zlon -> b zlat zlon c')
        z_diagnostic = rearrange(z_diagnostic, 'b c zlat zlon -> b zlat zlon c')

        return z_surface, z_multilevel,  z_diagnostic
    
class Decoder3D(nn.Module):
    def __init__(
        self,
        surface_channels,
        multilevel_channels,
        diagnostic_channels,
        nlevels=26,
        hidden_channels = (256, 128, 64),
        blocks_per_layer = (2, 2, 2),
        conv_type = "3d"
    ):
        super().__init__()

        num_layers = len(hidden_channels)

        self.surface_in = conv(
            conv_type='2d',
            in_channels = surface_channels,
            out_channels = hidden_channels[0],
            kernel_size = 3,
            padding = 1
        )

        self.diagnostic_in = conv(
            conv_type='2d',
            in_channels =  diagnostic_channels,
            out_channels = hidden_channels[0],
            kernel_size = 3,
            padding = 1
        )

        self.multilevel_in = conv(
            conv_type,
            in_channels = multilevel_channels,
            out_channels = hidden_channels[0],
            kernel_size = 3,
            padding = 1
        )
        
        self.up_layers = nn.ModuleList()
        for i, (out_channel, num_blocks) in enumerate(
            zip(hidden_channels, blocks_per_layer)
        ):
            for _ in range(num_blocks):
                block = ResBlock(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    conv_type=conv_type
                )
                self.up_layers.append(block)

            if i < num_layers - 1: # no downsample on last layer
                upsample_block = UpBlock3d(
                    in_channels=out_channel,
                    out_channels=hidden_channels[i + 1],
                    n_levels=nlevels+2, # account for surface + diagnostic levels
                )
                self.up_layers.append(upsample_block)

        self.multilevel_out = conv(
            conv_type=conv_type,
            in_channels = hidden_channels[-1], 
            out_channels = multilevel_channels, 
            kernel_size=3,
            padding=1)
        
        self.surface_out = conv(
            conv_type='2d',
            in_channels = hidden_channels[-1], 
            out_channels = surface_channels, 
            kernel_size=3,
            padding=1)

        self.diagnostic_out = conv(
            conv_type='2d',
            in_channels = hidden_channels[-1],
            out_channels = diagnostic_channels, 
            kernel_size=3,
            padding=1)

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
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, surface, multilevel, diagnostic) -> torch.Tensor:
        # surface in shape b nlat nlon c 
        # multilevel in shape b nlevel nlat nlon c
        # diagnostic in shape b nlat nlon c

        surface = rearrange(surface, 'b nlat nlon c -> b c nlat nlon')
        diagnostic = rearrange(diagnostic, 'b nlat nlon c -> b c nlat nlon')
        # flatten levels to channels. This is because we are purely compressing in lat/lon dimensions
        multilevel = rearrange(multilevel, 'b nlevel nlat nlon c -> b c nlevel nlat nlon')

        surface = self.surface_in(surface).unsqueeze(2) # b hidden_dim 1 nlat nlon
        diagnostic = self.diagnostic_in(diagnostic).unsqueeze(2) # b hidden_dim 1 nlat nlon
        multilevel = self.multilevel_in(multilevel) # b hidden_dim nlevel nlat nlon

        x = torch.cat([surface, diagnostic, multilevel], dim=2) # b hidden_dim nlevel+2 nlat nlon

        for up_block in self.up_layers:
            x = up_block(x) 

        z_surface = x[:, :, 0, :, :] # b hidden_dim zlat zlon
        z_diagnostic = x[:, :, 1, :, :] # b hidden_dim zlat zlon
        z_multilevel = x[:, :, 2:, :] # b hidden_dim nlevel zlat zlon

        z_surface = self.surface_out(z_surface) # b n_surface zlat zlon
        z_diagnostic = self.diagnostic_out(z_diagnostic) # b n_diagnostic zlat zlon
        z_multilevel = self.multilevel_out(z_multilevel) # b n_multi nlevel zlat zlon

        z_multilevel = rearrange(z_multilevel, 'b c nlevel zlat zlon -> b nlevel zlat zlon c')
        z_surface = rearrange(z_surface, 'b c zlat zlon -> b zlat zlon c')
        z_diagnostic = rearrange(z_diagnostic, 'b c zlat zlon -> b zlat zlon c')

        return z_surface, z_multilevel,  z_diagnostic