class cDiT(nn.Module):
    """
    Architecture:
    - PatchEmbed for main input
    - 2D Rotary Position Embedding (RoPE) on lat/lon patch grid
    - N blocks of DiTBlock (self-attn with AdaLN + RoPE)
    - Unpatchify: dim -> out_channels @ nlat x nlon
    - Zero-initialized output projection
    """

    def __init__(self,
                 in_channels=249,
                 out_channels=249,
                 dim=384,
                 num_heads=8,
                 num_blocks=8,
                 patch_size=2,
                 nlat=180,
                 nlon=360,
                 dropout=0.0,
                 #grid_in_dim = 1,
                 #cond_dim=4,
                 unpatch = "vanilla"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.nlat = nlat
        self.nlon = nlon
        self.dropout = dropout
        self.unpatchify = unpatch

        self.grid_x = self.nlat // patch_size
        self.grid_y = self.nlon // patch_size
        self.with_poles = False

        self.patch_embed_main = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            hidden_size=dim,
            flatten=False)
        
        #self.embed_grid = SphereConv2d(in_channels = grid_in_dim, out_channels = cond_dim, kernel_size=(3,3))
        #self.embed_calendar = CalendarEmbedding(nlon=nlon, nlat=nlat, embed_channels=cond_dim)

        # 2D RoPE: one RotaryEmbedding per spatial axis
        # Each axis gets half the head dimension
        dim_head = dim // num_heads
        self.rope_lat = RotaryEmbedding(dim_head // 2)
        self.rope_lon = RotaryEmbedding(dim_head // 2)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(dim, num_conds = 1)

        # Transformer blocks
        sa_blocks = []
        for _ in range(num_blocks):
            sa_blocks.append(DiTBlock(dim, num_heads, mlp_ratio=4, dropout=dropout))

        self.sa_blocks = nn.ModuleList(sa_blocks)

        if self.unpatchify == "interpolate":
            self.unpatchify_layer = PatchInterpolate2D(grid_size=(self.grid_x, self.grid_y),
                                                       patch_size=(patch_size, patch_size),
                                                       in_chans=dim,
                                                       out_chans=out_channels,
                                                       hidden_dim=dim // 2,)
        else:
            self.unpatchify_layer = Unpatchify(
                grid_size=(self.grid_x, self.grid_y),
                patch_size=(patch_size, patch_size),
                in_dim=dim,
                out_dim=out_channels,
                cond_dim=dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Re-zero-init AdaLN modulation outputs (apply overwrites them)
        for block in self.sa_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    @torch.no_grad()
    def get_grid(self, nlat, nlon, device):
        if self.with_poles:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat).to(device)
        else:
            lat_end = (nlat - 1) * (2 * math.pi / nlon) / 2
            lat = torch.linspace(-lat_end, lat_end, nlat).to(device)
        lon = torch.linspace(0, 2 * math.pi - (2 * math.pi / nlon), nlon).to(device)
        return lat, lon

    @torch.no_grad()
    def compute_rope_freqs(self, device):
        """Compute 2D RoPE cos/sin frequencies for the patch grid.

        Uses physical lat/lon coordinates at patch centers so the model
        encodes actual geographic position rather than integer indices.

        Returns cached buffers after first call.
        """
        if hasattr(self, '_rope_cos_lat') and self._rope_cos_lat.device == device:
            return (self._rope_cos_lat, self._rope_sin_lat,
                    self._rope_cos_lon, self._rope_sin_lon)

        # Get physical coordinates at padded resolution
        lat, lon = self.get_grid(self.nlat, self.nlon, device)

        # Average pool to patch centers: [nlat_pad] -> [grid_x], [nlon_pad] -> [grid_y]
        lat_patches = lat.reshape(self.grid_x, self.patch_size).mean(dim=1)  # [grid_x]
        lon_patches = lon.reshape(self.grid_y, self.patch_size).mean(dim=1)  # [grid_y]

        # Create 2D grid of patch positions and flatten to sequence
        # lat_grid[i,j] = lat of patch (i,j), lon_grid[i,j] = lon of patch (i,j)
        lat_grid, lon_grid = torch.meshgrid(lat_patches, lon_patches, indexing='ij')  # [grid_x, grid_y]
        lat_seq = lat_grid.reshape(-1)  # [n]
        lon_seq = lon_grid.reshape(-1)  # [n]

        # Compute RoPE frequencies: [n, dim_head//2] -> cos/sin each [1, n, dim_head//2]
        freqs_lat = self.rope_lat(lat_seq.unsqueeze(0))  # [1, n, dim_head//2]
        freqs_lon = self.rope_lon(lon_seq.unsqueeze(0))  # [1, n, dim_head//2]

        self._rope_cos_lat = freqs_lat.cos()
        self._rope_sin_lat = freqs_lat.sin()
        self._rope_cos_lon = freqs_lon.cos()
        self._rope_sin_lon = freqs_lon.sin()

        return (self._rope_cos_lat, self._rope_sin_lat,
                self._rope_cos_lon, self._rope_sin_lon)

    def forward(self, x_noised, t, cond,): # grid_cond, scalar_cond):
        """
        Args:
            x_noised: [b, c, nlat, nlon] — interpolant I_t (channel-first from assemble_input)
            cond: [b, c, zlat, zlon] — conditional information. Is the output of the VAE encoder
            grid_cond: [b c nlat nlon] — grid-scale conditional information (SST)
            scalar_cond: [b c] - scalar conditions (e.g. calendar, co2, etc.)
            t: [b, 1] — timestep + scalar conditions

        Returns:
            [b, c, nlat, nlon] — predicted velocity (channel-first)
        """

        #grid_embed = self.embed_grid(grid_cond) # [b, cond_dim, nlat, nlon]
        #scalar_embed = self.embed_calendar(scalar_cond) # [b, cond_dim, nlat, nlon]
        cond = F.interpolate(cond, size=(self.nlat, self.nlon), mode='bilinear', align_corners=False)  # [b, c, nlat, nlon]

        #x_input = torch.cat([x_noised, cond, grid_embed, scalar_embed], dim=1)  # [b, dim, nlat, nlon]
        x_input = torch.cat([x_noised, cond], dim=1)  # [b, dim, nlat, nlon]

        # Convert channel-first to channel-last for PatchEmbed: [b, c, h, w] -> [b, h, w, c]
        x_nhwc = x_input.permute(0, 2, 3, 1)

        # Patchify: [b, h, w, c] -> [b, h//p, w//p, dim]
        x = self.patch_embed_main(x_nhwc)

        # Flatten spatial dims for sequence processing: [b, h//p, w//p, dim] -> [b, n, dim]
        x = rearrange(x, 'b ny nx c -> b (ny nx) c')

        # Compute 2D RoPE frequencies for patch grid
        rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon = self.compute_rope_freqs(x.device)

        # Timestep embedding
        if len(t.shape) == 1:
            t = t[:, None]

        t_emb = self.t_embedder(t)  # [b, dim]

        # Transformer blocks with RoPE
        for sa_block in self.sa_blocks:
            x = sa_block(x, t_emb, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        # Unpatchify: [b, n, dim] -> [b, nlat, nlon, dim]
        x = self.unpatchify_layer(x, t_emb)

        # Convert back to channel-first: [b, h, w, c] -> [b, c, h, w]
        x = x.permute(0, 3, 1, 2)

        return x