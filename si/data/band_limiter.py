import torch
import torch.nn as nn
from torch_harmonics import RealSHT, InverseRealSHT, RealVectorSHT, InverseRealVectorSHT

class SphericalBandLimitVector(nn.Module):
    def __init__(self, nlat: int, nlon: int, K: int, grid: str = "equiangular",
                 surface_uv_idx=(4, 5),
                 multilevel_uv_idx=(1, 2),
                 nsurface=6,
                 nmulti=5,
                 nlevels=26):
        """
        K: max spherical harmonic degree to keep (exclusive).
           Coefficients with l >= K are zeroed out.

        Wind (u/v) channels are filtered via VectorSHT; all other channels
        use the scalar SHT.

        surface_uv_idx: tuple of 2 indices into the surface variable dim (u, v)
        multilevel_uv_idx: tuple of 2 indices into the multilevel variable dim (u, v)
        nsurface: number of surface variables
        nmulti: number of multilevel variables per level
        nlevels: number of pressure levels
        """
        super().__init__()
        self.K = K
        self.nsurface = nsurface
        self.nmulti = nmulti
        self.nlevels = nlevels

        # Scalar SHT (truncated to K modes)
        self.sht  = RealSHT(nlat, nlon, lmax=K, mmax=K, grid=grid)
        self.isht = InverseRealSHT(nlat, nlon, lmax=K, mmax=K, grid=grid)

        # Vector SHT (truncated to K modes)
        self.vsht  = RealVectorSHT(nlat, nlon, lmax=K, mmax=K, grid=grid)
        self.ivsht = InverseRealVectorSHT(nlat, nlon, lmax=K, mmax=K, grid=grid)

        # Wind index bookkeeping
        self.surface_uv_idx = list(surface_uv_idx)
        self.surface_scalar_idx = [i for i in range(nsurface) if i not in surface_uv_idx]
        self.multilevel_uv_idx = list(multilevel_uv_idx)
        self.multilevel_scalar_idx = [i for i in range(nmulti) if i not in multilevel_uv_idx]

    def _scalar_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Band-limit via scalar SHT. x: (N, nlat, nlon)"""
        return self.isht(self.sht(x))

    def _vector_filter(self, uv: torch.Tensor) -> torch.Tensor:
        """Band-limit via vector SHT. uv: (N, 2, nlat, nlon)"""
        return self.ivsht(self.vsht(uv))

    def forward(self, surface: torch.Tensor, multilevel: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        surface:    (B, nsurface, nlat, nlon)
        multilevel: (B, nmulti, nlevels, nlat, nlon)

        Returns filtered tensors with the same shapes.
        """
        B, _, nlat, nlon = surface.shape

        # --- Surface ---
        # Scalar channels
        surf_scalar = surface[:, self.surface_scalar_idx]             # (B, C_s, nlat, nlon)
        surf_scalar = self._scalar_filter(
            surf_scalar.reshape(-1, nlat, nlon)
        ).reshape(B, len(self.surface_scalar_idx), nlat, nlon)

        # Wind channels
        surf_uv = surface[:, self.surface_uv_idx]                    # (B, 2, nlat, nlon)
        surf_uv = self._vector_filter(surf_uv)                       # (B, 2, nlat, nlon)

        # Reassemble in original channel order
        surface_out = torch.empty_like(surface)
        for i, idx in enumerate(self.surface_scalar_idx):
            surface_out[:, idx] = surf_scalar[:, i]
        for i, idx in enumerate(self.surface_uv_idx):
            surface_out[:, idx] = surf_uv[:, i]

        # --- Multilevel ---
        # Scalar channels
        ml_scalar = multilevel[:, self.multilevel_scalar_idx]         # (B, C_m, nlevels, nlat, nlon)
        C_m = len(self.multilevel_scalar_idx)
        ml_scalar = self._scalar_filter(
            ml_scalar.reshape(-1, nlat, nlon)
        ).reshape(B, C_m, self.nlevels, nlat, nlon)

        # Wind channels: (B, 2, nlevels, nlat, nlon) -> (B*nlevels, 2, nlat, nlon)
        ml_uv = multilevel[:, self.multilevel_uv_idx]                # (B, 2, nlevels, nlat, nlon)
        ml_uv = ml_uv.permute(0, 2, 1, 3, 4).reshape(B * self.nlevels, 2, nlat, nlon)
        ml_uv = self._vector_filter(ml_uv)
        ml_uv = ml_uv.reshape(B, self.nlevels, 2, nlat, nlon).permute(0, 2, 1, 3, 4)  # (B, 2, nlevels, nlat, nlon)

        # Reassemble in original channel order
        multilevel_out = torch.empty_like(multilevel)
        for i, idx in enumerate(self.multilevel_scalar_idx):
            multilevel_out[:, idx] = ml_scalar[:, i]
        for i, idx in enumerate(self.multilevel_uv_idx):
            multilevel_out[:, idx] = ml_uv[:, i]

        return surface_out, multilevel_out



class SphericalBandLimitScalar(nn.Module):
    def __init__(self, nlat: int, nlon: int, K: int, grid: str = "equiangular"):
        """
        K: max spherical harmonic degree to keep (exclusive).
           Coefficients with l >= K are zeroed out.

        Same interface as SphericalBandLimitVector, but all channels
        (including wind u/v) are filtered via the scalar SHT.
        """
        super().__init__()
        lmax = nlat
        mmax = nlon // 2 + 1
        self.K = K

        # Scalar SHT (truncated to K modes)
        self.sht  = RealSHT(nlat, nlon, lmax=K, mmax=K, grid=grid)
        self.isht = InverseRealSHT(nlat, nlon, lmax=K, mmax=K, grid=grid)

    def _scalar_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Band-limit via scalar SHT. x: (N, nlat, nlon)"""
        return self.isht(self.sht(x))

    def forward(self, surface: torch.Tensor, multilevel: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        surface:    (B, nsurface, nlat, nlon)
        multilevel: (B, nmulti, nlevels, nlat, nlon)

        Returns filtered tensors with the same shapes.
        All channels are filtered via scalar SHT.
        """
        B, C_s, nlat, nlon = surface.shape

        # --- Surface ---
        surface_out = self._scalar_filter(
            surface.reshape(-1, nlat, nlon)
        ).reshape(B, C_s, nlat, nlon)

        # --- Multilevel ---
        B, C_m, L, nlat, nlon = multilevel.shape
        multilevel_out = self._scalar_filter(
            multilevel.reshape(-1, nlat, nlon)
        ).reshape(B, C_m, L, nlat, nlon)

        return surface_out, multilevel_out