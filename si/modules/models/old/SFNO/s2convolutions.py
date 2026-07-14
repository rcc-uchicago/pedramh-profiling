# flake8: noqa
# Copied from https://github.com/ai2cm/modulus/commit/22df4a9427f5f12ff6ac891083220e7f2f54d229
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import FactorizedTensor from tensorly for tensorized operations
import tensorly as tl
import torch
import torch.nn as nn
import torch.nn.functional as F

tl.set_backend("pytorch")
import torch_harmonics as th
import torch_harmonics.distributed as thd

# from tensorly.plugins import use_opt_einsum
# use_opt_einsum('optimal')
from tltorch.factorized_tensors.core import FactorizedTensor

# import convenience functions for factorized tensors
from modules.models.SFNO.activations import ComplexReLU

# for the experimental module
from modules.models.SFNO.contractions import (
    _contract_localconv_fwd,
    compl_exp_mul2d_fwd,
    compl_exp_muladd2d_fwd,
    compl_mul2d_fwd,
    compl_muladd2d_fwd,
    real_mul2d_fwd,
    real_muladd2d_fwd,
)
from modules.models.SFNO.factorizations import get_contract_fun

class SpectralRescale(nn.Module):
    def __init__(self, h, channels, momentum=.1, affine=True, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.affine=affine
        if affine:
            self.mags = nn.Parameter(torch.view_as_real(torch.ones(1, channels, 1, dtype=torch.cfloat)))
            self.bias = nn.Parameter(torch.view_as_real(torch.zeros(1, channels, 1, dtype=torch.cfloat)))
        self.register_buffer('mean', torch.view_as_real(torch.zeros(1, 1, h, dtype=torch.cfloat)))
        self.register_buffer('std', torch.ones(1, 1, h))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean, std = x.mean((0, 1), keepdims=True), x.std((0, 1), keepdims=True)
            with torch.no_grad():
                self.mean.copy_((1-self.momentum)*self.mean+self.momentum*torch.view_as_real(mean))
                self.std.copy_((1-self.momentum)*self.std+self.momentum*std)
        else:
            mean, std = torch.view_as_complex(self.mean), self.std

        x = (x - mean) / (self.eps + std)
        if self.affine:

            x = x * torch.view_as_complex(self.mags) + torch.view_as_complex(self.bias)
        return x

class mod_relu(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))
        #self.b.requiresGrad = True

    def forward(self, x):
        return torch.polar(F.gelu(torch.abs(x) + self.b), x.angle())
        #return F.relu(torch.abs(x) + self.b) * torch.exp(1.j * torch.angle(x)) 


def spectral_dfs_padding(x, padding, ew_padding=0):
    if padding==0:
        return x
    b, c, h, w = x.shape
    out = F.pad(x, (0, 0, padding, padding), 'reflect')
    phase_shift = torch.ones(w, device=x.device, dtype=x.dtype)
    phase_shift[1::2] *= -1
    out[:, :, :padding] *= phase_shift[None, None, None, :]
    out[:, :, -padding:] *= phase_shift[None, None, None, :]
    if ew_padding != 0:
        out = F.pad(out, (ew_padding, 0, 0, 0), 'reflect')
        out = F.pad(out, (0, ew_padding, 0, 0))
        out[:, :, :, :ew_padding] = out[:, :, :, :ew_padding].conj()
        #out = torch.cat([left, F.pad(out, (0, ew_padding, 0, 0))], -1)
    return out


class SlowComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__()
        padding_raw = [k//2 for k in kernel_size]
        self.padding = (padding_raw[1], padding_raw[1], padding_raw[0], padding_raw[0])
        self.groups = groups
        self.kernel_size = kernel_size
        temp_conv = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(temp_conv.weight))
        #print(temp_conv.weight.shape)
            
            
    def forward(self, x):
        x = spectral_dfs_padding(x, self.padding[-1], self.padding[0])
        return F.conv2d(x, torch.view_as_complex(self.weight), groups=self.groups)#F.fold(x, (h, w), (1, 1))

class SlowComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__()
        self.padding = kernel_size//2
        self.groups = groups
        self.kernel_size = kernel_size
        temp_conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups, dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(temp_conv.weight))
        #print(temp_conv.weight.shape)


    def forward(self, x):
        x = spectral_dfs_padding(x, self.padding)
        return F.conv1d(x, torch.view_as_complex(self.weight), groups=self.groups)#F.fold(x, (h, w), (1, 1))

def complex_glu(x, mag_bias, phase_bias):
    mag, phase = x.abs(), x.angle()
    return torch.polar(torch.sigmoid(mag+mag_bias), phase+phase_bias)

class DynamicSpectralConvS2(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        channels,
        bias=False,
    ):
        super().__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.lmax = inverse_transform.lmax
        self.mmax = inverse_transform.mmax
        self.channels = channels

        # Build mask for the triangular truncation region: m <= l < lmax
        ls = torch.arange(self.lmax).unsqueeze(1).expand(self.lmax, self.mmax)
        ms = torch.arange(self.mmax).unsqueeze(0).expand(self.lmax, self.mmax)
        useable = ms <= ls  # triangular region valid for SH
        self.register_buffer('useable_inds', useable)
        n_valid = useable.sum().item()

        # Learned static filter parameters (polar form)
        self.mags = nn.Parameter(torch.zeros(1, channels, n_valid))
        self.phases = nn.Parameter(torch.zeros(1, channels, n_valid))

        # Complex-valued MLP for dynamic offsets (matches paper)
        self.mlp = nn.Sequential(
            SlowComplexConv1d(channels, channels, 1),
            mod_relu(channels),
            SlowComplexConv1d(channels, channels, 1),
        )

        # Spectral normalization of input coefficients before MLP
        self.spectral_rescale = SpectralRescale(n_valid, channels)

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # x: (B, C, nlat, nlon)
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        with torch.amp.autocast("cuda", enabled=False):
            coeffs = self.forward_transform(x)  # (B, C, lmax, mmax) complex

        # Mask to valid triangular region and flatten
        # coeffs[..., useable_inds] -> (B, C, n_valid)
        x_masked = torch.masked_select(
            coeffs, self.useable_inds[None, None, :, :]
        ).reshape(B, C, -1)

        # Dynamic filter generation 
        x_linear = self.mlp(self.spectral_rescale(x_masked))
        conv = complex_glu(x_linear, self.mags, self.phases)

        # Apply filter (linear operation — no new frequencies)
        x_masked = x_masked * conv

        # Scatter back to full (lmax, mmax) coefficient array
        coeffs = torch.zeros_like(coeffs)
        coeffs[:, :, self.useable_inds] = x_masked

        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(coeffs)

        if hasattr(self, 'bias'):
            x = x + self.bias

        return x.to(dtype)

class SpectralConvS2(nn.Module):
    """
    Spectral Convolution according to Driscoll & Healy. Designed for convolutions on
    the two-sphere S2 using the Spherical Harmonic Transforms in torch-harmonics, but
    supports convolutions on the periodic domain via the RealFFT2 and InverseRealFFT2
    wrappers.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        scale="auto",
        operator_type="diagonal",
        rank=0.2,
        factorization=None,
        separable=False,
        decomposition_kwargs=dict(),
        bias=False,
        use_tensorly=True,
    ):  # pragma: no cover
        super(SpectralConvS2, self).__init__()

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (
            (self.forward_transform.nlat != self.inverse_transform.nlat)
            or (self.forward_transform.nlon != self.inverse_transform.nlon)
            or (self.forward_transform.grid != self.inverse_transform.grid)
        )

        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = "Dense"  # No factorization

        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        # remember factorization details
        self.operator_type = operator_type
        self.rank = rank
        self.factorization = factorization
        self.separable = separable

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        weight_shape = [in_channels]

        if not self.separable:
            weight_shape += [out_channels]

        if isinstance(self.inverse_transform, thd.DistributedInverseRealSHT):
            self.modes_lat_local = self.inverse_transform.lmax_local
            self.modes_lon_local = self.inverse_transform.mmax_local
            self.lpad_local = self.inverse_transform.lpad_local
            self.mpad_local = self.inverse_transform.mpad_local
        else:
            self.modes_lat_local = self.modes_lat
            self.modes_lon_local = self.modes_lon
            self.lpad = 0
            self.mpad = 0

        # padded weights
        # if self.operator_type == 'diagonal':
        #     weight_shape += [self.modes_lat_local+self.lpad_local, self.modes_lon_local+self.mpad_local]
        # elif self.operator_type == 'dhconv':
        #     weight_shape += [self.modes_lat_local+self.lpad_local]
        # else:
        #     raise ValueError(f"Unsupported operator type f{self.operator_type}")

        # unpadded weights
        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat_local, self.modes_lon_local]
        elif self.operator_type == "dhconv":
            weight_shape += [self.modes_lat_local]
        else:
            raise ValueError(f"Unsupported operator type f{self.operator_type}")

        if use_tensorly:
            # form weight tensors
            self.weight = FactorizedTensor.new(
                weight_shape,
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=False,
                **decomposition_kwargs,
            )
            # initialization of weights
            self.weight.normal_(0, scale)
        else:
            assert factorization == "ComplexDense"
            self.weight = nn.Parameter(scale * torch.randn(*weight_shape, 2))
            if self.operator_type == "dhconv":
                self.weight.is_shared_mp = ["matmul", "w"]
            else:
                self.weight.is_shared_mp = ["matmul"]

        # get the contraction handle
        self._contract = get_contract_fun(
            self.weight, implementation="factorized", separable=separable
        )

        if bias:
            self.bias = nn.Parameter(scale * torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):  # pragma: no cover
        dtype = x.dtype
        residual = x
        x = x.float()
        B, C, H, W = x.shape

        with torch.amp.autocast("cuda", enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        # approach with unpadded weights
        xp = torch.zeros_like(x)
        xp[..., : self.modes_lat_local, : self.modes_lon_local] = self._contract(
            x[..., : self.modes_lat_local, : self.modes_lon_local],
            self.weight,
            separable=self.separable,
            operator_type=self.operator_type,
        )
        x = xp.contiguous()

        # # approach with padded weights
        # x = self._contract(x, self.weight, separable=self.separable, operator_type=self.operator_type)
        # x = x.contiguous()

        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.type(dtype)

        return x, residual


class LocalConvS2(nn.Module):
    """
    S2 Convolution according to Driscoll & Healy
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        nradius=120,
        scale="auto",
        bias=False,
    ):  # pragma: no cover
        super(LocalConvS2, self).__init__()

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nradius = nradius

        self.forward_transform = forward_transform
        self.zonal_transform = th.RealSHT(
            forward_transform.nlat,
            1,
            lmax=forward_transform.lmax,
            mmax=1,
            grid=forward_transform.grid,
        ).float()
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax
        self.output_dims = (self.inverse_transform.nlat, self.inverse_transform.nlon)

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, nradius, 1)
        )

        self._contract = _contract_localconv_fwd

        if bias:
            self.bias = nn.Parameter(
                scale * torch.randn(1, out_channels, *self.output_dims)
            )

    def forward(self, x):  # pragma: no cover
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        with torch.amp.autocast("cuda", enabled=False):
            f = torch.zeros(
                (self.in_channels, self.out_channels, H, 1),
                dtype=x.dtype,
                device=x.device,
            )
            f[..., : self.nradius, :] = self.weight

            x = self.forward_transform(x)
            f = self.zonal_transform(f)[..., :, 0]

            x = torch.view_as_real(x)
            f = torch.view_as_real(f)

        x = self._contract(x, f)
        x = x.contiguous()

        x = torch.view_as_complex(x)

        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.type(dtype)

        return x


class SpectralAttentionS2(nn.Module):
    """
    Spherical non-linear FNO layer
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        operator_type="diagonal",
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        complex_activation="real",
        scale="auto",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralAttentionS2, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.operator_type = operator_type
        self.spectral_layers = spectral_layers

        if scale == "auto":
            self.scale = 1 / (embed_dim * embed_dim)

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        hidden_size = int(hidden_size_factor * self.embed_dim)

        if operator_type == "diagonal":
            self.mul_add_handle = compl_muladd2d_fwd
            self.mul_handle = compl_mul2d_fwd

            # weights
            w = [self.scale * torch.randn(self.embed_dim, hidden_size, 2)]
            for l in range(1, self.spectral_layers):
                w.append(self.scale * torch.randn(hidden_size, hidden_size, 2))
            self.w = nn.ParameterList(w)

            self.wout = nn.Parameter(
                self.scale * torch.randn(hidden_size, self.embed_dim, 2)
            )

            if bias:
                self.b = nn.ParameterList(
                    [
                        self.scale * torch.randn(hidden_size, 1, 1, 2)
                        for _ in range(self.spectral_layers)
                    ]
                )

            self.activations = nn.ModuleList([])
            for l in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(
                        mode=complex_activation,
                        bias_shape=(hidden_size, 1, 1),
                        scale=self.scale,
                    )
                )

        elif operator_type == "l-dependant":
            self.mul_add_handle = compl_exp_muladd2d_fwd
            self.mul_handle = compl_exp_mul2d_fwd

            # weights
            w = [
                self.scale * torch.randn(self.modes_lat, self.embed_dim, hidden_size, 2)
            ]
            for l in range(1, self.spectral_layers):
                w.append(
                    self.scale
                    * torch.randn(self.modes_lat, hidden_size, hidden_size, 2)
                )
            self.w = nn.ParameterList(w)

            if bias:
                self.b = nn.ParameterList(
                    [
                        self.scale * torch.randn(hidden_size, 1, 1, 2)
                        for _ in range(self.spectral_layers)
                    ]
                )

            self.wout = nn.Parameter(
                self.scale * torch.randn(self.modes_lat, hidden_size, self.embed_dim, 2)
            )

            self.activations = nn.ModuleList([])
            for l in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(
                        mode=complex_activation,
                        bias_shape=(hidden_size, 1, 1),
                        scale=self.scale,
                    )
                )

        else:
            raise ValueError("Unknown operator type")

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward_mlp(self, x):  # pragma: no cover
        """forward pass of the MLP"""
        B, C, H, W = x.shape

        if self.operator_type == "block-separable":
            x = x.permute(0, 3, 1, 2)

        xr = torch.view_as_real(x)

        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(xr, self.w[l], self.b[l])
            else:
                xr = self.mul_handle(xr, self.w[l])
            xr = torch.view_as_complex(xr)
            xr = self.activations[l](xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)

        # final MLP
        x = self.mul_handle(xr, self.wout)

        x = torch.view_as_complex(x)

        if self.operator_type == "block-separable":
            x = x.permute(0, 2, 3, 1)

        return x

    def forward(self, x):  # pragma: no cover
        dtype = x.dtype
        residual = x
        x = x.to(torch.float32)

        # FWD transform
        with torch.amp.autocast("cuda", enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        x = x.contiguous()
        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        # cast back to initial precision
        x = x.to(dtype)

        return x, residual


class RealSpectralAttentionS2(nn.Module):
    """
    Non-linear SFNO layer using a real-valued NN instead of a complex one
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        operator_type="diagonal",
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        complex_activation="real",
        scale="auto",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(RealSpectralAttentionS2, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.operator_type = operator_type
        self.spectral_layers = spectral_layers

        if scale == "auto":
            self.scale = 1 / (embed_dim * embed_dim)

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        hidden_size = int(hidden_size_factor * self.embed_dim * 2)

        self.mul_add_handle = real_muladd2d_fwd
        self.mul_handle = real_mul2d_fwd

        # weights
        w = [self.scale * torch.randn(2 * self.embed_dim, hidden_size)]
        for l in range(1, self.spectral_layers):
            w.append(self.scale * torch.randn(hidden_size, hidden_size))
        self.w = nn.ParameterList(w)

        self.wout = nn.Parameter(
            self.scale * torch.randn(hidden_size, 2 * self.embed_dim)
        )

        if bias:
            self.b = nn.ParameterList(
                [
                    self.scale * torch.randn(hidden_size, 1, 1)
                    for _ in range(self.spectral_layers)
                ]
            )

        self.activations = nn.ModuleList([])
        for l in range(0, self.spectral_layers):
            self.activations.append(nn.ReLU())

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward_mlp(self, x):  # pragma: no cover
        """forward pass of the MLP"""
        B, C, H, W = x.shape

        xr = torch.view_as_real(x)
        xr = xr.permute(0, 1, 4, 2, 3).reshape(B, C * 2, H, W)

        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(xr, self.w[l], self.b[l])
            else:
                xr = self.mul_handle(xr, self.w[l])
            xr = self.activations[l](xr)
            xr = self.drop(xr)

        # final MLP
        xr = self.mul_handle(xr, self.wout)

        xr = xr.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2)

        x = torch.view_as_complex(xr)

        return x

    def forward(self, x):  # pragma: no cover
        dtype = x.dtype
        x = x.to(torch.float32)

        # FWD transform
        with torch.amp.autocast("cuda", enabled=False):
            x = self.forward_transform(x)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        # cast back to initial precision
        x = x.to(dtype)

        return x