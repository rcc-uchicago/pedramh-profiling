import math
import torch
from torch import randn, Generator
from torch.nn import Module

class Perturber(Module):

    def __init__(self, params, dataset, device = "cpu", device_idx = 0, seed = None):
        super().__init__()
        self.device = device
        self.generator = Generator(device = device)
        if seed is not None:
            self.generator.manual_seed(seed + device_idx)
        self.upper_air_variables = dataset.upper_air_variables
        self.surface_variables = dataset.surface_variables
        self.levels = params.levels
            
        self.epsilon_factor = params.epsilon_factor
        self.surface_std = dataset.surface_std.to(device)
        self.upper_air_std = dataset.upper_air_std.to(device)
        if 'surface_ff_std' in params and 'upper_air_ff_std' in params:
            self.use_residual_norm = True
            self.surface_ff_std = dataset.surface_ff_std.to(device)
            self.upper_air_ff_std = dataset.upper_air_ff_std.to(device)
        else:
            self.use_residual_norm = False
        
        if params.perturbation_type == 'gaussian_noise':
            self.perturb_method = self.add_gaussian_noise
        elif params.perturbation_type == 'gaussian_noise_n_minus_1':
            # Number of ensemble members per prediction pass; first member is unperturbed.
            self.n_ens_members = getattr(params, 'ensemble_members_per_pred',
                                         getattr(params, 'num_ensemble_members', 1))
            self.perturb_method = self.add_gaussian_noise_n_minus_1
        elif params.perturbation_type == 'perlin_noise':
            # period_number: number of base-frequency cycles across the spatial domain.
            # octaves: number of doublings of frequency to superimpose.
            # persistence: amplitude scale per octave (< 1 → high-freq rolloff).
            self.octaves = getattr(params, 'octaves', 8)
            self.period_number = getattr(params, 'period_number', 4)
            self.persistence = getattr(params, 'persistence', 0.5)
            self.perturb_method = self.add_perlin_noise
            
    def forward(self, surface_t, upper_air_t):
        return self.perturb_method(surface_t, upper_air_t)
        
    def add_gaussian_noise(self, surface_t, upper_air_t):
        if self.use_residual_norm:
            surface_t_noise = randn(*surface_t.shape, generator=self.generator, device = self.device) * (self.epsilon_factor * self.surface_ff_std / self.surface_std).reshape(len(self.surface_variables), 1, 1)
        else:
            surface_t_noise = randn(*surface_t.shape, generator=self.generator, device = self.device) * self.epsilon_factor
        surface_t = surface_t + surface_t_noise
        if self.use_residual_norm:
            upper_air_t_noise = randn(*upper_air_t.shape, generator=self.generator, device = self.device) * (self.epsilon_factor * self.upper_air_ff_std / self.upper_air_std).reshape(len(self.upper_air_variables), len(self.levels), 1, 1)
        else:
            upper_air_t_noise = randn(*upper_air_t.shape, generator=self.generator, device = self.device) * self.epsilon_factor
        upper_air_t = upper_air_t + upper_air_t_noise
        return surface_t, upper_air_t

    def add_gaussian_noise_n_minus_1(self, surface_t, upper_air_t):
        """Gaussian noise on all ensemble members except the first.

        In ensemble mode the batch dimension is (batch_size * n_ens_members,),
        with members laid out via repeat_interleave:
            [particle_0_member_0, particle_0_member_1, ...,
             particle_1_member_0, particle_1_member_1, ...]
        Member 0 of each particle sits at indices 0, n_ens, 2*n_ens, …
        Those rows have their noise zeroed so they remain unperturbed.
        """
        if self.use_residual_norm:
            surface_t_noise = randn(*surface_t.shape, generator=self.generator, device=self.device) * \
                (self.epsilon_factor * self.surface_ff_std / self.surface_std).reshape(len(self.surface_variables), 1, 1)
        else:
            surface_t_noise = randn(*surface_t.shape, generator=self.generator, device=self.device) * self.epsilon_factor
        surface_t_noise[::self.n_ens_members] = 0.0
        surface_t = surface_t + surface_t_noise

        if self.use_residual_norm:
            upper_air_t_noise = randn(*upper_air_t.shape, generator=self.generator, device=self.device) * \
                (self.epsilon_factor * self.upper_air_ff_std / self.upper_air_std).reshape(len(self.upper_air_variables), len(self.levels), 1, 1)
        else:
            upper_air_t_noise = randn(*upper_air_t.shape, generator=self.generator, device=self.device) * self.epsilon_factor
        upper_air_t_noise[::self.n_ens_members] = 0.0
        upper_air_t = upper_air_t + upper_air_t_noise
        return surface_t, upper_air_t

    def _compute_power_spectrum(self, H, W):
        """Return a Perlin-like power spectrum tensor of shape (H, W//2+1).

        The spectrum is the sum of *octaves* Gaussian bumps.  Octave k is
        centred at frequency ``f_k = period_number * 2^k / H`` (cycles per
        grid cell) with amplitude ``persistence^k``.  The Gaussian width is
        half the octave centre frequency so adjacent octaves blend smoothly.
        """
        fy = torch.fft.fftfreq(H, device=self.device)   # (H,)
        fx = torch.fft.rfftfreq(W, device=self.device)  # (W//2+1,)
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')  # (H, W//2+1)
        freq = torch.sqrt(FY ** 2 + FX ** 2)            # radial frequency

        power = torch.zeros_like(freq)
        base_freq = self.period_number / H
        for k in range(self.octaves):
            f_k = base_freq * (2.0 ** k)
            sigma_k = f_k / 2.0 if f_k > 0 else 1e-6
            amplitude = self.persistence ** k
            power = power + amplitude * torch.exp(-0.5 * ((freq - f_k) / sigma_k) ** 2)
        return power

    def add_perlin_noise(self, surface_t, upper_air_t):
        """Add spatially-correlated (Perlin-like) noise scaled by epsilon_factor.

        Noise is generated in frequency space via random phases × sqrt(power
        spectrum), inverse-FFT'd, and normalised to unit spatial variance
        before scaling.  Each field in the batch is drawn independently.
        """
        H, W = surface_t.shape[-2], surface_t.shape[-1]
        sqrt_power = torch.sqrt(self._compute_power_spectrum(H, W))
        spectral_W = W // 2 + 1

        def _colored_noise(shape):
            n = math.prod(shape[:-2])
            phases = 2.0 * math.pi * torch.rand(
                n, H, spectral_W, generator=self.generator, device=self.device)
            noise_spectral = torch.complex(
                sqrt_power * torch.cos(phases),
                sqrt_power * torch.sin(phases),
            )
            noise = torch.fft.irfft2(noise_spectral, s=(H, W))  # (n, H, W)
            noise = noise / (noise.std(dim=(-2, -1), keepdim=True) + 1e-8)
            return noise.reshape(shape)

        if self.use_residual_norm:
            scale_s = (self.epsilon_factor * self.surface_ff_std / self.surface_std).reshape(
                len(self.surface_variables), 1, 1)
        else:
            scale_s = self.epsilon_factor
        surface_t = surface_t + _colored_noise(surface_t.shape) * scale_s

        if self.use_residual_norm:
            scale_u = (self.epsilon_factor * self.upper_air_ff_std / self.upper_air_std).reshape(
                len(self.upper_air_variables), len(self.levels), 1, 1)
        else:
            scale_u = self.epsilon_factor
        upper_air_t = upper_air_t + _colored_noise(upper_air_t.shape) * scale_u

        return surface_t, upper_air_t