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
        elif params.perturbation_type == 'perlin_noise': #Not yet implemented
            self.octaves = params.octaves
            self.period_number = params.period_number
            self.persistence = params.persistence
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