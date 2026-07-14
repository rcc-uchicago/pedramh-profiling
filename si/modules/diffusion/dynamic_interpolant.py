import torch
import torch.nn as nn
from modules.diffusion.utils import SphereNoiseGenerator, sample_logit_normal

class Integrator:
    def __init__(self,
                 method='em',
                 ):
        self.method = method

    def step_fn(self, y, drift, dt, noise_t, generator=None):
        # g_t: (b,) scalar sigma per sample; broadcast per-tensor to handle different ndims
        if self.method == 'em':  # Euler-Maruyama
            if generator is not None:
                noise = generator(y.shape[0], y.shape[1], device=y.device)
            else:
                noise = torch.randn_like(y)
            dW = torch.sqrt(dt) * noise
            y_next = y + drift * dt + noise_t * dW
        elif self.method == 'euler':  # ODE
            y_next = y + drift * dt
        return y_next

    def integrate(self,
                  y, cond, c_grid,
                  model, timesteps, noise_fn,
                  generator=None):

        # y is current state along interpolant (noised prognostic states)
        # cond is conditioning (current prognostic state)
        # c_grid is grid-scale conditioning (forcings + invariants, original resolution)

        for i_t in range(len(timesteps) - 1):
            t_current = timesteps[i_t]
            t_next = timesteps[i_t + 1]
            dt = t_next - t_current
            noise_t = noise_fn(t_current.expand(y.shape[0]))  # shape (b, 1, 1, 1)

            scalar_in = t_current.float().expand(y.shape[0]).unsqueeze(-1)

            drift = model(y, cond, scalar_in, c_grid)

            y = self.step_fn(y, drift, dt, noise_t, generator)

        return y

class DriftScheduler(nn.Module):
    def __init__(self,
                 num_refinement_steps,  # this corresponds to physical time steps
                 num_train_steps=None,  # number of training steps
                 integrator='em',
                 sigma_coef=1.0,
                 beta_fn="t",
                 antithetic_sampling=False,
                 sigma_sample=None,
                 l_max=None,
                 train_sampler='discrete'
                 ):
        super(DriftScheduler, self).__init__()

        self.num_train_timesteps = num_train_steps if num_train_steps is not None else num_refinement_steps + 1
        self.num_refinement_steps = num_refinement_steps
        self.sigma_coef = sigma_coef
        self.method = integrator
        self.integrator = Integrator(method=integrator)

        self.beta_fn = beta_fn
        self.antithetic_sampling = antithetic_sampling
        self.sigma_sample = sigma_sample if sigma_sample is not None else sigma_coef
        self.train_sampler = train_sampler

        if l_max is not None:
            self.generator = SphereNoiseGenerator(l_max=l_max)
        else:
            self.generator = None

        print(f'Scheduler initialized with {self.num_train_timesteps} training steps and {self.num_refinement_steps} refinement steps.')
        print(f"sigma_coef: {self.sigma_coef}, integrator: {integrator}, beta_fn: {self.beta_fn}, antithetic_sampling: {self.antithetic_sampling}, train_sampler: {self.train_sampler}")

    def wide(self, t, ndim=2):
        if ndim == 2:
            return t[:, None, None, None]
        elif ndim == 3:
            return t[:, None, None, None, None]

    def alpha(self, t, ndim=2):
        return self.wide(1 - t, ndim)

    def alpha_dot(self, t, ndim=2):
        return self.wide(-1.0 * torch.ones_like(t), ndim)

    def beta(self, t, ndim=2):
        if self.beta_fn == "t":
            return self.wide(t, ndim)
        elif self.beta_fn == "t^2":
            return self.wide(t ** 2, ndim)

    def beta_dot(self, t, ndim=2):
        if self.beta_fn == "t":
            return self.wide(torch.ones_like(t), ndim)
        elif self.beta_fn == "t^2":
            return self.wide(2.0 * t, ndim)

    def sigma(self, t, sample=False, ndim=2):
        if sample:
            return self.sigma_sample * self.wide(1 - t, ndim)
        else:
            return self.sigma_coef * self.wide(1 - t, ndim)

    def sigma_dot(self, t, sample=False, ndim=2):
        if sample:
            return self.sigma_sample * self.wide(-1.0 * torch.ones_like(t), ndim)
        else:
            return self.sigma_coef * self.wide(-1.0 * torch.ones_like(t), ndim)

    def I(self, x0, x1, t, ndim=2):
        return self.alpha(t, ndim) * x0 + self.beta(t, ndim) * x1

    def dIdt(self, x0, x1, t, ndim=2):
        return self.alpha_dot(t, ndim) * x0 + self.beta_dot(t, ndim) * x1

    def get_noise(self, x):
        if self.generator is not None:
            return self.generator(x.shape[0], x.shape[1], device=x.device)
        else:
            return torch.randn(x.shape, device=x.device, dtype=x.dtype)
    
    def image_sq_norm(self, x):
        return x.pow(2).sum(-1).sum(-1).sum(-1)

    def compute_loss(self, model, x, c_grid, y):
        # x contains current prognostic state
        # c_grid contains current forcing state
        # y contains next prognostic state 

        device = x.device

        noise = self.get_noise(x)
        # sample timestep
        if self.train_sampler == 'logit_normal':
            t = sample_logit_normal(x.shape[0], device=device)
        elif self.train_sampler == 'uniform':
            t = torch.rand(x.shape[0], device=device)
        else:  # discrete
            t = torch.randint(0, self.num_train_timesteps - 1, device=device, size=(x.shape[0],)).float() / (self.num_train_timesteps - 1)

        sigma_t = self.sigma(t)          # shape (b, 1, 1, 1)
        sigma_dot_t = self.sigma_dot(t)  # shape (b, 1, 1, 1)
        W_t = self.wide(torch.sqrt(t))   # shape (b, 1, 1, 1)

        I = self.I(x, y, t)  # shape (b, d, nx, ny)
        dIdt = self.dIdt(x, y, t)  # shape (b, d, nx, ny)

        if self.antithetic_sampling:
            raise NotImplementedError("Antithetic sampling not implemented yet.")
        else:
            I_noised = I + sigma_t * W_t * noise
            target = dIdt + sigma_dot_t * W_t * noise

            pred = model(I_noised, x, t.view(-1, 1), c_grid)

            loss= self.image_sq_norm(pred - target).mean()

        return loss

    def sample(self, model, x, c_grid, refinement_steps=None):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)

        if refinement_steps is None:
            refinement_steps = self.num_refinement_steps

        timesteps = torch.linspace(0, 1, refinement_steps + 1, device=x.device)

        # start y at source distribution, which is current state
        y = x.clone()

        # conditioning is the current prognostic state (latent)
        cond = x

        # first step taken analytically to avoid g_T singularity issues at t=0 with EM
        sigma_0 = self.sigma(timesteps[0].expand(x.shape[0]), sample=True)  # shape (b, 1, 1, 1)
        dt_0 = timesteps[1] - timesteps[0]
        scalar_in = timesteps[0].float().expand(x.shape[0]).unsqueeze(-1)

        drift = model(y, cond, scalar_in, c_grid)

        if self.method == 'em':
            dW = torch.sqrt(dt_0)
            y = y + drift * dt_0 + sigma_0 * dW * torch.randn_like(y)
        else:
            y = y + drift * dt_0

        noise_fn = lambda t: self.sigma(t, sample=True)
        y = self.integrator.integrate(y, cond, c_grid, model, timesteps[1:], noise_fn, self.generator)

        return y

    def forward(self, model, x, c_grid, refinement_steps=None):
        return self.sample(model, x, c_grid, refinement_steps)
