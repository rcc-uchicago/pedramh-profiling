import torch
import torch.nn as nn
from modules.diffusion.utils import sample_logit_normal, power_sampler, get_log_uniform_t

class FlowMatching(nn.Module):
    def __init__(self,
                 num_steps,  # this corresponds to physical time steps
                 sigma_coef=1.0,
                 train_sampler='logit_normal',
                 l_max = 180,
                 spectral_weight = 0.00,
                 noise = "spherical",
                 model_last = False,
                 noise_scale_path = None,
                 loss_form = "x",
                 tau = 1.1
                 ):
        super(FlowMatching, self).__init__()

        self.num_steps = num_steps
        self.sigma_coef = sigma_coef
        self.train_sampler = train_sampler 
        self.model_last = model_last
        self.loss_form = loss_form
        self.tau = tau

        if noise == "spherical":
            from modules.diffusion.utils import SphereNoiseGenerator
            self.generator = SphereNoiseGenerator(l_max=l_max)
        else:
            self.generator = None

        self.spectral_weight = spectral_weight

        if self.spectral_weight > 0: # apply  spectral regularization to model outputs
            from common.loss import SpectralScalarLoss
            self.spectral_criterion = SpectralScalarLoss(img_shape=(l_max, l_max*2))

        if noise_scale_path is not None:
            noise_scales = torch.load(noise_scale_path) # 1 c 1 1 
            self.register_buffer("noise_scales", noise_scales)
        else:
            self.noise_scales = None

        print(f"sigma_coef: {self.sigma_coef}, train_sampler: {self.train_sampler}")

    def wide(self, t):
        return t[:, None, None, None]
    
    def get_noise(self, x):
        if self.generator is None:
            return torch.randn_like(x, device=x.device)
        else:
            return self.generator(x.shape[0], x.shape[1], device=x.device)

    def compute_loss(self, model, x, c_grid, y):
        # x contains current prognostic state
        # c_grid contains current forcing state
        # y contains next prognostic state 

        device = x.device

        noise = self.get_noise(x)

        if self.noise_scales is not None:
            noise = noise * self.noise_scales 

        # sample timestep
        if self.train_sampler == 'logit_normal':
            t = sample_logit_normal(x.shape[0], device=device)
        elif self.train_sampler == 'power':
            t = power_sampler(x.shape[0], p=2.0, device=device)
        elif self.train_sampler == 'uniform':
            t = torch.rand(x.shape[0], device=device)

        t = self.wide(t) 

        X_t = (1-t) * noise + t * y 

        pred_y = model(X_t, x, t.squeeze(dim=[2, 3]), c_grid)

        if self.loss_form == 'x':
            loss = ((pred_y - y) ** 2).sum(dim=[1, 2, 3]).mean() 
        elif self.loss_form == 'v': # same as x loss, but scaled by 1-t
            target = (y - X_t) / (1 - t + 5e-2)
            pred = (pred_y - X_t) / (1 - t + 5e-2)
            loss = ((pred - target) ** 2).sum(dim=[1, 2, 3]).mean()

        if self.spectral_weight > 0:
            spectral_loss = self.spectral_weight * self.spectral_criterion(pred_y, y)
        else:
            spectral_loss = 0

        loss = loss + spectral_loss

        return loss, spectral_loss
    
    @torch.no_grad()
    def sample(self, model, x, c_grid, num_steps=None):
        """
        Forward Euler ODE integration. Reparameterized for stability and x-prediction

        Draw x_0 ~ N(0, I)
        define ratio r
        define dt_k = (1-r) / (1-t_k)
        this simplifies the Euler update to:
            x_{t+1} = r*x_t + (1-r)x_1

        originally:
            v_hat = (\hat x_1 - x_t) / (1 - t)
            x_{t+1} = x_t + (1-r) / (1-t_k) * v_hat => x_t + (1-r)(x_1 - x_t) => r * x_t + (1-r)x_1

        Args:
            x_lowres: [b, c, h, w] — m(x1), upsampled low-res conditioning
            model: predictor
            num_steps: number of integration steps N 

        Returns:
            [b, c, h, w] predicted high-res output
        """

        if num_steps is None:
            num_steps = self.num_steps

        # Starting point: X_0 = eps
        y = self.get_noise(x)

        if self.noise_scales is not None:
            y = y * self.noise_scales

        timesteps, ratio = get_log_uniform_t(n_t = num_steps - 1, scale=self.tau,  device = x.device)
        
        ratio_batch = ratio.expand(x.shape[0], 1, 1, 1)

        if self.model_last:
            num_steps_euler = num_steps - 1
        else:
            num_steps_euler = num_steps

        for k in range(num_steps_euler):
            t_k = timesteps[k]
            t_batch = torch.full((x.shape[0], 1), t_k, device=x.device, dtype=x.dtype)

            x1_pred = model(y, x, t_batch, c_grid)

            y = ratio_batch * y + (1-ratio_batch) * x1_pred

        # take last step w/o implied velocity and Euler step
        if self.model_last:
            t_batch = torch.full((x.shape[0], 1), timesteps[-1], device=x.device, dtype=x.dtype)
            y = model(y, x, t_batch, c_grid)

        return y

    def forward(self, model, x, c_grid, num_steps=None):
        return self.sample(model, x, c_grid, num_steps)
