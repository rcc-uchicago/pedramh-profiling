import torch
import torch.nn as nn
from modules.diffusion.utils import sample_logit_normal, power_sampler, get_log_uniform_t
import math
from collections import deque

class DynamicInterpolant(nn.Module):
    def __init__(self,
                 num_steps,  # this corresponds to physical time steps
                 sigma_coef=1.0,
                 train_sampler='uniform',
                 integrator='euler',
                 l_max = 180,
                 spectral_weight = 0.01,
                 noise = "spherical",
                 model_last = False,
                 loss_form = "x",
                 noise_scale_path = None,
                 gamma = 0.5,
                 tau = 1.3,
                 S_churn = 0.0,
                 t_churn_min = 0.05,
                 t_churn_max = 0.95,
                 S_noise = 2.0
                 ):
        super(DynamicInterpolant, self).__init__()

        self.num_steps = num_steps
        self.sigma_coef = sigma_coef
        self.train_sampler = train_sampler
        self.model_last = model_last
        self.loss_form = loss_form
        self.integrator = integrator
        self.gamma = gamma
        self.S_churn = S_churn
        self.t_churn_min = t_churn_min
        self.t_churn_max = t_churn_max
        self.S_noise = S_noise
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
            noise_scales = torch.load(noise_scale_path)
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

    def compute_multistep_loss(self, model, x, c_grids, y, num_sample_steps=None):
        # x contains initial prognostic state, shape b c h w
        # c_grids contains the forcing state for each rollout step, shape b rollout c h w
        # y contains the final prognostic state after rollout steps, shape b c h w
        # num_sample_steps: truncated schedule length for intermediate rollout samples

        rollout = c_grids.shape[1]

        x_current = x
        if rollout > 1:
            with torch.no_grad():
                for step in range(rollout - 1):
                    x_current = self.sample(model, x_current, c_grids[:, step], num_steps=num_sample_steps)

        return self.compute_loss(model, x_current, c_grids[:, -1], y)

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
        W_t = torch.sqrt(t) * noise
        X_t = (1-t) * x + t * y + (1-t) * self.sigma_coef * W_t

        pred_y = model(X_t, x, t.squeeze(dim=[1, 2, 3]), c_grid)

        if self.loss_form == 'x':
            loss = ((pred_y - y) ** 2).sum(dim=[1, 2, 3]).mean() 
        elif self.loss_form == 'v': # this is trivial, since additive terms cancel before gradient computation
            # construct target
            target = (y - x) - self.sigma_coef * W_t
            pred = (pred_y - x) - self.sigma_coef * W_t
            loss = ((pred - target) ** 2).sum(dim=[1, 2, 3]).mean()

        if self.spectral_weight > 0:
            spectral_loss = self.spectral_weight * self.spectral_criterion(pred_y, y)
        else:
            spectral_loss = 0

        loss = loss + spectral_loss

        return loss, spectral_loss
    
    def sample_ddim(self, model, x, c_grid, num_steps=None, gamma=0.0,
                    return_model_last=False):
        """
        DDIM-style sampler for the stochastic interpolant
            I_t = (1-t) x_0 + t x_1 + sigma_coef * (1-t) * sqrt(t) * z.

        At each step:
        1. Predict x_hat_1 = model(y, x, t, c_grid).
        2. Invert the interpolant to extract the noise implicit in y:
                z_hat = (y - (1-t) x - t x_hat_1) / (sigma_coef (1-t) sqrt(t))
        3. Mix z_hat with fresh Gaussian noise xi (variance-preserving):
                z_next = gamma * z_hat + sqrt(1 - gamma^2) * xi
        4. Reconstruct y at the next timestep directly from the interpolant:
                y = (1 - t') x + t' x_hat_1 + sigma_coef (1-t') sqrt(t') * z_next

        Because sigma_coef * (1-t) * sqrt(t) vanishes at t=1, the final step
        lands exactly on x_hat_1 — no residual Brownian noise, no W_t to
        track, no brittle cancellation. Stochastic exploration (needed for
        high-frequency content) is preserved as long as gamma < 1.

        Args:
            gamma: noise-retention coefficient in [0, 1].
                0.0  -> fully stochastic, fresh noise every step.
                        Marginals match your SDE; this is the default.
                1.0  -> deterministic (probability-flow-like).
                ~0.5 -> "churn", often a good compromise.
        """
        if num_steps is None:
            num_steps = self.num_steps

        timesteps = torch.linspace(0, 1, num_steps + 1, device=x.device)
        y = x.clone()
        y_pred = None

        g_old = gamma
        g_new = math.sqrt(max(0.0, 1.0 - gamma * gamma))

        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            t_curr_batch = t_curr.expand(x.shape[0])

            # 1. Predict clean target from current state.
            y_pred = model(y, x, t_curr_batch, c_grid)

            # Interpolant noise scales. sigma_curr is 0 at t=0, sigma_next is 0 at t=1.
            sigma_curr = self.sigma_coef * (1.0 - t_curr) * torch.sqrt(t_curr)
            sigma_next = self.sigma_coef * (1.0 - t_next) * torch.sqrt(t_next)

            # Fresh noise for this step.
            xi = self.get_noise(x)
            if self.noise_scales is not None:
                xi = xi * self.noise_scales

            # 2 + 3. Build the noise for the next timestep.
            if i == 0:
                # At t=0, y == x exactly and sigma_curr == 0; no noise to invert.
                # Seed the first marginal with full-variance fresh noise.
                z_next = xi
            else:
                z_hat = (y - (1.0 - t_curr) * x - t_curr * y_pred) / sigma_curr
                z_next = g_old * z_hat + g_new * xi

            # 4. Reconstruct y at t_next. At t_next == 1, sigma_next == 0 and
            #    y collapses to y_pred — clean termination, no extra logic needed.
            y = (1.0 - t_next) * x + t_next * y_pred + sigma_next * z_next

        if return_model_last:
            return y, y_pred
        return y
    

    def sample_exponential(self, model, x, c_grid, num_steps=None, return_model_last=False):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)
        if num_steps is None:
            num_steps = self.num_steps

        timesteps, ratio = get_log_uniform_t(n_t = num_steps - 1, scale = self.tau, device = x.device)
        
        ratio_batch = ratio.expand(x.shape[0], 1, 1, 1)

        y = x.clone()

        for k in range(num_steps):
            t_k = timesteps[k]
            t_batch = torch.full((x.shape[0],), t_k, device=x.device, dtype=x.dtype)

            x1_pred = model(y, x, t_batch, c_grid)

            noise = self.get_noise(x)
            if self.noise_scales is not None:
                noise = noise * self.noise_scales

            diffusion_scale = self.sigma_coef * (1-self.wide(t_batch)) * torch.sqrt((1-ratio_batch) * (1-t_batch))
            
            y = ratio_batch * y + (1-ratio_batch) * x1_pred + diffusion_scale * noise

        if return_model_last:
            assert self.model_last is False
            return y, x1_pred

    def sample_uniform(self, model, x, c_grid, num_steps=None, return_model_last=False):
        # x contains current prognostic state (latent space)
        # c_grid contains current forcing state (original resolution)
        if num_steps is None:
            num_steps = self.num_steps

        timesteps = torch.linspace(0, 1, num_steps + 1, device=x.device)
        # start y at source distribution, which is current state
        y = x.clone()
        W_t = torch.zeros_like(x)

        if self.model_last:
            num_steps_drift = num_steps - 1
        else:
            num_steps_drift = num_steps

        # --- Churn configuration (no-op if S_churn == 0) ---
        S_churn    = getattr(self, 'S_churn',    0.0)
        S_noise    = getattr(self, 'S_noise',    1.0)
        t_churn_lo = getattr(self, 't_churn_min', 0.05)
        t_churn_hi = getattr(self, 't_churn_max', 0.95)
        a_step = min(S_churn / max(num_steps_drift, 1), 1.0) if S_churn > 0 else 0.0

        # Buffer to store drift history [v_{n-2}, v_{n-1}, v_n]
        if self.integrator == 'AB3':
            drift_history = deque(maxlen=3)

        for i in range(num_steps_drift):
            t_current = timesteps[i]
            t_next    = timesteps[i + 1]
            dt = t_next - t_current
            t_curr_batch = t_current.expand(x.shape[0])

            # ================== EDM-style churn ==================
            # Refresh part of W_t with fresh Gaussian noise of the correct scale,
            # keeping Var(W_t) = t (marginal-preserving), then update y consistently
            # with the interpolant y = (1-t)x_0 + t x_1 + sigma_coef*(1-t)*W_t.
            # Skipped at i == 0 because W_0 = 0 (nothing to refresh).
            if a_step > 0 and i > 0:
                t_val = t_current.item()
                if t_churn_lo <= t_val <= t_churn_hi:
                    eps = self.get_noise(x) * S_noise
                    if self.noise_scales is not None:
                        eps = eps * self.noise_scales
                    sqrt_keep = (1.0 - a_step) ** 0.5
                    sqrt_fresh = (a_step * t_current) ** 0.5     # tensor * float
                    W_churned = sqrt_keep * W_t + sqrt_fresh * eps
                    churn_scale = self.sigma_coef * (1 - self.wide(t_curr_batch))
                    y = y + churn_scale * (W_churned - W_t)
                    W_t = W_churned
            # =====================================================

            y_pred = model(y, x, t_curr_batch, c_grid)  # Predict x_1
            drift_curr = (y_pred - x) - self.sigma_coef * W_t  # v_theta(t)

            if self.integrator == 'AB3': # don't use this with stochastic churn
                drift_history.append(drift_curr)
                history_len = len(drift_history)

                if history_len == 1:
                    drift_step = drift_curr
                elif history_len == 2:
                    v_n         = drift_history[-1]
                    v_n_minus_1 = drift_history[-2]
                    drift_step  = 1.5 * v_n - 0.5 * v_n_minus_1
                else:
                    v_n         = drift_history[-1]
                    v_n_minus_1 = drift_history[-2]
                    v_n_minus_2 = drift_history[-3]
                    drift_step  = (23 * v_n - 16 * v_n_minus_1 + 5 * v_n_minus_2) / 12.0
                drift_curr = drift_step

            noise = self.get_noise(x)
            if self.noise_scales is not None:
                noise = noise * self.noise_scales
            dW = torch.sqrt(dt) * noise
            diffusion_scale = self.sigma_coef * (1 - self.wide(t_curr_batch))

            # Euler predictor
            y_next_euler = y + drift_curr * dt + diffusion_scale * dW
            W_next       = W_t + dW

            if self.integrator == 'heun':
                t_next_batch = t_next.expand(x.shape[0])
                y_pred_next = model(y_next_euler, x, t_next_batch, c_grid)
                drift_next  = (y_pred_next - x) - self.sigma_coef * W_next
                y = y + 0.5 * (drift_curr + drift_next) * dt + diffusion_scale * dW
            else:
                y = y_next_euler

            W_t = W_next

        if return_model_last:
            assert self.model_last is False
            return y, y_pred

        if self.model_last:
            y = model(y, x, timesteps[-2].expand(x.shape[0]), c_grid)

        return y
    
    def sample(self, model, x, c_grid, num_steps=None, return_model_last=False):
        if self.integrator == 'ddim':
            gamma = self.gamma
            return self.sample_ddim(model, x, c_grid,
                                    num_steps=num_steps,
                                    gamma=gamma,
                                    return_model_last=return_model_last)
        elif self.integrator == 'exponential':
            return self.sample_exponential(model, x, c_grid,
                                     num_steps=num_steps,
                                     return_model_last=return_model_last)
        else:
            return self.sample_uniform(model, x, c_grid,
                                       num_steps=num_steps,
                                       return_model_last=return_model_last)

    def forward(self, model, x, c_grid, num_steps=None):
        return self.sample(model, x, c_grid, num_steps)
