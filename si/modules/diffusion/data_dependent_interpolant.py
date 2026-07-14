import torch
import torch.nn as nn
import torch.nn.functional as F

class DataDependentInterpolant(nn.Module):
    """
    Stochastic interpolant with data-dependent couplings (Albergo et al. 2310.03725).

    Implements Algorithm 1 (training) and Algorithm 2 (sampling) from the paper.

    The interpolant is defined as:
        I_t = alpha_t * x0 + beta_t * x1

    where x0 = m(x1) + sigma * zeta is the data-dependent coupling
    (m is a corruption map, e.g. downsample-then-upsample, and zeta ~ N(0,I)).

    The velocity target is:
        dI_t/dt = alpha_dot_t * x0 + beta_dot_t * x1

    With alpha_t = 1-t, beta_t = t: dI_t/dt = x1 - x0.

    No additional gamma_t * z noise in the interpolant. All stochasticity
    comes from the data-dependent coupling x0 = m(x1) + sigma * zeta.
    """

    def __init__(self,
                 num_refinement_steps=5,
                 num_train_steps=None,
                 sigma_coupling=0.0,
                 l_max=None):
        super().__init__()
        self.num_refinement_steps = num_refinement_steps
        self.num_train_timesteps = num_train_steps if num_train_steps is not None else num_refinement_steps + 1
        self.sigma_coupling = sigma_coupling

        if l_max is not None:
            from modules.diffusion.dynamic_interpolant import SphereNoiseGenerator
            self.l_max = l_max 
            self.generator = SphereNoiseGenerator(l_max=l_max) 
        else:
            self.generator = None

    def alpha(self, t):
        """Interpolation coefficient for x0: alpha(t) = 1 - t"""
        return 1.0 - t

    def beta(self, t):
        """Interpolation coefficient for x1: beta(t) = t"""
        return t

    def alpha_dot(self, t):
        """Time derivative of alpha: -1"""
        return -torch.ones_like(t)

    def beta_dot(self, t):
        """Time derivative of beta: 1"""
        return torch.ones_like(t)
    
    def get_noise(self, x):
        """Generate noise for the data-dependent coupling."""
        if self.generator is not None:
            return self.generator(x.shape[0], x.shape[1], device=x.device)
        else:
            return torch.randn_like(x)

    def compute_loss(self, x_lowres, x_highres, model, cond=None):
        """
        Algorithm 1 from the paper: velocity matching training.

        For i = 1, ..., nb:
            Sample x1_i ~ rho_1, zeta_i ~ N(0, I), t_i ~ U(0, 1)
            x0_i = m(x1_i) + sigma * zeta_i
            I_t_i = alpha_t * x0_i + beta_t * x1_i

        Loss = (1/nb) * sum_i [ |b_hat(I_t_i, t_i)|^2 - 2 * dI_t_i/dt . b_hat(I_t_i, t_i) ]
        (equivalent to MSE up to a constant)

        Args:
            x_lowres: [b, c, h, w] — m(x1), the upsampled low-res (source base);
                      also concatenated channel-wise with I_t as model input
            x_highres: [b, c, h, w] — x1, ground truth (target distribution)
            model: velocity predictor, called as model(I_t, t, cond=x_lowres, history=cond)
            cond: [b, c, h, w] — optional high-res prior state/history for cross-attention

        Returns:
            scalar loss
        """
        b = x_lowres.shape[0]
        device = x_lowres.device

        # Data-dependent coupling: x0 = m(x1) + sigma * zeta
        if self.sigma_coupling > 0:
            zeta = self.get_noise(x_lowres)
            x0 = x_lowres + self.sigma_coupling * zeta
        else:
            x0 = x_lowres

        x1 = x_highres

        # sample timestep (shape (b, )) no need to train on t=1
        t = torch.randint(0, self.num_train_timesteps-1, device=device, size=(b,)) / (self.num_train_timesteps - 1)  # shape (b,)

        # Reshape for broadcasting: [b, 1, 1, 1]
        t_wide = t[:, None, None, None]

        # Interpolant: I_t = alpha_t * x0 + beta_t * x1
        alpha_t = self.alpha(t_wide)
        beta_t = self.beta(t_wide)
        I_t = alpha_t * x0 + beta_t * x1

        # Velocity target: dI_t/dt = alpha_dot_t * x0 + beta_dot_t * x1
        alpha_dot_t = self.alpha_dot(t_wide)
        beta_dot_t = self.beta_dot(t_wide)
        v_target = alpha_dot_t * x0 + beta_dot_t * x1  # = x1 - x0

        # Model predicts velocity

        if cond is not None:
            x_lowres = torch.cat([x_lowres, cond], dim=1)

        v_pred = model(I_t, x_lowres, t=t[:, None])

        # Loss: |b_hat|^2 - 2 * v_target . b_hat  (equivalent to MSE up to constant |v_target|^2)
        loss = (v_pred ** 2).sum(dim=[1, 2, 3]) - 2 * (v_target * v_pred).sum(dim=[1, 2, 3])  # shape (b,)

        #loss = F.mse_loss(v_pred, v_target, reduction='mean')

        return loss.mean()

    @torch.no_grad()
    def sample(self, x_lowres, model, cond=None, num_steps=None):
        """
        Algorithm 2 from the paper: forward Euler ODE integration.

        Draw zeta ~ N(0, I)
        X_0 = m(x1) + sigma * zeta
        For n = 0, ..., N-1:
            X_{n+1} = X_n + (1/N) * b_hat_{n/N}(X_n)

        Args:
            x_lowres: [b, c, h, w] — m(x1), upsampled low-res conditioning
            model: velocity predictor
            num_steps: number of integration steps N (default: self.num_refinement_steps)
            cond: [b, c, h, w] — optional high-res prior state/history for cross-attention

        Returns:
            [b, c, h, w] predicted high-res output
        """
        if num_steps is None:
            num_steps = self.num_refinement_steps

        # Starting point: X_0 = m(x1) + sigma * zeta
        if self.sigma_coupling > 0:
            zeta = self.get_noise(x_lowres)
            y = x_lowres + self.sigma_coupling * zeta
        else:
            y = x_lowres.clone()

        dt = 1.0 / num_steps

        if cond is not None:
            x_lowres = torch.cat([x_lowres, cond], dim=1)

        for n in range(num_steps):
            t_n = n / num_steps
            t_batch = torch.full((x_lowres.shape[0], 1), t_n,
                                 device=x_lowres.device, dtype=x_lowres.dtype)

            v = model(y, x_lowres, t_batch)
            y = y + dt * v

        return y

    def forward(self, x_lowres, x_highres, model):
        return self.compute_loss(x_lowres, x_highres, model)
