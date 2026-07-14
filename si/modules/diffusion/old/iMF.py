'''
Implements improved Mean Flow (iMF).
Translated from JAX to PyTorch, with classifier-free guidance removed.
Uses conditioning input instead of class labels.
'''

import torch
import torch.nn as nn
from modules.models.old.pmfDiT import pmfDiT

class iMeanFlow(nn.Module):
    """improved MeanFlow (without CFG)"""

    def __init__(
        self,
        modelconfig,
        dtype: torch.dtype = torch.float32,
        img_size = [180, 360],
        img_channels: int = 3,
        # Noise distribution
        P_mean: float = -0.4,
        P_std: float = 1.0,
        # Loss
        data_proportion: float = 0.5,
        # Training dynamics
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        num_steps: int = 2,
    ):
        super().__init__()
        self.dtype = dtype
        self.img_size = img_size
        self.img_channels = img_channels

        # Noise distribution
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_steps = num_steps

        # Loss
        self.data_proportion = data_proportion

        # Training dynamics
        self.norm_p = norm_p
        self.norm_eps = norm_eps

        self.net = pmfDiT(**modelconfig)

    #######################################################
    #                       Solver                        #
    #######################################################

    def u_fn(self, x, t, h, cond):
        """
        Compute the predicted u and v components from the model.

        Args:
            x: Noisy image at time t. (B, C, H, W)
            t: Current time step. (B,)
            h: Time difference t - r. (B,)
            cond: Conditioning information.
        Returns: (u, v)
            u: Predicted u (average velocity field).
            v: Predicted v (instantaneous velocity field).
        """
        bz = x.shape[0]
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            cond,
        )

    def sample_one_step(self, z_t, i, t_steps, cond):
        """
        Perform one sampling step given current state z_t at time step i.

        Args:
            z_t: Current noisy image at time step t. (B, C, H, W)
            i: Current time step index.
            t_steps: Array of time steps.
            cond: Conditioning information.
        """
        t = t_steps[i]
        r = t_steps[i + 1]
        bsz = z_t.shape[0]

        t = t.expand(bsz)
        r = r.expand(bsz)

        u = self.u_fn(z_t, t, t - r, cond)[0]

        return z_t - (t - r)[:, None, None, None] * u

    #######################################################
    #                       Schedule                      #
    #######################################################

    def logit_normal_dist(self, bz, device):
        """Sample from logit-normal distribution. Returns (B, 1, 1, 1)."""
        rnd_normal = torch.randn(bz, 1, 1, 1, dtype=self.dtype, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz, device):
        """
        Sample t and r from logit-normal distribution.

        Returns:
            t: (B, 1, 1, 1)
            r: (B, 1, 1, 1)
            fm_mask: (B, 1, 1, 1) bool mask for flow matching samples
        """
        t = self.logit_normal_dist(bz, device)
        r = self.logit_normal_dist(bz, device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = torch.arange(bz, device=device) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = torch.where(fm_mask, t, r)

        return t, r, fm_mask

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    def forward(self, images, cond):
        """
        Forward process of improved MeanFlow and compute loss.

        Args:
            images: A batch of images, shape (B, C, H, W).
            cond: Conditioning information.

        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = images.to(self.dtype)
        bz = images.shape[0]
        device = images.device

        # Instantaneous velocity computation
        t, r, fm_mask = self.sample_tr(bz, device)

        e = torch.randn_like(x)
        z_t = (1 - t) * x + t * e
        v_t = e - x  # true instantaneous velocity

        # Without CFG, target velocity is the true instantaneous velocity
        v_g = v_t

        # Get model's predicted v at current time (used as jvp tangent)
        t_flat = t.reshape(bz)
        h_zero = torch.zeros(bz, dtype=self.dtype, device=device)
        v_c = self.u_fn(z_t, t_flat, h_zero, cond)[1]

        # Compute u and du/dt via forward-mode autodiff (jvp)
        def u_fn_primary(z_t_in, t_in, r_in):
            t_f = t_in.reshape(bz)
            r_f = r_in.reshape(bz)
            h = t_f - r_f
            return self.u_fn(z_t_in, t_f, h, cond)[0]

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(t)

        # Different from original MeanFlow: we use predicted v in the jvp
        (u,), (du_dt,) = torch.autograd.functional.jvp(
            lambda z, ti, ri: (u_fn_primary(z, ti, ri),),
            (z_t, t, r),
            (v_c, dtdt, dtdr),
        )

        # Get v from a separate forward pass
        t_flat = t.reshape(bz)
        r_flat = r.reshape(bz)
        _, v = self.u_fn(z_t, t_flat, t_flat - r_flat, cond)

        # Our compound function V = u + (t - r) * du/dt
        V = u + (t - r) * du_dt.detach()

        v_g = v_g.detach()

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / adp_wt.detach()

        # improved MeanFlow objective is conceptually v-loss
        loss_u = ((V - v_g) ** 2).sum(dim=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u).mean()

        # auxiliary v-head loss
        loss_v = ((v - v_g) ** 2).sum(dim=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v).mean()

        loss = loss_u + loss_v

        dict_losses = {
            "loss": loss,
            "loss_u": ((V - v_g) ** 2).mean(),
            "loss_v": ((v - v_g) ** 2).mean(),
        }

        return loss, dict_losses

    @torch.no_grad()
    def generate(self, cond, num_steps=None):
        """
        Generate samples from the model.

        Args:
            cond: Conditioning information.
            num_steps: Number of sampling steps.
        """
        device = cond.device
        n_sample = cond.shape[0]

        if num_steps is None:
            num_steps = self.num_steps

        x_shape = (n_sample, self.img_channels, self.img_size[0], self.img_size[1])
        z_t = torch.randn(x_shape, dtype=self.dtype, device=device)

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=self.dtype, device=device)

        for i in range(num_steps):
            t = t_steps[i]
            r = t_steps[i + 1]
            bsz = z_t.shape[0]
            t_b = t.expand(bsz)
            r_b = r.expand(bsz)

            u = self.u_fn(z_t, t_b, t_b - r_b, cond)[0]
            z_t = z_t - (t_b - r_b)[:, None, None, None] * u

        return z_t
