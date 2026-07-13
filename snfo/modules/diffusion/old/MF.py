import torch
import torch.nn as nn
import numpy as np
from torch.func import jvp
from modules.models.old.SiT import SiT

class MeanFlow(nn.Module):
    def __init__(
            self,
            modelconfig,
            path_type="linear",
            weighting="uniform",
            # New parameters
            time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
            time_mu=-0.4,                 # Mean parameter for logit_normal distribution
            time_sigma=1.0,               # Std parameter for logit_normal distribution
            ratio_r_not_equal_t=0.75,     # Ratio of samples where r≠t
            adaptive_p=1.0,               # Power param for adaptive weighting
            in_channels=3,
            img_size = [180, 360],
            num_steps = 2,
            ):
        super().__init__()

        self.weighting = weighting
        self.path_type = path_type
        self.img_size = img_size
        self.in_channels = in_channels
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        # Adaptive weight config
        self.adaptive_p = adaptive_p
        self.num_steps = num_steps
        self.model = SiT(**modelconfig)
    
    @torch.no_grad()
    def generate(
        self,
        y, 
        num_steps=None, 
    ):
        """
        MeanFlow sampler supporting both single-step and multi-step generation
        
        Based on Eq.(12): z_r = z_t - (t-r)u(z_t, r, t)
        For single-step: z_0 = z_1 - u(z_1, 0, 1)
        For multi-step: iteratively apply the Eq.(12) with intermediate steps
        """
        if num_steps is None:
            num_steps = self.num_steps

        device = y.device
        latents = torch.randn(y.shape[0], self.in_channels, self.img_size[0],self.img_size[1], device=device)
        batch_size = latents.shape[0]
        device = latents.device
        
        if num_steps == 1:
            r = torch.zeros(batch_size, device=device)
            t = torch.ones(batch_size, device=device)
            
            u = self.model(latents, r, t, y=y)
            
            # x_0 = x_1 - u(x_1, 0, 1)
            x0 = latents - u
            
        else:
            z = latents
            
            time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
            
            for i in range(num_steps):
                t_cur = time_steps[i]
                t_next = time_steps[i + 1]
                
                t = torch.full((batch_size,), t_cur, device=device)
                r = torch.full((batch_size,), t_next, device=device)
                
                u = self.model(z, r, t, y=y)
                
                # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
                z = z - (t_cur - t_next) * u
            
            x0 = z
        
        return x0

    def interpolant(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        
        return r, t 
    
    def forward(self, images, y):
        """
        Compute MeanFlow loss function with bootstrap mechanism
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)

        noises = torch.randn_like(images)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)
                        
        #u = self.model(z_t, r, t, y)
        
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        
        def fn_current(z, cur_r, cur_t):
            return self.model(z, cur_r, cur_t, y)

        u, dudt = jvp(fn_current,primals,tangents)
        
        u_target = v_t - time_diff * dudt
                
        # Detach the target to prevent gradient flow        
        error = u - u_target.detach()
        loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
        # Apply adaptive weighting based on configuration
        if self.weighting == "adaptive":
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid          
        else:
            loss = loss_mid
        loss_mean_ref = torch.mean((error**2))
        return loss.mean(), loss_mean_ref