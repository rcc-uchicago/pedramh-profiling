import torch
import torchvision
from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch.nn.functional as F


def latitude_weighting_factor_torch(latitudes):
    lat_weights_unweighted = torch.cos(3.1416/180. * latitudes)
    n_lat = latitudes.shape[0]
    return n_lat*lat_weights_unweighted/torch.sum(lat_weights_unweighted)

def weighted_mse(pred, target, latitudes, reduction='mean'):
    #takes in arrays of size [n, c, h, w]  or [n, c, l, h, w]
    reshape_shape = tuple(1 if i != len(pred.shape) - 2 else -1 for i in range(len(pred.shape)))
    weight = torch.reshape(latitude_weighting_factor_torch(latitudes), reshape_shape)
    if reduction == 'mean':
        result = torch.mean(weight * (pred - target)**2)
    elif reduction == 'sum':
        result = torch.sum(weight * (pred - target)**2)
    else:
        result = weight * (pred - target)**2
    return result

def weighted_mae(pred, target, latitudes, reduction='mean'):
    #takes in arrays of size [n, c, h, w]  or [n, c, l, h, w]
    reshape_shape = tuple(1 if i != len(pred.shape) - 2 else -1 for i in range(len(pred.shape)))
    weight = torch.reshape(latitude_weighting_factor_torch(latitudes), reshape_shape)
    if reduction == 'mean':
        result = torch.mean(weight * torch.abs(pred - target))
    elif reduction == 'sum':
        result = torch.sum(weight * torch.abs(pred - target))
    else:
        result = weight * torch.abs(pred - target)
    return result

class Latitude_weighted_MSELoss(_Loss):
    def __init__(self, latitudes) -> None:
        super().__init__()
        self.latitudes = latitudes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return weighted_mse(input, target, self.latitudes)
    

class Latitude_weighted_L1Loss(_Loss):
    def __init__(self, latitudes) -> None:
        super().__init__()
        self.latitudes = latitudes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return weighted_mae(input, target, self.latitudes)
    
class Masked_L1Loss(_Loss):
    def __init__(self, mask) -> None:
        super().__init__()
        self.mask = mask

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        elem_loss =  F.l1_loss(input, target, reduction = 'none')
        masked_loss = torch.where(self.mask, elem_loss, torch.nan)
        return torch.nanmean(masked_loss)

class Masked_MSELoss(_Loss):
    def __init__(self, mask) -> None:
        super().__init__()
        self.mask = mask

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        elem_loss =  F.mse_loss(input, target, reduction = 'none')
        masked_loss = torch.where(self.mask, elem_loss, torch.nan)
        return torch.nanmean(masked_loss)
    
class Latitude_weighted_masked_L1Loss(_Loss):
    def __init__(self, latitudes, mask) -> None:
        super().__init__()
        self.latitudes = latitudes
        self.mask = mask

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        elem_loss =  weighted_mae(input, target, self.latitudes, reduction = 'none')
        masked_loss = torch.where(self.mask, elem_loss, torch.nan)
        return torch.nanmean(masked_loss)
    
class Latitude_weighted_masked_MSELoss(_Loss):
    def __init__(self, latitudes, mask) -> None:
        super().__init__()
        self.latitudes = latitudes
        self.mask = mask

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        elem_loss =  weighted_mse(input, target, self.latitudes, reduction = 'none')
        masked_loss = torch.where(self.mask, elem_loss, torch.nan)
        return torch.nanmean(masked_loss)
    
class Latitude_weighted_CRPSLoss(_Loss):
    def __init__(self, latitudes, num_ensemble_members, mask=None) -> None:
        super().__init__()
        self.latitudes = latitudes
        self.num_ensemble_members = num_ensemble_members
        self.mask = mask
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        B = input.shape[0] // self.num_ensemble_members
        reshaped_input = input.view(B, self.num_ensemble_members, *input.shape[1:])
        reshaped_target = target.view(B, self.num_ensemble_members, *target.shape[1:])
        total_loss = []
        for i in range(B):
            loss = self.CRPSSkill(reshaped_input[i], reshaped_target[i]) - \
                0.5 * self.CRPSSpread(reshaped_input[i], reshaped_input[i])
            if self.mask:
                loss = torch.where(self.mask, loss, torch.nan)
                
            total_loss.append(torch.nanmean(loss))

        return torch.mean(torch.stack(total_loss))
    
    def CRPSSkill(self, input: Tensor, target: Tensor) -> Tensor:
        return weighted_mae(input, target, self.latitudes, reduction='none').mean(dim=0)
    
    def CRPSSpread(self, input: Tensor, target: Tensor) -> Tensor:
        # compute (1/(M-1)) * mean(x_i - x_j) summed over all i,j
        # only compute for i<j since sum is symmetric
        spread = torch.zeros_like(input[0])
        for i in range(input.shape[0]):
            for j in range(i+1, input.shape[0]):
                spread += 2 * weighted_mae(input[i], target[j], self.latitudes, reduction='none')
        prefactor = 1 / (self.num_ensemble_members * (self.num_ensemble_members - 1))
        spread = prefactor * spread
        return spread
    



class Kl_divergence_gaussians(_Loss):
    def __init__(self) -> None:
        super().__init__()
  
    def forward(self, mu_q, logvar_q, mu_p=None, logvar_p=None):
        """
        Computes the KL divergence between two multivariate Gaussians with diagonal covariances.
        
        q ~ N(mu_q, var_q), p ~ N(mu_p, var_p)
        logvar_* are the log variances (for numerical stability).

        If mu_p and logvar_p are None, assumes p is standard normal (mu=0, var=1).

        Args:
            mu_q.   : Tensor of shape [batch_size, latent_dim]
            logvar_q: Tensor of shape [batch_size, latent_dim]
            mu_p.   : Tensor of shape [batch_size, latent_dim] or None
            logvar_p: Tensor of shape [batch_size, latent_dim] or None

        Returns:
            kl: average KL divergence over the batch, a scalar tensor.
        """
        if mu_p is None:
            mu_p = torch.zeros_like(mu_q)
        if logvar_p is None:
            logvar_p = torch.zeros_like(logvar_q)

        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)

        kl = 0.5 * (
            logvar_p - logvar_q
            + (var_q + (mu_q - mu_p).pow(2)) / var_p
            - 1
        )

        return kl.mean()    # Sum over latent dimension