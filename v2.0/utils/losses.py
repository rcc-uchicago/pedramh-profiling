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
