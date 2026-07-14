import sys
sys.path.append('/eagle/MDClimSim/awikner/PanguWeather-UC/v1.0/')
from networks.pangu import PanguModel_Plasim
from utils.YParams import YParams
import torch
import os
import time
from torch.profiler import profile, record_function, ProfilerActivity

yaml_config = 'config/PANGU_PLASIM_POLARIS.yaml'
params = YParams(os.path.abspath(yaml_config), 'PLASIM')
print('Horizontal resolution:') 
print(params.horizontal_resolution)

torch.set_float32_matmul_precision('high')
torch.cuda.set_device(0)
device = torch.cuda.current_device()
dtype = torch.float16
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.allow_tf32 = True
TORCH_CUDNN_SDPA_ENABLED=1

"""
torch.manual_seed(0)
model = PanguModel(params).to(dtype).to(device)
print("# parameters: ", sum(param.numel() for param in model.parameters()))
#model = torch.compile(model, mode='default')

surface = torch.randn(1, num_surface, 721//img_scale, 1440//img_scale).to(dtype).to(device)
surface_mask = torch.randn(1, num_constant, 721//img_scale, 1440//img_scale).to(dtype).to(device)
upper_air = torch.randn(1, num_vars, num_levels, 721//img_scale, 1440//img_scale).to(dtype).to(device)

out = model(surface, surface_mask, upper_air)
(out[0].sum() + out[1].sum()).backward()
model.zero_grad()

with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
    out = model(surface, surface_mask, upper_air)
    (out[0].sum() + out[1].sum()).backward()

print(out[0][0,0,0,-5:])
print('Finished backward pass')
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
"""
torch.manual_seed(0)
model = PanguModel_Plasim(params).to(dtype).to(device)
print("# parameters: ", sum(param.numel() for param in model.parameters()))
model = torch.compile(model, mode='default')

surface = torch.randn(1, len(params.surface_variables), params.horizontal_resolution[0], params.horizontal_resolution[1]).to(dtype).to(device)
surface_mask = torch.randn(1, len(params.constant_boundary_variables), params.horizontal_resolution[0], params.horizontal_resolution[1]).to(dtype).to(device)
upper_air = torch.randn(1, len(params.upper_air_variables), params.num_levels, params.horizontal_resolution[0], params.horizontal_resolution[1]).to(dtype).to(device)
varying_boundary = torch.randn(1, len(params.varying_boundary_variables), params.horizontal_resolution[0], params.horizontal_resolution[1]).to(dtype).to(device)

out = model(surface, surface_mask, varying_boundary, upper_air)
(out[0].sum() + out[1].sum()).backward()
model.zero_grad()

with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
    out = model(surface, surface_mask, varying_boundary, upper_air)
    (out[0].sum() + out[1].sum()).backward()

print(out[0][0,0,0,-5:])
print('Finished backward pass')
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
