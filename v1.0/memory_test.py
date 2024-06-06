import sys
sys.path.append('/eagle/MDClimSim/awikner/PanguWeather-UC/v1.0/')
from networks.pangu import PanguModel, PanguModelMemEff
from utils.YParams import YParams
import torch
import os
import time
from torch.profiler import profile, record_function, ProfilerActivity

yaml_config = 'config/PANGU.yaml'
params = YParams(os.path.abspath(yaml_config), 'base_config')
print('img_scale: %d' % params.img_scale)
img_scale = params.img_scale

num_vars = 5
num_levels = 13
num_surface = 4
num_constant = 3

torch.set_float32_matmul_precision('high')
torch.cuda.set_device(0)
device = torch.cuda.current_device()
dtype = torch.float16
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.allow_tf32 = True
TORCH_CUDNN_SDPA_ENABLED=1

torch.manual_seed(0)
model = PanguModelMemEff(params, embed_dim = 156).to(dtype).to(device)
print("# parameters: ", sum(param.numel() for param in model.parameters()))
model = torch.compile(model, mode='default')

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

torch.manual_seed(0)
model = PanguModel(params, embed_dim = 156).to(dtype).to(device)
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
