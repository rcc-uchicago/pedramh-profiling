import sys
sys.path.append('/eagle/MDClimSim/awikner/PanguWeather-UC/v1.0/')
from networks.pangu import EarthAttention3D, EarthAttention3DMemEff
from utils.memory_efficient_attention import MemEffAttentionTorch
import torch
import os
import time
from torch.profiler import profile, record_function, ProfilerActivity

input_size = [3, 12, 144, 192]
torch.set_float32_matmul_precision('high')
torch.cuda.set_device(0)
device = torch.cuda.current_device()
#device = "cpu"
dtype = torch.float32

torch.manual_seed(0)
mem_attention = MemEffAttentionTorch(dim = 192, num_heads = 6, qkv_bias = True, proj_bias=False).to(dtype).to(device)
torch.manual_seed(0)
ea3d_attention = EarthAttention3D(192, [8, 18, 36], (2, 6, 12), 6).to(dtype).to(device)
torch.manual_seed(0)
mem_ea3d_attention = EarthAttention3DMemEff(192, [8, 18, 36], (2, 6, 12), 6).to(dtype).to(device)


x = torch.randn(*input_size).to(dtype).to(device)
x_mem = x.reshape(input_size[0]*input_size[1], input_size[2], input_size[3])
attn_bias = torch.randn(1, 6, input_size[2], input_size[2]).to(dtype).to(device)

with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
    out = mem_attention(x_mem)#, attn_bias)
    out.sum().backward()

print('Finished backward pass')
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
    out = ea3d_attention(x)
    out.sum().backward()

print('Finished backward pass')
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
print(out[0,0,0,-5:])


"""
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
    out = mem_ea3d_attention(x)
    out.sum().backward()

print('Finished backward pass')
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
print(out[0,0,0,-5:])
"""
