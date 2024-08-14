from networks.pangu import PanguModel
import os
import time
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from utils.YParams import YParams

params = YParams(os.path.abspath(args.yaml_config), args.config)


model = PanguModel(params)