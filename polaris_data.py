import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
#import cv2
from utils.img_utils import reshape_fields
import pdb

class ERA5OneStepRandomizedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        transform,
        dict_diff_transform,
        list_intervals=[6, 12, 24],
        data_freq=1, # 1-hourly or 3-hourly or 6-hourly data
        year_list=None,
        region_info=None,
        flip_data=False,
    ):
        super().__init__()
        
        for l in list_intervals:
            assert l % data_freq == 0
        
        self.root_dir = root_dir
        self.variables = variables
        # self.transform = transform
        # self.dict_diff_transform = dict_diff_transform
        self.list_intervals = list_intervals
        self.data_freq = data_freq
        self.year_list = year_list
        self.region_info = region_info
        self.flip_data = flip_data
        if year_list is not None:
            self.year_idx_map = {year: i for i, year in enumerate(year_list)}
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        # self.file_paths = sorted(file_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        self.file_paths = sorted(file_paths)
        
    def __len__(self):
        return len(self.file_paths) - max(self.list_intervals) // self.data_freq
    
    def get_out_path(self, year, inp_file_idx, steps):
        out_file_idx = inp_file_idx + steps
        out_path = os.path.join(
            self.root_dir,
            f'{year}_{out_file_idx:04}.h5'
        )
        if not os.path.exists(out_path):
            for i in range(steps):
                out_file_idx = inp_file_idx + i
                out_path = os.path.join(
                    self.root_dir,
                    f'{year}_{out_file_idx:04}.h5'
                )
                if os.path.exists(out_path):
                    max_step_forward = i
            remaining_steps = steps - max_step_forward
            if self.year_list is None:
                next_year = year + 1
            else:
                next_year = self.year_list[self.year_idx_map[year] + 1]
            out_path = os.path.join(
                self.root_dir,
                f'{next_year}_{remaining_steps-1:04}.h5'
            )
        return out_path
    
    def get_data_given_path(self, path):
        with h5py.File(path, 'r') as f:
            data = {
                main_key: {
                    sub_key: np.array(value) for sub_key, value in group.items() if sub_key in self.variables + ['time']
            } for main_key, group in f.items() if main_key in ['input']}
        
        if self.flip_data:
            for main_key, group in data.items():
                for sub_key, value in group.items():
                    if sub_key != 'time':
                        data[main_key][sub_key] = np.flip(value, axis=0)
                    else:
                        data[main_key][sub_key] = value
        
        return data
    
    def __getitem__(self, index):
        path = self.file_paths[index]
        interval = np.random.choice(self.list_intervals)
        
        steps = interval // self.data_freq
        year, inp_file_idx = os.path.basename(path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        out_path = self.get_out_path(year, inp_file_idx, steps)
        inp_data = self.get_data_given_path(path)
        out_data = self.get_data_given_path(out_path)
        
        # inp = [inp_data['input'][v] for v in self.variables]
        # inp = np.stack(inp, axis=0)
        inp = []
        for v in self.variables:
            if inp_data['input'][v].shape[0] < inp_data['input'][v].shape[1]:
                inp.append(inp_data['input'][v])
            else: # transpose if long before lat
                inp.append(inp_data['input'][v].T)
        inp = np.stack(inp, axis=0)
        
        # out = [out_data['input'][v] for v in self.variables]
        # out = np.stack(out, axis=0)
        out = []
        for v in self.variables:
            if out_data['input'][v].shape[0] < out_data['input'][v].shape[1]:
                out.append(out_data['input'][v])
            else: # transpose if long before lat
                out.append(out_data['input'][v].T)
        out = np.stack(out, axis=0)
    
        diff = out - inp
        inp = torch.from_numpy(inp)
        diff = torch.from_numpy(diff)
        
        interval_tensor = torch.Tensor([interval]) / 10.0
        
        if self.region_info is None:
            return (
                self.transform(inp).unsqueeze(0), # normalized
                inp.unsqueeze(0), # raw
                self.dict_diff_transform[interval](diff), # normalized
                interval_tensor,
                self.variables,
                self.variables
            )
        else:
            return (
                self.transform(inp).unsqueeze(0), # normalized
                inp.unsqueeze(0), # raw
                self.dict_diff_transform[interval](diff), # normalized
                interval_tensor,
                self.variables,
                self.variables,
                self.region_info
            )


if __name__ == '__main__':
    root_dir = '/eagle/lighthouse-uchicago/members/hyadav/Pangu-Working/small_data'
    sfc_variables = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed', 'surface_pressure']
    pl_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
    heights = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    pl_variables_final = [f"{s}_{h}" for s in pl_variables for h in heights]
    variables = sfc_variables + pl_variables_final
    dataset = ERA5OneStepRandomizedDataset(root_dir,variables,transform,dict_diff_transform)