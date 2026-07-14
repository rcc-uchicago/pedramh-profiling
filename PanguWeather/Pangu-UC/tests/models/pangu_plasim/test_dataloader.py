import torch
import numpy as np
import sys
sys.path.append('/glade/work/awikner/Pangu-UC')
#import modulus
from dataclasses import dataclass


#from weatherlearn.models import PanguPlasim
from examples.pangu_plasim.data_loader_nc import *
from torch.utils.data import DataLoader
from tqdm import tqdm
#from weatherlearn.models.pangu.pangu import PanguPlasimModulus

import unittest

#class TestMain(unittest.TestCase):
#    def test_loader_init(self):
#datadir = 'C:\\Users\\user\\Documents\\PLASIM\\data\\plasim_reduced_data'
#datadir = '/Users/Alexander/Documents/PLASIM/data/plasim_reduced_data'
datadir = '/glade/derecho/scratch/awikner/PLASIM/data/plasim_reduced_data'
boundary_dir = 'boundary_vars'
upper_air_vars = ['ta', 'ua', 'va', 'hus', 'clw']
surface_vars = ['pl', 'tas']
boundary_vars_yearly = ['sst', 'rsdt', 'sic']
boundary_vars_constant = ['lsm', 'z0', 'sg']
start_year = 100
end_year = 102
flag = 'train'
surface_mean_file = 'plasim_surface_test_mean.nc'
surface_std_file = 'plasim_surface_test_stds.nc'
upper_air_mean_file = 'plasim_test_mean.nc'
upper_air_std_file = 'plasim_test_std.nc'
calendar = 'proleptic_gregorian'
timedelta_hours = 6
dataset = DatasetFromFolder(datadir, start_year, end_year, flag, surface_vars,
                            upper_air_vars, boundary_vars_constant, boundary_vars_yearly,
                            boundary_dir, surface_mean_file, surface_std_file, upper_air_mean_file,
                            upper_air_std_file, calendar, timedelta_hours)
dataloader = tqdm(DataLoader(dataset, batch_size=32, shuffle=True))
for input_surface, input_upper_air, target_surface, target_upper_air, \
        boundary_data in dataloader:
    pass

