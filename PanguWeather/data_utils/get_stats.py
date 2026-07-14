### From FourCastNet repo

#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import torch
import numpy as np
import h5py


years = [1979] #, 1989, 1999, 2004, 2010]

global_means_sfc = np.zeros((1,4,1,1))
global_stds_sfc = np.zeros((1,4,1,1))
time_means_sfc = np.zeros((1,4,721,1440))

global_means_pl = np.zeros((1,5,13,1,1))
global_stds_pl = np.zeros((1,5,13,1,1))
time_means_pl = np.zeros((1,5,13,721,1440))

for ii, year in enumerate(years):
    
    with h5py.File('data/h5/train/'+ str(year) + '_sfc.h5', 'r') as f:
        rnd_idx = np.random.randint(0, 1460-500)
        global_means_sfc += np.mean(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,2,3))
        global_stds_sfc += np.var(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,2,3))

    with h5py.File('data/h5/train/'+ str(year) + '_pl.h5', 'r') as f:
        rnd_idx = np.random.randint(0, 1460-500)
        global_means_pl += np.mean(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,3,4))
        global_stds_pl += np.var(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,3,4))


global_means_sfc = global_means_sfc/len(years)
global_stds_sfc = np.sqrt(global_stds_sfc/len(years))
time_means_sfc = time_means_sfc/len(years)

np.save('data/stats/global_means_sfc.npy', global_means_sfc)
np.save('data/stats/global_stds_sfc.npy', global_stds_sfc)
np.save('data/stats/time_means_sfc.npy', time_means_sfc)

global_means_pl = global_means_pl/len(years)
global_stds_pl = np.sqrt(global_stds_pl/len(years))
time_means_pl = time_means_pl/len(years)

np.save('data/stats/global_means_pl.npy', global_means_pl)
np.save('data/stats/global_stds_pl.npy', global_stds_pl)
np.save('data/stats/time_means_pl.npy', time_means_pl)

print("means_sfc: ", global_means_sfc)
print("stds_sfc: ", global_stds_sfc)

print("means_pl: ", global_means_pl)
print("stds_pl: ", global_stds_pl)