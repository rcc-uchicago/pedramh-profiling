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


# Instructions: 
# Set Nimgtot correctly

import os
#os.system('module load hdf5')

import h5py
from mpi4py import MPI
import numpy as np
import time
from netCDF4 import Dataset as DS
from scipy.interpolate import griddata



def writetofile(src, dest, channel_idx, varslist, src_idx=0, frmt='nc', Nimgtot=1460):
    if os.path.isfile(src):
        batch = 2**4
        rank = MPI.COMM_WORLD.rank
        Nproc = MPI.COMM_WORLD.size

        Nimg = Nimgtot//Nproc
        base = rank*Nimg
        end = (rank+1)*Nimg if rank<Nproc - 1 else Nimgtot
        idx = base
        
        fdest = h5py.File(dest, 'a', driver='mpio', comm=MPI.COMM_WORLD)

        for variable_name in varslist:

            if frmt == 'nc':
                fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
            elif frmt == 'h5':
                fsrc = h5py.File(src, 'r')[varslist[0]]

            print("fsrc shape", fsrc.shape)

            '''# Identify the indices of the missing values
            missing_value_mask = (fsrc == fsrc._FillValue)
            if missing_value_mask.any():
                # Get the coordinates of the data
                coords = np.array(np.meshgrid(
                    np.arange(fsrc.shape[0]), 
                    np.arange(fsrc.shape[1]), 
                    np.arange(fsrc.shape[2]), 
                    indexing='ij'
                    )).reshape(3, -1).T

                # Get valid and missing points
                valid_points = coords[~missing_value_mask.ravel()]
                missing_points = coords[missing_value_mask.ravel()]
                valid_values = fsrc[~missing_value_mask]

                # Interpolate missing values
                interpolated_values = griddata(valid_points, valid_values, missing_points, method='linear')

                # Fill in the missing values in the original array
                fsrc[missing_value_mask] = interpolated_values'''

            start = time.time()
            while idx<end:
                if end - idx < batch:
                    if len(fsrc.shape) == 4:
                        ims = fsrc[idx:end,src_idx]
                    else:
                        ims = fsrc[idx:end]
                    print('ch:', channel_idx, 'var:', variable_name, 'idx:', idx, "shape (last batch)", ims.shape)
                    fdest['fields'][idx:end, channel_idx, :, :] = ims
                    break
                else:
                    if len(fsrc.shape) == 4:
                        ims = fsrc[idx:idx+batch,src_idx]
                    else:
                        ims = fsrc[idx:idx+batch]
                    #ims = fsrc[idx:idx+batch]
                    print('ch:', channel_idx, 'var:', variable_name, 'idx:', idx, "shape", ims.shape)
                    fdest['fields'][idx:idx+batch, channel_idx, :, :] = ims
                    idx += batch
                    ttot = time.time() - start
                    eta = (end - base)/((idx - base)/ttot)
                    hrs = eta//3600
                    mins = (eta - 3600*hrs)//60
                    secs = (eta - 3600*hrs - 60*mins)

            ttot = time.time() - start
            hrs = ttot//3600
            mins = (ttot - 3600*hrs)//60
            secs = (ttot - 3600*hrs - 60*mins)
            channel_idx += 1 


def writetofile_pl(src, dest, channel_idxs, varslist, src_idx=0, frmt='nc', Nimgtot=1460):
    if os.path.isfile(src):
        batch = 2**4
        rank = MPI.COMM_WORLD.rank
        Nproc = MPI.COMM_WORLD.size

        Nimg = Nimgtot//Nproc
        base = rank*Nimg
        end = (rank+1)*Nimg if rank<Nproc - 1 else Nimgtot
        
        fdest = h5py.File(dest, 'a', driver='mpio', comm=MPI.COMM_WORLD) #, locking=False)

        for channel_idx, variable_name in enumerate(varslist):
            idx = base

            if frmt == 'nc':
                fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
            elif frmt == 'h5':
                fsrc = h5py.File(src, 'r')[varslist[0]]

            print("fsrc shape", fsrc.shape)
            
            '''# Identify the indices of the missing values
            missing_value_mask = (fsrc == fsrc._FillValue)
            if missing_value_mask.any():
                # Get the coordinates of the data
                coords = np.array(np.meshgrid(
                    np.arange(fsrc.shape[0]), 
                    np.arange(fsrc.shape[1]), 
                    np.arange(fsrc.shape[2]), 
                    indexing='ij'
                    )).reshape(3, -1).T

                # Get valid and missing points
                valid_points = coords[~missing_value_mask.ravel()]
                missing_points = coords[missing_value_mask.ravel()]
                valid_values = fsrc[~missing_value_mask]

                # Interpolate missing values
                interpolated_values = griddata(valid_points, valid_values, missing_points, method='linear')

                # Fill in the missing values in the original array
                fsrc[missing_value_mask] = interpolated_values'''

            start = time.time()
            while idx<end:
                if end - idx < batch:
                    if len(fsrc.shape) == 4:
                        ims = fsrc[idx:end,:]
                    else:
                        ims = fsrc[idx:end]
                    print('ch:', channel_idxs[channel_idx], 'var:', variable_name, 'idx:', idx, "shape (last batch)", ims.shape)
                    fdest['fields'][idx:end, channel_idxs[channel_idx], ...] = ims
                    break
                else:
                    if len(fsrc.shape) == 4:
                        ims = fsrc[idx:idx+batch,:]
                    else:
                        ims = fsrc[idx:idx+batch]
                    #ims = fsrc[idx:idx+batch]
                    print('ch:', channel_idxs[channel_idx], 'var:', variable_name, 'idx:', idx, "shape", ims.shape)
                    fdest['fields'][idx:idx+batch, channel_idxs[channel_idx], ...] = ims
                    idx += batch
                    ttot = time.time() - start
                    eta = (end - base)/((idx - base)/ttot)
                    hrs = eta//3600
                    mins = (eta - 3600*hrs)//60
                    secs = (eta - 3600*hrs - 60*mins)

            ttot = time.time() - start
            hrs = ttot//3600
            mins = (ttot - 3600*hrs)//60
            secs = (ttot - 3600*hrs - 60*mins)

            
if __name__ == '__main__':

    for year in range(1979, 2018):
        
        filestr = str(year)
        dest_sfc = 'data/h5/train/' + filestr + '_sfc.h5'
        dest_pl = 'data/h5/train/' + filestr + '_pl.h5'
        src_sfc = 'data/raw/' + filestr + '/' + filestr + '_sfc.nc'
        src_pl = 'data/raw/' + filestr + '/' + filestr + '_pl.nc'

        ######
        time_steps = 1460 #days*4 if dt=6h 
        with h5py.File(dest_sfc, 'w') as f:
            f.create_dataset('fields', shape = (time_steps, 4, 721, 1440), dtype='f')
        with h5py.File(dest_pl, 'w') as f:
            f.create_dataset('fields', shape = (time_steps, 5, 13, 721, 1440), dtype='f')
        ######
        
        #z, q, t, u, v
        writetofile_pl(src_pl, dest_pl, [0, 1, 2, 3, 4], ['z', 'q', 't', 'u', 'v'], Nimgtot=time_steps)

        
        #mslp u10 v10 t2m
        writetofile(src_sfc, dest_sfc, 0, ['msl'], Nimgtot=time_steps)
        writetofile(src_sfc, dest_sfc, 1, ['u10'], Nimgtot=time_steps)
        writetofile(src_sfc, dest_sfc, 2, ['v10'], Nimgtot=time_steps)
        writetofile(src_sfc, dest_sfc, 3, ['t2m'], Nimgtot=time_steps)

