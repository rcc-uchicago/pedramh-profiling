#!/bin/bash
#PBS -N convert
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -q debug 
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle                          
#PBS -A MDClimSim

module use /soft/modulefiles
ml conda
conda activate base
cd /eagle/MDClimSim/awikner/PanguWeather-UC
mpirun -np 64 python data_utils/netcdf_to_h5.py
