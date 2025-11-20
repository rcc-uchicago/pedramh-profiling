#!/bin/bash -l
#PBS -N to_zarr
#PBS -l select=1:ncpus=16:mem=235G
#PBS -q main
#PBS -l walltime=02:00:00
#PBS -A UCHI0014
#PBS -o /glade/u/home/aasche/PanguWeather/v2.0/logs
#PBS -j oe
# PBS -J 1-2

ml conda
conda activate aires

cd /glade/u/home/aasche/PanguWeather/data_utils

INPUT_DIR="/glade/derecho/scratch/aasche/PLASIM/data/ensemble_test/paper"
OUTPUT_ZARR="/glade/derecho/scratch/aasche/PLASIM/data/ensemble_test/paper/ens_50.zarr"
# MEAN=false # whether to only save ensemble mean
ENS_SIZE=50 # number of ensemble members to work with

# Set MEAN and OUTPUT_ZARR based on job index
# if [ "$PBS_ARRAY_INDEX" -eq 1 ]; then
#     MEAN=false
#     OUTPUT_ZARR="/glade/derecho/scratch/aasche/PLASIM/data/ensemble_test/paper/all.zarr"
# elif [ "$PBS_ARRAY_INDEX" -eq 2 ]; then
#     MEAN=true
#     OUTPUT_ZARR="/glade/derecho/scratch/aasche/PLASIM/data/ensemble_test/paper/mean.zarr"
# fi

# Launch the script
# pass --mean if you want to only save the mean
python netcdf_to_zarr.py --input_dir=$INPUT_DIR --output_zarr=$OUTPUT_ZARR --ens_size=$ENS_SIZE