#!/bin/bash -l
#PBS -N random
#PBS -l select=1:system=polaris:ngpus=4:gputype=A100
#PBS -l place=scatter
#PBS -q debug 
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle                          
#PBS -A lighthouse-uchicago
#PBS -e logs/
#PBS -o logs/

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

# Change to working directory
cd ${PBS_O_WORKDIR}

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=$(nvidia-smi -L | wc -l)
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind core --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads python train.py
