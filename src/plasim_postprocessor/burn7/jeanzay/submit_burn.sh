#!/bin/bash
#SBATCH -A huw@cpu
#SBATCH -p cpu_p1
#SBATCH --job-name=postprocess_output
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1      # number of MPI tasks per node
#SBATCH --hint=nomultithread 
#SBATCH --cpus-per-task=1 
#SBATCH --time=00:20:00              # temps maximum d'execution demande (HH:MM:SS)

module purge
module load intel-oneapi-compilers/2023.1.0 intel-mpi/2021.9

# DIR=/glade/derecho/scratch/awikner/PLASIM/postprocessor
DIR=/lustre/fswork/projects/rech/huw/uoj62aw/PLASIM-utilities/PLASIM/postprocessor2.0/burn7/jeanzay
# export LD_LIBRARY_PATH=/glade/u/apps/derecho/23.09/spack/opt/spack/netcdf/4.9.2/packages/netcdf-c/4.9.2/gcc/13.2.0/chn4/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$CONDA_PREFIX/lib
cd $DIR
PATH_NL=$1
INPUT_FILE=$2
OUTPUT_FILE=$3
echo "PATH_NL: $PATH_NL"
echo "INPUT_FILE: $INPUT_FILE"
echo "OUTPUT_FILE: $OUTPUT_FILE"

./burn7 < $PATH_NL $INPUT_FILE $OUTPUT_FILE
echo "=============================================================="