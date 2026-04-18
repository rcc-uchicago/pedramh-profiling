 #!/bin/bash
#PBS -N pumaburn
#PBS -A URIC0009
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mem=4GB

module purge
module load gcc netcdf

# DIR=/glade/derecho/scratch/awikner/PLASIM/postprocessor
DIR="$WORK/PLASIM/postprocessor2.0/burn7/derecho"
export LD_LIBRARY_PATH=/glade/u/apps/derecho/23.09/spack/opt/spack/netcdf/4.9.2/packages/netcdf-c/4.9.2/gcc/13.2.0/chn4/lib

cd $DIR
PATH_NL=$1
INPUT_FILE=$2
OUTPUT_FILE=$3
echo "PATH_NL: $PATH_NL"
echo "INPUT_FILE: $INPUT_FILE"
echo "OUTPUT_FILE: $OUTPUT_FILE"

./burn7 < $PATH_NL $INPUT_FILE $OUTPUT_FILE
echo "=============================================================="