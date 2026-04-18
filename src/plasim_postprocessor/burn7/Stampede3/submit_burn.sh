#!/bin/bash
#SBATCH -J pumaburn
#SBATCH -p skx
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A TG-ATM170020

module purge
module load gcc netcdf

# DIR=/glade/derecho/scratch/awikner/PLASIM/postprocessor
# DIR=/work/10165/alancelin/stampede3/AI-RES/RES/namelists_postproc/Stampede3
# export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:${LD_LIBRARY_PATH}


# DIR=/work2/09979/awikner/stampede3/PLASIM/postprocessor
DIR=/work/10165/alancelin/stampede3/AI-RES/PLASIM/postprocessor2.0/burn7/Stampede3
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:${LD_LIBRARY_PATH}


cd $DIR
PATH_NL=$1
INPUT_FILE=$2
OUTPUT_FILE=$3
echo "PATH_NL: $PATH_NL"
echo "INPUT_FILE: $INPUT_FILE"
echo "OUTPUT_FILE: $OUTPUT_FILE"

./burn7 < $PATH_NL $INPUT_FILE $OUTPUT_FILE
echo "=============================================================="