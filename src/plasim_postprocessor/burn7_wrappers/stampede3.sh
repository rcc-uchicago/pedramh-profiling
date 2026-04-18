#!/bin/bash
# burn7_wrappers/stampede3.sh
#
# Cluster-specific burn7 environment wrapper for Stampede3 (TACC).
# Called by plasim_postprocessor.py as:
#   stampede3.sh <namelist_path> <input_file> <output_file>

set -e

NAMELIST=$1
INPUT_FILE=$2
OUTPUT_FILE=$3

if [ -z "$NAMELIST" ] || [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <namelist> <input_file> <output_file>" >&2
    exit 1
fi

module purge
module load gcc netcdf

export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:${LD_LIBRARY_PATH}

BURN7_DIR="$(cd "$(dirname "$0")/../../postprocessor2.0/burn7/Stampede3" && pwd)"

"$BURN7_DIR/burn7" < "$NAMELIST" "$INPUT_FILE" "$OUTPUT_FILE"
