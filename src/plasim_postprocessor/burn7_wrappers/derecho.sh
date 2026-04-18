#!/bin/bash
# burn7_wrappers/derecho.sh
#
# Cluster-specific burn7 environment wrapper for Derecho (NCAR).
# Called by plasim_postprocessor.py as:
#   derecho.sh <namelist_path> <input_file> <output_file>
#
# Sets up the required modules and LD_LIBRARY_PATH, then invokes burn7.
# The burn7 binary stays at its existing location under postprocessor2.0/burn7/.
# This wrapper only sets up the environment; it does not move or copy the binary.

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

export LD_LIBRARY_PATH=/glade/u/apps/derecho/23.09/spack/opt/spack/netcdf/4.9.2/packages/netcdf-c/4.9.2/gcc/13.2.0/chn4/lib

# Resolve burn7 binary relative to this wrapper's location:
# wrapper is at postprocessor/burn7_wrappers/derecho.sh
# binary  is at postprocessor2.0/burn7/derecho/burn7
BURN7_DIR="$(cd "$(dirname "$0")/../../postprocessor2.0/burn7/derecho" && pwd)"

"$BURN7_DIR/burn7" < "$NAMELIST" "$INPUT_FILE" "$OUTPUT_FILE"
