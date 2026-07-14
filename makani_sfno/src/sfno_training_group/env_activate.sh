#!/bin/bash
# Activate the group_pangu_sfno_v2 conda env on Stampede3.
# Source this from a slurm script:
#   source "$REPO_ROOT/src/sfno_training_group/env_activate.sh"
set -euo pipefail

module purge
# Stampede3: gcc must be loaded BEFORE cuda (cuda/12.4 has gcc dependency).
module load gcc/13.2.0
module load cuda/12.4

source /work2/11114/zhixingliu/stampede3/miniforge3/etc/profile.d/conda.sh
conda activate /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/envs/group_pangu_sfno_v2

export GROUP_PANGU_ROOT="/work2/09979/awikner/stampede3/PanguWeather/v2.0"
export PYTHONPATH="${GROUP_PANGU_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
