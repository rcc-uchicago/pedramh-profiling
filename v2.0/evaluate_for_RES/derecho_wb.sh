#!/bin/bash -l
#PBS -N wb_eval
#PBS -l select=1:ncpus=8:mem=235G
#PBS -q main
#PBS -l walltime=01:00:00
#PBS -A UCHI0014
#PBS -o /glade/u/home/aasche/PanguWeather/v2.0/logs
#PBS -j oe
# PBS -J 0-2

export PYTHONPATH=/glade/u/home/aasche/weatherbench2:$PYTHONPATH

ml conda
conda activate aires

# export PYTHONPATH="/glade/u/home/aasche/weatherbench2/weatherbench2:$PYTHONPATH"

echo "Job started at: $(date)"
which python
python --version

cd /glade/u/home/aasche/PanguWeather/v2.0
# mkdir -p logs # Optional: safer to make logs directory

CONFIG="/glade/u/home/aasche/PanguWeather/v2.0/config/ENSEMBLE_EVAL_DERECHO.yaml"
CONFIG_NAME="base"
# CONFIG_NAME="ta"

# CONFIG_NAMES=("tas" "zg" "ua" "hus")
# CONFIG_NAMES=("chicago" "france" "pnw")
# CONFIG_NAME=${CONFIG_NAMES[$PBS_ARRAY_INDEX]}
# echo "Running configuration: $CONFIG_NAME"

python -u wb_eval.py --yaml_config=$CONFIG --config=$CONFIG_NAME