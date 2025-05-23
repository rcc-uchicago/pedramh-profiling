# PlaSEM
v1.0 - Reimplementation of Pangu-Weather paper

v2.0 - Modified implementation of PanguWeather, currently used for PanguPLASIM emulator. Also contains the modulus implementation of the SFNO that can be trained on PlaSim.

# PanguPLASIM

## Getting Started
1. Either clone or fork this repository from the `main` branch and create and checkout your own branch from it.
2. Create a conda environment using the environment file at `v2.0/env_files/pangu_sfno_env.yml`.

## Training
1. Either clone or fork this repository from the `main` branch and create and checkout your own branch from it.
2. Before beginning a training or inference run, you'll first need to create a configuration file. These should be stored in the `v2.0/config` directory. The naming convention I've been using is `PANGU_PLASIM_H5_${CLUSTER}_${RUN_NUM}.yaml`.
4. Edit your configuration file to set the parameters you'd like to use for the run. Remember to set the `data_dir` to point to the data location for the cluster you're using.
5. If beginning a training run for the first time, log in to your weights and biases account first. This can be done by activating the environment using the information above, then running `wandb login`. You'll be prompted to open a link to login to your account, then will receive an access code to enter in the command line.
6. To start a training, run `sbatch -J ${RUN_NUM} ${cluster}_training.sh ${RUN_NUM} ${CONFIG_FILE_PATH}`. Some submit scripts may require additional inputs, so check them before you use them.

## Inference
For ensemble inference, use `v2.0/ensemble_inference.py`. For long inferences, use `v2.0/long_inference`.
