# PanguWeather
v1.0 - Reimplementation of Pangu-Weather paper

v2.0 - Modified implementation of PanguWeather, currently used for PanguPLASIM

# PanguPLASIM

## Code Locations
Midway3: `/project/pedramh/awikner/PanguWeather/`

FASTER: `/scratch/group/p.atm170020.000/PanguWeather-UC/`

Anvil: `/anvil/projects/x-atm170020/awikner/PanguWeather`

## Data Locations

### Pressure Level Data

Midway3: `/scratch/midway3/tvallabh/pangu_data/PLASIM/train_val_test_data_pl`

FASTER: `/scratch/group/p.atm170020.000/PLASIM/data/train_val_test_data_pl/`

Anvil: `/anvil/projects/x-atm170020/awikner/PLASIM/data/train_val_test_data_pl`

### Sigma Level Data

Midway3: `/scratch/midway2/awikner/PLASIM/data/train_val_test_data`

FASTER: `/scratch/user/u.aw164890/PLASIM/train_val_test_data`

Anvil: `/anvil/projects/x-atm170020/awikner/PLASIM/data/train_val_test_data`

## Activating Environments

In general, you should make your own virtual environment on top of the base conda environment.

Midway3:

```
ml python/anaconda-2023.09
conda activate /project/pedramh/anaconda/py311
source /home/awikner/venvs/pangu-wandb/bin/activate
```

FASTER:

```
ml Anaconda3
conda activate /scratch/group/p.atm170020.000/anaconda/py311
source /home/u.aw164890/venvs/pangu/bin/activate
```

Anvil:

```
ml anaconda/2024.02-py311
conda activate /anvil/projects/x-atm170020/anaconda/py311
source /home/x-awikner/venvs/anvil/pangu/bin/activate
```

To create your own virtual environment on top of the existing conda environment, run:
```
ml ${conda_module}
conda activare ${conda_env_path}
mkdir -p ~/venvs/${your_venv_name}
python3 -m venv ~/venvs/${your_venv_name} --system-site-packages
source ~/venvs/${your_venv_name}/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install everything else
```

## Getting Started
1. Either clone or fork this repository from the `optim-dev` branch (this is the main branch we'll be using for optimization) and create and checkout your own branch from it.
2. Before beginning a training or inference run, you'll first need to create a configuration file. These should be stored in the `v2.0/config` directory. The naming convention I've been using is `PANGU_PLASIM_${CLUSTER}_${RUN_NUM}.yaml`. Remember to use a `RUN_NUM` beginning with your assigned number.
   The base configuration file you can edit to create your own can be found at `v2.0/config/BASE_CONFIG.yaml`.
4. Edit your configuration file to set the parameters you'd like to use for the run. Remember to set the `data_dir` to point to the data location for the cluster you're using.
5. If beginning a training run for the first time, log in to your weights and biases account first. This can be done by activating the environment using the information above, then running `wandb login`. You'll be prompted to open a link to login to your account, then will receive an access code to enter in the command line.
6. To start a training, run `sbatch -J ${RUN_NUM} ${cluster}_training.sh ${RUN_NUM} ${CONFIG_FILE_PATH}`
