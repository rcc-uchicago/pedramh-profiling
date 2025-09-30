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

# Data Structure

## Overview
DATA_DIRECTORY/
├── InputOutput/ # raw data, one HDF5 file per sample
│ ├── {year}{year_time_idx:04}.h5
│ │ └── group: "input"
│ │ ├── var2d_name(lat, lon) → float32
│ │ ├── var3d_name{int(level)}.0(lat, lon) → float32 # pressure levels
│ │ └── var3d_name_{level}(lat, lon) → float32 # sigma levels
│ │ * missing values set to NaN
│ │
│ ├── ... (other HDF5 files, one per time sample)
│ │
│ ├── mean/std.nc # normalization statistics
│ │ ├── coords:
│ │ │ ├── Z (pressure levels, descending) → float64
│ │ │ └── Z_2 (sigma levels, descending) → float64
│ │ ├── data:
│ │ │ ├── 2D variable mean, std → single float32 values
│ │ │ └── 3D variable mean, std → 1D float32 arrays along Z or Z_2
│ │
│ ├── ... (can be multiple files if non-full-field normalization is used)
│ │
│ └── daily_climatology.nc
│ ├── coords:
│ │ ├── time → 366 cftime.datetime values (day of year, leap included)
│ │ ├── lat → float64
│ │ ├── lon → float64
│ │ ├── plev → float64 (descending, pressure levels)
│ │ └── lev → float64 (descending, sigma levels, optional)
│ └── data variables:
│ ├── 3D: (time, lat, lon) → float32
│ └── 4D: (time, plev/lev, lat, lon) → float32
│
└── bias/ # separate directory for climatological biases
├── 2D variables:
│ ├── {var}bias.npy → float32
│ └── {var}bias{init_hour}z.npy → float32 # if timestep mismatch
│
└── 3D variables:
├── {var}{int(level)}.0_bias.npy → float32 # pressure levels
├── {var}{level}bias.npy → float32 # sigma levels
├── {var}{level}.0_bias{init_hour}z.npy → float32
└── {var}_{level}bias{init_hour}z.npy → float32


---

## Data Types
All data except for the biases should be saved in a single directory.

---

### Input and Output Data
- The raw **unstandardized data** used as input and target model output should be saved as HDF5 files, each containing a sample of the data at one time.
- Each file contains a single top-level group named **`"input"`**.
- Within this group:
  - Each variable at each vertical level is saved as a **2D dataset** with axes `(lat, lon)` and `float32` data type.
  - For **2D variables**, the dataset name is the variable name.
  - For **3D atmospheric variables**:
    - Pressure levels:
      `f"{variable_name}_{int(level)}{level_unit}"`, with default `level_unit = ".0"`.
    - Sigma levels:
      `f"{variable_name}_{level}"`.

- Missing data → `NaN`. Masking values can be set in the config file.
- File naming convention:
  `f"{year}_{year_time_idx:04}.h5"`, where `year_time_idx` starts at `0` (0z Jan. 1st).
- If yearly samples exceed 10,000, the data loader must be modified.

---

### Standardization Data
- Saved in **NetCDF** files.
- Coordinates (dtype `float64`):
  - `Z`: descending pressure levels
  - `Z_2`: descending sigma levels
- Statistics:
  - 2D → single `float32` values
  - 3D → 1D `float32` arrays along `Z` or `Z_2`

⚠️ If training with tendencies or ACE’s residual normalization, full-field mean and std must be provided.

---

### Climatology Data
- Daily climatology (for validation ACC) stored in a **single NetCDF file**.
- Coordinates:
  - `time`: 366 `cftime.datetime` values (day of year, leap included)
  - `lat`, `lon`: `float64`
  - `plev`: descending pressure levels (`float64`)
  - `lev`: descending sigma levels (`float64`, optional)
- Data variables:
  - 3D: `(time, lat, lon)` → `float32`
  - 4D: `(time, plev/lev, lat, lon)` → `float32`

---

### Annual Climatological Bias Data
- Only used if `long_validation = True` and either:
  - Model timestep = 24h, OR
  - Model timestep < 24h and equals data timestep.

- Saved as `.npy` (`float32`) in a **bias directory**.

File naming:
- 2D:
  - `{variable}_bias.npy`
- 3D:
  - Pressure levels: `{variable}_{int(level)}.0_bias.npy`
  - Sigma levels: `{variable}_{level}_bias.npy`

If model timestep = 24h but data timestep < 24h, bias files must be time-of-day specific, e.g. `{variable}_bias_{init_hour}z.npy`.

---

## Config File Data Variables
```yaml
data_dir: Absolute path to data directory (DATA_DIRECTORY/InputOutput)
upper_air_variables: list of prognostic atmospheric variables
surface_variables: list of prognostic surface variables
diagnostic_variables: list of diagnostic variables
land_variables: list of prognostic land variables
ocean_variables: list of prognostic ocean variables
constant_boundary_variables: list of constant boundary variables
varying_boundary_variables: list of varying boundary variables
train_year_start: first year of training (inclusive)
train_year_end: last year of training (non-inclusive)
val_year_start: first year of validation (inclusive)
val_year_end: last year of validation (non-inclusive)
data_timedelta_hours: timestep between samples (hours)
*_mean: normalization means file (upper_air, surface, boundary, diagnostic)
*_std: normalization std file (upper_air, surface, boundary, diagnostic)
*_ff_mean: full-field normalization means (required if predict_delta = True)
*_ff_std: full-field normalization stds (required if predict_delta = True)
climatology_file: name of climatology file (within data_dir)
calendar: cftime.datetime calendar type
has_year_zero: boolean
num_levels: total vertical levels
use_sigma_levels: boolean
levels: list of pressure levels
sigma_levels: list of sigma levels (if used)
predict_delta: boolean (True = predict tendencies, False = full-fields)
lev: "plev" (pressure) or "lev" (sigma)
lat: list of latitude values (in data order)
lon: list of longitude values (in data order)


