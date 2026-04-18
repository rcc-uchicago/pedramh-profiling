# PlaSim Postprocessor

Converts PlaSim binary output to NetCDF files. This is the unified v3.0 postprocessor used by AI-RES experiments.

## Quick start

```bash
git clone -b feature/plasim-postprocessor git@github.com:amaurylancelin/AI-RES.git
cd AI-RES
```

## Directory layout

```
RES/
├── postprocessor/                    # v3.0 (this directory)
│   ├── plasim_postprocessor.py       # Main conversion script
│   ├── burn7_wrappers/               # Cluster-specific environment wrappers
│   │   ├── derecho.sh                # NCAR Derecho
│   │   └── stampede3.sh              # TACC Stampede3
│   └── config/                       # Experiment YAML configs
│       ├── EXP15_postproc.yaml
│       ├── EXP25_postproc.yaml
│       └── EXP26_postproc.yaml
├── postprocessor2.0/
│   ├── burn7/
│   │   ├─��� burn7.cpp                 # burn7 C++ source (~7000 lines)
│   │   ├── makefile                  # Generic makefile
│   │   ├── make_burn.sh              # Derecho-specific build script
│   │   └── example.nl                # Example burn7 namelist
│   ├── plasim_variables.json         # Variable name mapping
│   └── src/
│       ├── compute_zg.ncl            # Z500 computation via NCL (legacy)
│       └── interpolate_data.ncl      # Sigma-to-pressure interpolation
└── namelists_postproc/               # burn7 namelists per experiment
    ├── EXP15_AIRES.nl
    ├── EXP25_AIRES.nl
    ├── EXP26_AIRES.nl
    └── ...
```

## Prerequisites

| Dependency | Purpose | Install |
|------------|---------|---------|
| **GCC (g++)** | Compile burn7 | `module load gcc` |
| **NetCDF C library** | burn7 I/O | `module load netcdf` |
| **NetCDF C++ library** | burn7 C++ bindings | See [Compiling burn7](#compiling-burn7) |
| **CDO** | NetCDF merging, regridding, precipitation accumulation | `module load cdo` or `conda install -c conda-forge cdo` |
| **Python 3** + **PyYAML** | Run the postprocessor | `pip install pyyaml` |
| **NCL** (optional) | Legacy Z500 computation via `hydro()` | Only needed if `zg_source: "ncl"` |

## Compiling burn7

burn7 converts PlaSim binary dumps to NetCDF. It must be compiled on each cluster because it links against cluster-specific NetCDF libraries.

### Step 1: Locate your NetCDF paths

```bash
# Find NetCDF C headers and libs
nc-config --includedir   # e.g. /usr/include
nc-config --libdir       # e.g. /usr/lib64

# Find or install NetCDF C++ (netcdf-cxx4)
# Some clusters have it as a module; otherwise build from source:
# https://github.com/Unidata/netcdf-cxx4
```

### Step 2: Compile

```bash
cd RES/postprocessor2.0/burn7

g++ -O2 -o burn7 burn7.cpp \
    -I<NETCDF_C_INCLUDE> \
    -I<NETCDF_CXX_INCLUDE> \
    -L<NETCDF_C_LIB> \
    -L<NETCDF_CXX_LIB> \
    -lm -lnetcdf_c++ -lnetcdf
```

Replace `<NETCDF_C_INCLUDE>`, etc. with paths from Step 1.

### Derecho example (NCAR)

The existing `make_burn.sh` compiles for Derecho:

```bash
module load gcc netcdf
cd RES/postprocessor2.0/burn7
bash make_burn.sh
```

### Verify

```bash
echo "Test" | ./burn7    # Should print burn7 version header, then exit
```

## Adding a new cluster

Create a wrapper script at `RES/postprocessor/burn7_wrappers/<cluster>.sh`:

```bash
#!/bin/bash
# burn7_wrappers/<cluster>.sh
set -e

NAMELIST=$1
INPUT_FILE=$2
OUTPUT_FILE=$3

if [ -z "$NAMELIST" ] || [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <namelist> <input_file> <output_file>" >&2
    exit 1
fi

# Load your cluster's modules
module purge
module load gcc netcdf   # Adjust for your cluster

# Set library path if needed
export LD_LIBRARY_PATH=/path/to/netcdf/lib:${LD_LIBRARY_PATH}

# Resolve burn7 binary (compile it first, place in burn7/<cluster>/)
BURN7_DIR="$(cd "$(dirname "$0")/../../postprocessor2.0/burn7/<cluster>" && pwd)"

"$BURN7_DIR/burn7" < "$NAMELIST" "$INPUT_FILE" "$OUTPUT_FILE"
```

Then update your experiment YAML config to point to the new wrapper:

```yaml
postprocessing:
  burn7_wrapper: "/path/to/AI-RES/RES/postprocessor/burn7_wrappers/<cluster>.sh"
```

## Usage

### Standalone (single file conversion)

```bash
python3 RES/postprocessor/plasim_postprocessor.py \
    --config RES/postprocessor/config/EXP15_postproc.yaml \
    --input /path/to/plasim_binary_dump \
    --plasim_output /path/to/output.nc
```

### With Pangu regridding (for Pangu-Plasim experiments)

```bash
python3 RES/postprocessor/plasim_postprocessor.py \
    --config RES/postprocessor/config/EXP25_postproc.yaml \
    --input /path/to/plasim_binary_dump \
    --plasim_output /path/to/plasim_out.nc \
    --pangu_output /path/to/pangu_input.nc
```

### Within AI-RES (automatic)

Set `postprocessor_version: "3.0"` in your experiment JSON config. The QDMC driver calls `plasim_postprocessor.py` automatically for each particle at each step.

## Processing pipeline

```
PlaSim binary dump
        │
        ▼
  1. burn7 (sigma-level extraction)
        │  Extracts specified variables (ta, ua, va, hus, pl, tas, ...)
        │  from PlaSim binary format to NetCDF on sigma levels
        ▼
  2. Z500 computation
        │  Adds geopotential height at 500 hPa
        │  Method: burn7 VTYPE=P (default) or NCL hydro() (legacy)
        ▼
  3. Precipitation accumulation (optional)
        │  Computes 6h/24h running accumulations using CDO runmean
        ▼
  4. Pangu regridding (optional)
        │  Regrids T42 → 0.25° Pangu grid using CDO remapbil
        ▼
   NetCDF output(s)
```

## YAML config reference

```yaml
postprocessing:
  # Variables to extract from PlaSim binary
  upper_air_variables: ["ta", "ua", "va", "hus"]
  surface_variables: ["pl", "tas"]
  land_variables: []
  ocean_variables: []

  # Z500 computation method: "burn7" (recommended) or "ncl" (legacy)
  zg_source: "burn7"
  pressure_levels: [500]

  # Precipitation accumulation
  accumulate_precip: false
  precip_accumulation_hours: [6, 24]

  # Output targets
  outputs:
    plasim:
      enabled: true
    pangu:
      enabled: false       # Set true for Pangu-Plasim experiments
      grid_file: ""        # Path to 0.25° Pangu grid description
      variables: []

  # Tool paths
  burn7_wrapper: "/path/to/burn7_wrappers/<cluster>.sh"
  cdo_path: "cdo"
```

## burn7 reference

- Quick reference card: `PLASIM/postprocessor/burn7qr.pdf`
- Compilation notes: `PLASIM/postprocessor/README_POSTPROCESSOR`
- Namelist examples: `RES/postprocessor2.0/burn7/example.nl`

## Troubleshooting

**burn7 compilation fails with "netcdf_c++ not found":**
You need the NetCDF C++ legacy bindings (netcdf-cxx4). Check if your cluster provides it as a module, or build from source: https://github.com/Unidata/netcdf-cxx4

**"CDO not found" at runtime:**
Load CDO via `module load cdo` before running, or install via conda: `conda install -c conda-forge cdo`

**Z500 values have a -20 to -30 m bias:**
You are likely using `zg_source: "ncl"`. Switch to `zg_source: "burn7"` which uses direct pressure-level extraction and eliminates the interpolation bias.
