# Pangu Weather Project Environment Tracker
# WARNING: THIS IS OUTDATED

## Current Environment

- Python version: 3.11
- CUDA version: 12.1
- cuDNN version: 8.9.2
- GCC/G++ version: 12.3.0 (must point to the conda environment)

## Package Requirements

| Package Name | Version | Installation Method | Notes |
|--------------|---------|---------------------|-------|
| PyTorch | 2.3.0+cu121 | pip | |
| ninja | Latest | pip | |
| CUDA | 12.1 | conda | |

## New Package Installations

| Package Name | Version | Installation Method | Reason for Addition |
|--------------|---------|---------------------|---------------------|
| cuDNN | 8.9.2 | conda | Required for Transformer Engine. Note: TE may have trouble finding cuDNN if it's named differently (e.g., nvidia-cudnn-cu12) |
| Transformer Engine | Latest stable | pip | `pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable` |
| GCC/G++ | 12.3.0 | conda | Required for compilation, must point to the conda environment |

## Package Updates/Downgrades

| Package Name | Old Version | New Version | Reason for Change |
|--------------|-------------|-------------|-------------------|
| | | | |

## Functionality Tests

| Script | Status | Notes |
|--------|--------|-------|
| train.py | | |
| inference.py | | |

## Environment Variables

- CUDA_HOME
- CUDA_PATH

Note: Ensure that these environment variables point to the correct software installations within your conda environment.

## Known Issues and Workarounds

1. Transformer Engine installation might require adjustments to find cuDNN, especially if cuDNN is named differently in your system (e.g., nvidia-cudnn-cu12).
2. Ensure compatibility between CUDA, cuDNN, PyTorch, and Transformer Engine versions.
3. Make sure GCC/G++ from the conda environment is used for compilation.

## TODO

- [ ] Regularly update this file with any new package installations or version changes
- [ ] Perform functionality tests after any significant environment changes
- [ ] Document any new issues or workarounds discovered during development
- [ ] Verify that Transformer Engine can locate cuDNN correctly
- [ ] Ensure GCC/G++ from conda environment is properly configured and used
- [ ] Verify that environment variables (CUDA_HOME, CUDA_PATH) point to the correct locations in the conda environment

