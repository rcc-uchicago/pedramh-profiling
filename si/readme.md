## Requirements

To install requirements:
```setup
conda create -n "my_env" python=3.13
conda install pip 
pip install torch torchvision
pip install lightning matplotlib wandb h5py timm einops h5pickle
```
Currently running with pytorch 2.10 and CUDA 12.8

To run SFNO:
```
conda install torch-harmonics
pip install -U tensorly tensorly-torch
```

qstat -u ayz | grep "^[0-9]" | cut -d'.' -f1 | xargs -I {} qdel {}.desched1