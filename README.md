## Pangu-Weather
Reimplementation of Pangu-Weather paper

### Usage
Adjust the hyperparameters and model parameters in 

```bash
v1.0/config/PANGU.yaml
```

Train the model

```bash
v1.0/python train.py
```

Run on an HPC cluster such as FASTER or Anvil:

FASTER:
```bash
sbatch v1.0/faster_ddp.sh
```

Anvil:
```bash
sbatch v1.0/anvil_ddp.sh
```