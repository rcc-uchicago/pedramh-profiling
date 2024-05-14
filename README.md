## Pangu-Weather
Reimplementation of Pangu-Weather paper

### Usage
Adjust the hyperparameters and model parameters in 

```bash
config/PANGU.yaml
```

Train the model

```bash
python train.py
```

Run on an HPC cluster such as FASTER or Anvil:

FASTER:
```bash
sbatch faster_ddp.sh
```

Anvil:
```bash
sbatch anvil_ddp.sh
```