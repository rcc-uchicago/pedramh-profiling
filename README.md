# AI-RES

AI + Rare Event Sampling experiments on climate / weather models, on Stampede3.

## Layout

Code and lightweight files live under `$HOME`. Heavy artifacts live on `$SCRATCH` / `$WORK` and are exposed here as symlinks (gitignored).

| Path           | Where it lives                             | Purpose                               |
| -------------- | ------------------------------------------ | ------------------------------------- |
| `src/`         | `$HOME`                                    | Python / core implementation          |
| `scripts/`     | `$HOME`                                    | Shell + SLURM job scripts             |
| `skills/`      | `$HOME`                                    | Claude Code skills (SKILL.md)         |
| `configs/`     | `$HOME`                                    | Experiment configs (YAML / JSON)      |
| `notebooks/`   | `$HOME`                                    | Exploratory analysis                  |
| `tests/`       | `$HOME`                                    | Unit / integration tests              |
| `docs/`        | `$HOME`                                    | Documentation                         |
| `logs/`        | `$HOME`                                    | Job logs (small)                      |
| `data/`        | → `$SCRATCH/AI-RES/data`                   | Raw & intermediate simulation data    |
| `checkpoints/` | → `$SCRATCH/AI-RES/checkpoints`            | Model checkpoints / training state    |
| `results/`     | → `$WORK/AI-RES/results`                   | Curated results for medium-term keep  |

## Stampede3 paths

```
$HOME    = /home1/11114/zhixingliu
$SCRATCH = /scratch/11114/zhixingliu        # purged periodically — backups required
$WORK    = /work2/11114/zhixingliu/stampede3 # medium-term, no backup
```
