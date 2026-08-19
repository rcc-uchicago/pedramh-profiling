# Profiling ACE2 on Delta — step by step

Produces **one `.sqlite` file** to send back. Nothing else needs copying, and
the far end needs no nsys installed.

Total time: ~10 minutes if you already run ACE2 on Delta, plus queue.

---

## Step 1 — get the code

```bash
git clone git@github.com:rcc-uchicago/pedramh-profiling.git
cd pedramh-profiling
git checkout fix/tsoi-fill-270
```

Already have it?

```bash
cd pedramh-profiling
git checkout fix/tsoi-fill-270
git pull
```

---

## Step 2 — find your two paths

You need the two things you already use to train ACE2 on Delta:

1. **How you activate python** (the env with `torch` and `fme`). In
   `ACE2_retrain/train.sh` this is:
   ```
   module load python/miniforge3_pytorch && source activate <your env>
   ```
2. **Where your ACE2 data lives on Delta** — the directory containing
   `ace_training/` and `normalization/`. `train.sh` hints it was staged under
   `/work/nvme/bdiu/jlandsberg/`.

`config_nsight.yaml` in this repo points at **Midway** paths
(`/project/pedramh/shared/ACE2_retrain/...`), so it will not work on Delta
unchanged. Generate a Delta version:

```bash
cd ACE2_retrain
python make_cluster_config.py \
    --data-root /work/nvme/bdiu/jlandsberg \
    --experiment-dir $SCRATCH/ace2_profile \
    --out config_delta.yaml
```

It prints every path it wrote and whether each exists — check that list before
step 3. It cannot be done with `--override`: three training paths live inside a
`concat:` list and OmegaConf's dotlist cannot index into lists.

If you do not have an env yet, see the Appendix.

---

## Step 3 — run it

From the repo root:

```bash
cd ACE2_retrain

ACE2_ACTIVATE='module load python/miniforge3_pytorch && source activate /path/to/your/fme/env' \
ACE2_CONFIG=$PWD/config_delta.yaml \
  sbatch delta_bench_nsys.sh
```

Check on it with `squeue -u $USER`. It runs ~5-10 minutes once started.

The script only profiles — it trains for a few dozen steps and stops. It writes
nothing outside `ACE2_retrain/outs/`.

---

## Step 4 — send the file back

When the job finishes, the last line of `ace2_delta_nsys_<jobid>.out` reads:

```
ACE2_DELTA_NSYS_OK rep_mb=... sqlite_mb=... gpus=4
SCP THIS: /.../outs/delta_nsys_<jobid>/nsys_ace2_delta_<jobid>.sqlite
```

Copy that one file:

```bash
scp <delta>:/.../nsys_ace2_delta_<jobid>.sqlite .
```

Also useful, if it is short: the `.out` file itself.

---

## If it fails

The script fails loudly with the reason. The common ones:

| message | fix |
|---|---|
| `ERROR ACE2_ACTIVATE_FAILED` | `ACE2_ACTIVATE` is wrong — paste exactly what you type to activate python |
| `ERROR ACE2_ENV_NOT_ACTIVE` | env activated but has no `torch`/`fme` — wrong env |
| `ERROR ACE2_CONFIG_MISSING` | `ACE2_CONFIG` path is wrong |
| `ERROR ACE2_NSYS_MISSING` | set `ACE2_NSYS_BIN=/path/to/nsys` (often in the CUDA toolkit's `bin/`) |
| `sbatch: error: ... account/partition` | edit the `#SBATCH` lines at the top — defaults were taken from `train.sh` |
| out of memory | add `ACE2_BATCH_SIZE=2` to the command in step 3 |

`nsys rc=<non-zero>` on its own is **not** a failure — nsys stops the app when
the capture window closes. The gate is the `ACE2_DELTA_NSYS_OK` line.

---

## What the trace contains

By default it captures the **whole run** — startup, training, and validation —
because validation is the main suspect and a windowed capture would cut it out.
NVTX ranges are emitted throughout, so the trace is broken down by phase:

| range | what it is |
|---|---|
| `step_{N}` | one training step (outer) |
| `forward_loss` / `backward` / `optimizer` | the shared phase names, comparable to other models |
| `spectral_filter`, `sfno_block`, `sfno_mlp`, `sfno_net` | inside the SFNO |
| `amp_region` | where AMP is switched on |
| `stack`, `normalize`, `denormalize` | the dict/tensor and normalisation work |

It also writes fme's `GlobalTimer` category totals to `outs/.../metrics` as
JSONL — that breakdown normally only reaches wandb, so it is otherwise invisible
in an offline run. Send that directory too if it is small.

Knobs if the first run is the wrong size:

| variable | default | effect |
|---|---|---|
| `ACE2_SAMPLES` | 128 | training samples (÷ batch = steps) |
| `ACE2_VAL_STOP` | `1996-01-05` | end of the validation window |
| `ACE2_BATCH_SIZE` | 4 | global batch |
| `ACE2_NSYS_WINDOWED` | 0 | set to 1 for a training-only capture (smaller, no validation) |

## What this measures, and why

ACE2 was profiled on Midway (H100/H200). Three findings there are properties of
the model rather than the machine, so they should reproduce on Delta:

1. **~61% of an epoch is validation**, and ~52% of that is drawing diagnostic
   images that are **thrown away** when wandb is off. May be worse on Grace ARM
   cores.
2. **~28% of GPU time is tensor copies**, at the fp32/bf16 boundary around the
   spherical harmonic transform.
3. **~9% of training time is kernel-launch latency**, from ~2,900 tiny kernels
   per step per rank.

A fourth, communication cost, was 52% on one Midway node because its GPUs are
only NVLink-paired. Delta's GH200s should not have that, and this capture
confirms it — the script records `nvidia-smi topo -m` into its own output.

---

## Appendix — building the `fme` env from scratch

Only if you have no working ACE2 env on Delta.

```bash
cd pedramh-profiling/ACE2_retrain/ace_exp

module load python/miniforge3_pytorch      # or whatever provides conda
conda create -y --prefix $HOME/envs/fme python=3.11 pip
conda activate $HOME/envs/fme

pip install uv
uv pip install -c constraints.txt -e ".[dev]"

python -c "import torch, fme; print(torch.__version__, torch.cuda.is_available())"
```

Two notes for Delta specifically:

- **GH200 is ARM (aarch64)**, so conda must not pull x86 packages. `ace_exp`'s
  own Makefile handles this (`CONDA_SUBDIR=linux-aarch64`); if you use
  `make create_environment` instead of the above, it is already correct.
- **Do not remove the torch pin** in `constraints.txt`. Dropping it makes the
  resolver install the newest torch and CUDA stack available, which is how a
  Midway build ended up with torch 2.13/cu130 by accident.

Then use `ACE2_ACTIVATE='conda activate $HOME/envs/fme'` in step 3.
