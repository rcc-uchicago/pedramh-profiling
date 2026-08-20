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

**You submit one script, but three files must be present in `ACE2_retrain/`:**

| file | why |
|---|---|
| `delta_bench_nsys.sh` | the only thing you `sbatch` |
| `ace2_nvtx.py` | the script above launches **this**, not `fme.ace.train` — it injects the NVTX ranges that give the trace its phase labels |
| a config (step 2) | `config_delta.yaml`, or your own |

Copying only `delta_bench_nsys.sh` to Delta produces a job that dies at
`torchrun` with a missing-file error that does not obviously point at
`ace2_nvtx.py`. Clone the repo; do not cherry-pick the script.

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

## Step 4 — send the bundle back

The run produces artifacts in three different directories — the trace, fme's
python log, the GlobalTimer metrics, the resolved config, and SLURM's own
`.out`/`.err`. The script collects all of them into **one tarball** so a
partial transfer is detectable rather than silent.

The last lines of `ace2_delta_nsys_<jobid>.out` read:

```
ACE2_DELTA_NSYS_OK   rep_mb=... sqlite_mb=... gpus=4
ACE2_DELTA_BUNDLE_OK files=7 size_mb=...
SCP THIS: /.../ACE2_retrain/outs/ace2_delta_<jobid>.tar.gz
sha256:   3f9c...
verify at the far end:  sha256sum ace2_delta_<jobid>.tar.gz   # expect 3f9c...
manifest without extracting: /.../outs/ace2_delta_<jobid>.MANIFEST.txt
```

Copy it and check it arrived intact:

```bash
scp <delta>:/.../ACE2_retrain/outs/ace2_delta_<jobid>.tar.gz .
sha256sum ace2_delta_<jobid>.tar.gz      # compare against the sha256 printed above
tar -xzf ace2_delta_<jobid>.tar.gz
cat MANIFEST.txt                          # per-file sha256 + byte sizes
```

**What's in it:**

| file | what it is |
|---|---|
| `nsys_ace2_delta_<jobid>.sqlite` | the trace — all analysis reads this; needs no nsys at the far end |
| `fme_out.log` | fme's own python log: per-step losses, epoch timings |
| `metrics/` | GlobalTimer category breakdown (train / valid / inference split) |
| `resolved_config.yaml` | the config **after** `--override`, i.e. what actually ran |
| `slurm.out`, `slurm.err` | scheduler logs, including the NCCL/topology banner |
| `MANIFEST.txt` | job id, node, GPU model, git commit, torch/NCCL/fme versions, topology, and a sha256 + size for every file above |

Two things worth knowing:

- The **`.nsys-rep` is deliberately excluded** — it is hundreds of MB and the
  `.sqlite` is derived from it. Set `ACE2_BUNDLE_REP=1` if you need the raw
  report to open the timeline in the Nsight GUI.
- **`slurm.out` in the bundle is truncated by one line.** SLURM is still writing
  to it when the tarball is made, so the final `ACE2_DELTA_NSYS_OK` line is not
  in the captured copy. The manifest says so; nothing is wrong.

A checksum mismatch means re-transfer. A missing entry — most often an empty
`metrics/` — means that part of the run did not produce output, which is worth
knowing *before* the analysis rather than halfway through it.

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
3. **~19% of training time is GPU idle, and it is host synchronisation, not
   launch latency.** An earlier version of this file said launch latency, from
   ~2,900 tiny kernels per step per rank. That was refuted: the CPU runs
   thousands of kernels ahead (median launch→execute queue depth **8.87 ms**)
   and spends only 7.5% of the window in `cudaLaunchKernel`. The cost is
   **`cudaStreamSynchronize` — ~7 calls per step, 163 ms of a 346 ms step.**
   Expect the Delta capture to test this, not confirm it.

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
