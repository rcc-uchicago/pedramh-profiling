# Handoff: get the three pedramh-profiling models running on ALCF Polaris (PBS)

> Paste everything below the line into a fresh Claude Code session running on a
> **Polaris login node**, inside a checkout of this repo (`pedramh-profiling`).
> It is written **to that Claude**. It was authored from the RCC Midway checkout
> where the Slurm version of all three models already runs end-to-end (the
> `midway_*` scripts in each model dir are GREEN).

---

You are a Claude Code session on **ALCF Polaris**. Your job is to get the three
models in this repo running on Polaris and prove each one with a **small smoke /
bench run on Polaris GPUs**: a job that starts, completes a handful of
optimization (or bench) steps without crashing, and writes a log/CSV proving the
loop closed.

**Scope: cluster bring-up only.** NOT the optimization work (torch.compile /
FlexAttention / DDP-comm-hook — that comes *after* bring-up), NOT paper repro,
NOT hyperparameter tuning, NOT a full training run. Same scope discipline as the
Midway setup: get it to *run correctly on Polaris hardware*, prove it, stop.

Polaris uses **PBS Pro** (`qsub`/`qstat`/`qdel`), not Slurm. This repo's `midway_*`
Slurm scripts are your **template** — your task is the PBS analog of each. If
you've seen the MARSHAL or Decrypto Polaris handoffs, this one is different in an
important way: **there is no container and no server-discovery here.** These are
plain conda-env PyTorch/Lightning jobs. Your porting surface is the scheduler
wrap, the env/module setup, the storage paths, and — the one real trap — the
**multi-GPU launcher** (three different mechanisms across the three models; see
below).

## The three models and their Midway launchers (read these first)

All three share the same model/loss/data-loader code under `s2s/v2.0/`
(`networks/pangu.py`, `utils/losses.py`, `utils/data_loader_multifiles.py`). They
differ in the training harness and how they spawn the 4 ranks.

| Model | Dir | Entry point | Midway launcher (your template) | 4-GPU mechanism |
|---|---|---|---|---|
| **S2S** (canonical, bench-instrumented) | `s2s/v2.0/` | `train.py`, `inference.py` | `s2s/v2.0/HPC_scripts/midway_training.sh`, `midway_bench_nsys.sh` | **`torchrun --standalone --nproc_per_node=4`** |
| **S2S-Lightning** (the port) | `s2s-lightning/` | `train.py`, `bench.py`, `smoke_train_module.py` | `s2s-lightning/midway_smoke_train_module.sh`, `midway_bench_nsys_port.sh` | **Lightning + `srun` (SLURM launcher, `ntasks-per-node=4`)** |
| **SI** (DiT/SiT sibling) | `si/` | `bench.py`, `train.py` | `si/bench_midway.sh`, `si/bench_nvtx.sh` (config `si/configs/SI_midway.yaml`) | **Lightning + `DDPStrategy`** |

Also read:
- **`README.md`** (root) and each model's README — layout + how the port imports `s2s/v2.0`.
- **`s2s-lightning/LIGHTNING_PORT.md`** — the port's Lightning/DDP/AMP wiring (precision mapping, static-graph, bench callback).
- The `s2s/v2.0/config/exp2.yaml` and `si/configs/SI_midway.yaml` configs — `data_dir`, `checkpoint_path`, and the mean/std `.nc` filenames are **cluster-specific** and MUST be repointed to Polaris storage before anything runs (they fail deep in the data loader, not early).

The model/loss/loader code itself (`s2s/v2.0/networks`, `utils`) is
**scheduler-agnostic** — you should not need to touch it. Same for the Lightning
`modules/`, `data/` in `s2s-lightning/` and `si/`.

## The Slurm → PBS porting surface

| Concern | Midway (Slurm) | Polaris (PBS Pro) — what you do |
|---|---|---|
| Submit | `sbatch script.sh` | `qsub script.pbs` |
| Directives | `#SBATCH ...` | `#PBS ...` |
| Account/allocation | `--account=pi-pedramh` | `-A <project>` (your active Polaris allocation) |
| Queue | `-p pedramh-gpu` | `-q debug` for the smoke (see queues below) |
| Walltime | `--time=00:45:00` | `-l walltime=00:45:00` |
| Node + GPU shape | `--nodes=1 --gres=gpu:4` (or `--ntasks-per-node=4`) | `-l select=1:system=polaris` — **allocates a whole node = 4× A100**; add `-l place=scatter` |
| **Filesystems** | (implicit) | `-l filesystems=home:eagle` — **Polaris REJECTS jobs that omit this.** Declare every FS the job reads/writes (home + your project FS on eagle or grand) |
| Job id in script | `$SLURM_JOB_ID` | `$PBS_JOBID` (looks like `1234567.polaris-pbs-...`; use `${PBS_JOBID%%.*}` for a clean numeric tag) |
| Output/error | `-o ...%j.out -e ...%j.err` | `#PBS -o <path> -e <path>` (no `%j`; PBS writes `<jobname>.o<jobid>` by default) |
| List / cancel | `squeue -u $USER` / `scancel` | `qstat -u $USER` / `qstat -f <jid>` / `qdel <jid>` |
| Nodefile | (implicit) | `$PBS_NODEFILE` — `NNODES=$(wc -l < $PBS_NODEFILE)` |
| Env passthrough | `--export=ALL` | prefer setting vars *inside* the script; `qsub -v FOO=bar` for specific, `-V` for full login env |
| Node-local scratch | `/tmp/${USER}_${SLURM_JOB_ID}` | Polaris has a node-local SSD (commonly `/local/scratch`) — use it for `TMPDIR`; **confirm the path on a compute node** |

There is **no `--wrap` and no `srun` on PBS.** Where the Midway script used
`srun`, you replace it (see the launcher section — this is the load-bearing part).

## The multi-GPU launcher — the one real trap (per model)

Polaris gives you a whole node (4 A100s). How you fan out to those 4 GPUs differs
per model, and the Midway mechanisms do **not** all port trivially:

**S2S (`torchrun --standalone`) — easiest.** `torchrun --standalone
--nproc_per_node=4` needs no scheduler integration; it spawns 4 local ranks on
the one allocated node. Keep it almost as-is:
```bash
cd <repo>/s2s
PYTHONPATH=$(pwd)/v2.0 torchrun --standalone --nproc_per_node=4 \
    v2.0/train.py --yaml_config=v2.0/config/exp2.yaml --run_num=0100
```
No `mpiexec`, no affinity script needed for a single node (torchrun's local ranks
bind fine). Only the PBS wrap + module/env + repointed paths change.

**S2S-Lightning (`srun` + Lightning SLURM launcher) — needs a rewrite.** The
Midway bench (`midway_bench_nsys_port.sh`) uses `srun --ntasks-per-node=4` because
Lightning's **SLURM launcher** requires `ntasks == devices`. **Polaris has no
`srun`, and Lightning must NOT think it's under SLURM.** For a single Polaris node,
drop `srun` entirely and let Lightning's default subprocess launcher spawn the 4
ranks from one `python`:
```bash
cd <repo>/s2s-lightning
PYTHONPATH=<repo>/s2s/v2.0:$(pwd) python bench.py \
    --yaml_config <repo>/s2s/v2.0/config/exp2.yaml --config S2S \
    --batch_size 2 --devices 0 1 2 3 --strategy ddp
```
`Trainer(devices=4, num_nodes=1, strategy="ddp")` + the plain-`python` entry uses
Lightning's `subprocess_script`/`ddp` launcher — no `srun`, no MPI. (For
*multi-node* later you'd add a `ClusterEnvironment` plugin or drive it under
`mpiexec`; single-node smoke does not need that.) **Verify Lightning does not
auto-detect a stray SLURM env** — unset any `SLURM_*` vars if present.

**SI (`DDPStrategy`) — same as the port.** `si/bench.py` builds an explicit
`DDPStrategy`; run it as one `python si/bench.py ...` with `devices=4`. No `srun`.

**Optional (ALCF-canonical, if you prefer MPI):** you *may* instead launch any of
these under `mpiexec -n 4 --ppn 4 --depth=8 --cpu-bind depth
./set_affinity_gpus_polaris.sh python ...` (affinity helper from
`argonne-lcf/GettingStarted/Examples/Polaris/affinity_gpu`), but that requires the
training code to init the process group from MPI rank env vars. The torchrun /
Lightning-subprocess paths above are less invasive — prefer them for bring-up.

## Environment on Polaris (replaces Midway's mamba env)

Midway did `module load python/miniforge-25.3.0 && mamba activate
/project/pedramh/shared/S2S/v2.0/venv`. On Polaris:
```bash
module use /soft/modulefiles
module load conda
conda activate base          # ALCF base already has a CUDA-matched PyTorch
```
Decide one of:
- **Use the ALCF base conda** (fastest; confirm `python -c "import torch;
  print(torch.__version__, torch.cuda.is_available())"` sees the A100), then
  `pip install --user pytorch-lightning wandb` on top for the port/SI. OR
- **Build the env** from `s2s/v2.0/environment.yml` (+ `pytorch-lightning` for the
  port and `si/environment.yml` — now `name: si` — for SI) into project storage:
  `conda env create -f s2s/v2.0/environment.yml --prefix /eagle/<project>/<user>/envs/s2s`.
  ⚠️ **Match the torch build to Polaris's CUDA driver** (Polaris is CUDA 12.x). Do
  not blindly reuse a Midway-pinned wheel.

`WANDB_MODE=offline` as on Midway (login nodes have network, compute nodes may not).

## Data staging (do this before any run)

The HDF5 ERA5 dataset is **not** in the repo. On Midway it's at
`/project/pedramh/h5data/h5data`; on Polaris it must live on a Lustre FS
(`/eagle/<project>/...` or `/grand/<project>/...`). Stage it with **Globus** (the
ALCF-blessed path for large inter-center transfers; both RCC and ALCF have
endpoints). Then repoint, in each config you run:
- `exp2.yaml` / `SI_midway.yaml`: `data_dir`, `checkpoint_path`, and the mean/std
  `.nc` filenames → Polaris paths.
- The port reads the same `s2s/v2.0/config/exp2.yaml`, so fixing it once covers S2S + the port.

The `s2s-lightning/data/constant_mask/*.npy` boundary constants ship in the repo —
nothing to stage there.

(The `s2s/v2.0/HPC_scripts/nvidia_*.sh` NGC-container scripts read `$NGC_API_KEY`
from the environment — not relevant to the bare-metal Polaris bring-up; ignore
them unless you deliberately go the apptainer route.)

## Polaris facts to CONFIRM on the cluster (don't trust these blindly)

I'm authoring from Midway and cannot see Polaris. Verify each and record it in
your notes:

- **Allocation/project** for `-A` (need an active Polaris allocation under pedramh).
- **Queue** — use `debug` for smokes (8 dedicated nodes, ≤1 hr, 1 job/user). Confirm
  max nodes/walltime with `qstat -Q`. Others: `debug-scaling`, `prod` (default;
  routes to small/medium/large), `preemptable` (can be killed by `demand`).
- **`-l filesystems=`** — almost certainly `home:eagle` (or `home:grand`). Jobs are
  **rejected** without it; declare every FS touched.
- **GPUs** — **4× A100 40 GB SXM4 per node.** Confirm with `nvidia-smi`. **This is far
  tighter than Midway's H100 NVL (~94 GB) / H200.** The Midway bench ran `exp2`
  `batch_size=8` (2/GPU) at bf16; on 40 GB **expect OOM at those settings** — start
  the smoke at **per-GPU batch 1, bf16**, confirm it fits, then scale. The
  `PanguModel_Plasim` activations are large.
- **CPU / threads** — 1× AMD EPYC "Milan" 32-core. With 4 ranks that's ~8 cores/rank;
  set `OMP_NUM_THREADS` accordingly (Midway used 2).
- **Node-local scratch path** (`/local/scratch`?) for `TMPDIR` and the
  torch-inductor cache — confirm on a compute node.
- **Module incantation** — confirm `module use /soft/modulefiles && module load
  conda` and the base torch version/CUDA.
- **Project storage root** — `/eagle/<Project>/...` vs `/grand/...`.

## Deliverables

Mirror the Midway naming so the two ports sit side by side:

1. `s2s/v2.0/HPC_scripts/polaris_training.pbs` and `polaris_bench.pbs` — PBS analogs of `midway_training.sh` / `midway_bench_nsys.sh` (torchrun path).
2. `s2s-lightning/polaris_smoke_train_module.pbs` and `polaris_bench_nsys_port.pbs` — PBS analogs, **`srun`-free** single-`python` Lightning launch.
3. `si/bench_polaris.pbs` — PBS analog of `bench_midway.sh`.
4. A one-GPU **toolchain probe** PBS script (`select=1:system=polaris`, but run one rank): confirm `nvidia-smi` sees an A100, torch imports, and each model's package imports (`PYTHONPATH` correct). Run and pass this FIRST.
5. **`polaris_pbs_notes.md`** (repo root) — the PBS equivalent of a Midway notes doc: a cluster-facts table (the confirmed values above), the env/module strategy, the per-model GREEN entries (probe → 1-GPU smoke → 4-GPU smoke), repointed-path map, and a dated decisions log. **This is the deliverable that proves the bring-up.**

## Validation ladder (per model, in order)

1. **Probe** — 1 A100: `nvidia-smi`, `import torch` sees CUDA, each model's imports resolve with the Polaris `PYTHONPATH`. Don't proceed until green.
2. **1-GPU smoke** — smallest config, per-GPU batch 1, bf16, a few steps; confirm it completes and writes its log/CSV. (S2S: `torchrun --nproc_per_node=1`; port/SI: `--devices 0`.)
3. **4-GPU DDP smoke** — full node, `devices=4`; confirm all 4 ranks start, DDP all-reduce works, no OOM, log/CSV written.
4. **Bench parity (optional, still bring-up)** — run the S2S_BENCH / SI bench for a few steps and confirm the CSV columns populate; do NOT chase Midway-parity numbers here (A100 ≠ H100 NVL) — that's the later optimization phase.

## Working method

- **One model, one rung at a time.** Get S2S (torchrun — easiest) green top-to-bottom
  first; it de-risks the env/paths/queue for the other two. Then the port, then SI.
- **Read the `.err` first** when a job fails — most bring-up failures are path/module/OOM, visible immediately in stderr. (On Midway a whole class of "missing kernels" turned out to be an early crash before any GPU work — same discipline applies.)
- **Never claim a rung passed without reading the job's actual output.** Smoke scripts print a success token / write a CSV row — key on that, not on exit code alone.
- Keep the Slurm scripts intact; **add** `polaris_*` files beside them. Don't break the Midway path.
- Record every confirmed Polaris fact and every path you repointed in `polaris_pbs_notes.md` as you go.
