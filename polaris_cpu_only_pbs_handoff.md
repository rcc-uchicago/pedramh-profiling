# Running CPU-only PBS jobs on Polaris (ALCF)

A self-contained handoff for someone who has an ALCF account and a Polaris allocation
and wants to run **CPU-only work** — data conversion, preprocessing, statistics,
parameter sweeps, anything with no GPU kernel in it. No project-specific paths or
account names appear here; substitute your own wherever you see `<...>`.

Everything in the tables below was **queried from PBS or measured on a compute node on
2026-09-22**, not quoted from documentation. The validation record is §12.

---

## 1. The one structural fact to internalise

> **Polaris has no CPU-only partition.** Every compute node is a GPU node
> (1× AMD EPYC 7543P + 4× A100), and PBS allocates them `sharing = force_exclhost` —
> **whole node, exclusively, always**. There is no way to ask for "just 8 cores".

Three consequences that decide how you should plan CPU work:

1. **A CPU-only job costs exactly the same as a GPU job**: full node-hours for the
   whole node, with the four A100s sitting idle. Your allocation does not get a
   discount for not using them.
2. **Therefore: fill the node.** A CPU-only job that runs one single-threaded Python
   process is burning 63/64 of what it is charged. If your work is embarrassingly
   parallel, pack as many units per node as memory allows (§9).
3. **You get the whole 64-way CPU and ~503 GiB of RAM for free** — no `--mem` request,
   no core count to negotiate, no other user's job on your node. That is genuinely
   generous for preprocessing work; the constraint is queue time, not resources.

A "CPU-only job" on Polaris is therefore just a normal node allocation in which you
never touch the GPUs. Make that explicit so no library grabs one by accident:

```bash
export CUDA_VISIBLE_DEVICES=""     # nothing can silently initialise a GPU context
```

---

## 2. Node hardware (measured, one node)

| Item | Value | Source |
|---|---|---|
| CPU | **AMD EPYC 7543P**, 1 socket, **32 physical cores / 64 hardware threads** (SMT-2) | `lscpu` on-node |
| PBS view | `resources_available.ncpus = 64`, `pcpus = 64` | `pbsnodes` |
| RAM | `resources_available.mem = 527672488 kb` ≈ **503 GiB usable** | `pbsnodes` |
| NUMA | **4 domains (NPS4)**, 8 physical cores each, ~128 GB each | measured |
| NUMA→CPU map | node0 `0-7,32-39` · node1 `8-15,40-47` · node2 `16-23,48-55` · node3 `24-31,56-63` | measured |
| SMT siblings | logical CPU `i` and `i+32` are the **same physical core** — so CPUs `0-31` are 32 *distinct* cores and `32-63` are their siblings | derived from the map above; confirmed by binding masks (§12) |
| GPUs (present, ignorable) | 4× A100, `resources_available.ngpus = 4`, `gputype = A100` | `pbsnodes` |
| Node-local scratch | **`/local/scratch`** — 2.9 TB device, 2.8 TB free, real disk | measured |
| `/tmp` | **tmpfs, 252 GB — RAM-backed.** Writing there consumes node memory | measured |
| Allocation mode | `sharing = force_exclhost` | `pbsnodes` |

**The SMT map is the thing people get wrong.** `os.cpu_count()` returns **64**, but
there are only **32 physical cores**. Launching 64 compute-bound workers gives you
roughly the throughput of 32 plus the overhead of 64 — see the measured scaling curve
in §12.

---

## 3. A minimal CPU-only job script

```bash
#!/bin/bash -l
#PBS -N cpu_job
#PBS -A <your_project>                    # allocation to charge
#PBS -q debug                             # see §5 for the queue decision
#PBS -l select=1:system=polaris           # N nodes; whole-node either way
#PBS -l place=scatter                     # one chunk per node
#PBS -l filesystems=home:eagle            # REQUIRED — see §4
#PBS -l walltime=00:30:00                 # min 00:05:00, max is per-queue
#PBS -j oe                                # merge stderr into stdout
#PBS -r n                                 # not rerunnable (see §4 for when to flip)

cd "${PBS_O_WORKDIR:-$PWD}"

# --- environment (pick one route from §7) -------------------------------
# Do env setup BEFORE `set -e`: module and conda shell functions return
# nonzero on harmless conditions, and `conda activate` trips `set -u`.
# Plain statements only — never pipe `module load` (§7.4, trap 1).
module use /soft/modulefiles
module load conda
conda activate base
source /path/to/<your_venv>/bin/activate   # if you built one

set -eo pipefail                           # now it is safe

# --- CPU-only hygiene ---------------------------------------------------
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1                  # ignore ~/.local — see §7.4
export OMP_NUM_THREADS=1                   # see §8; set deliberately, never leave unset
export HDF5_USE_FILE_LOCKING=FALSE         # Lustre; see §11

echo "jobid=${PBS_JOBID%%.*} host=$(hostname) nodes=$(wc -l < "$PBS_NODEFILE")"

python my_cpu_work.py --workers 32

echo "CPU_JOB_OK"                          # a PASS token — see §4, rule 5
```

Submit and watch:

```bash
qsub cpu_job.pbs                       # prints <jobid>.polaris-pbs-01...
qstat -u $USER                         # Q = queued, R = running, F = finished
qstat -x -f <jobid>                    # full record, including why it is not running
qdel <jobid>
```

With `-j oe` and **no** `-o`, PBS writes `<jobname>.o<jobid>` into the submit
directory — one file per job, per-user for free. (If you point `-o` at a *directory*
you instead get `<full_jobid>.OU`, and a fixed `-o` **file** path is **appended** to,
not truncated, so several runs pile into one file.)

---

## 4. Directives: the ones that bite

| Rule | Detail |
|---|---|
| **1. `-l filesystems=` is mandatory** | Omit it and `qsub` **rejects the job outright**. List every filesystem you touch: `home`, `eagle`, `grand`. A job that does not declare a filesystem will not be scheduled onto nodes where it is mounted. |
| **2. Minimum walltime is `00:05:00`** | `resources_min.walltime` is 5 minutes on every queue. A 2-minute request is rejected. |
| **3. Every job attribute is immutable after submit** | `qalter` is refused on this system for **all** attributes (`Exception in account_check hook encountered`, rc=32) — walltime, dependencies, everything. **Size walltime with margin at submit time; there is no second chance.** To stagger already-queued jobs use `qhold` / `qrls`. Dependencies can only be set at submit: `qsub -W depend=afterany:<jobid>`. |
| **4. `-r n` vs `-r y`** | `-r n` = not rerunnable. On a preemptable queue that means PBS **kills** rather than requeues you. If your work checkpoints and resumes correctly, `-r y` turns a preemption into a requeue. If it does not, keep `-r n` — a silent restart from scratch is worse. |
| **5. Never trust `rc=0`** | A killed, walltime-truncated, or no-op run can still exit 0. End your script with an explicit **PASS token** (`CPU_JOB_OK`) written only after the real work, and key success on that token, not the exit code. |
| **6. `place=scatter`** | One chunk per node. Relevant only for multi-node; harmless and correct for one. |
| **7. `$PBS_NODEFILE` has one line per NODE** | Not one per core. `NNODES=$(wc -l < $PBS_NODEFILE)`. |

Useful PBS variables inside the job: `$PBS_JOBID` (use `${PBS_JOBID%%.*}` for a numeric
tag), `$PBS_O_WORKDIR` (the submit directory — your job does **not** start there by
default under all shells, so `cd` to it), `$PBS_NODEFILE`.

Pass variables in at submit time with `qsub -v NAME=value,OTHER=value script.pbs`.

---

## 5. The queues

Queried from `qstat -Qf` on **2026-09-22**. "nodes" is `resources_min.nodect` –
`resources_max.nodect`; limits in brackets are PBS's own notation (`u:` per user,
`p:` per project, `o:PBS_ALL` = system-wide).

### Queues you can actually submit to

| queue | walltime | nodes | limits | priority | what it is for |
|---|---|---|---|---|---|
| **`debug`** | 5 min – **1 h** | 1–2 | **`max_run` 1/user**, **`queued_jobs_threshold` 1/user**, system-wide 24 nodes | 150 | Smoke tests, short CPU chunks. Starts fast (historically seconds). **Cannot be chained or pre-loaded** — one job in Q *or* R per user, and even a dependency-held job counts. |
| **`debug-scaling`** | 5 min – **1 h** | 1–**10** | `max_queued` 1/user, `queued_jobs_threshold` 1/user | 150 | Same as `debug` but up to 10 nodes. The right home for a short multi-node CPU test. |
| **`preemptable`** | ≤ **72 h** | 1–**10** | `max_queued` 20/user **and** 20/project, **`max_run` 10/project** | 155 | **Concurrency.** Ten jobs running at once per project. The catch is §6. |
| **`capacity`** | 5 min – **168 h (7 d)** | 1–**4** | **`max_run` 1/project**, **`max_queued` 2/project**, system-wide 32 nodes | 150 | Long uninterrupted single-node-ish runs. One slot for the *whole project* — coordinate before taking it. |
| **`prod`** (routing) | ≤ 24 h | **≥ 10** | `max_queued` 100/project | — | Routes to `small` (10–24 nodes, ≤3 h), `medium` (25–99, ≤6 h), `large` (100–496, ≤24 h), and their `backfill-*` twins (priority 1). **You cannot submit to the execution queues directly** (`from_route_only = True`) — submit to `prod` and let it route by node count. |

### Queues that exist but you probably cannot use

| queue | why not |
|---|---|
| `demand` | ACL-restricted to a named list of groups. ≤1 h, ≤56 nodes, priority 160 — **this is what preempts `preemptable`**. Access by request to ALCF support. |
| `build`, `ds_build`, `jenre-build` | ACL-restricted (named users / the `datascience` group). |
| `visualization` | ACL-restricted (`visualization`, `viz_polaris` groups). |
| `run_next` | ACL-restricted to two admin accounts. |
| `alcf_training*`, `ATPESC*`, `ALCFAITP`, `gpu_hack*` | Event/workshop routing queues, closed outside those events. |

### Picking one, for CPU work

* **Anything under an hour → `debug`** (1–2 nodes) or **`debug-scaling`** (up to 10).
  Deterministic and near-immediate. Chunk longer work into ≤55-minute pieces driven by
  a serial driver script rather than waiting on a big queue.
* **Long single job → `capacity`.** Up to 7 days, no preemption. But `max_run` is **1
  per project**, so you are competing with your own colleagues, not with the machine.
  Check who holds it first:
  ```bash
  qstat -a | awk 'NR>5 && $3 ~ /capacity/ {print $1, $2, $10}'
  ```
* **Many jobs at once → `preemptable`.** `max_run` 10/project versus `capacity`'s 1 is
  the axis on which it wins. Start latency is unpredictable (§6), and `demand` can
  preempt you, so use it for work that is either short-per-unit or checkpointed.
* **≥10 nodes → `prod`.** For CPU-only work this is rarely the right answer; ten whole
  GPU nodes to run CPU code is expensive, and `debug-scaling` covers 10 nodes for free
  for an hour.

> **A note on `max_queued` in PBS Pro:** the counter includes **running** jobs, not
> just queued ones. On `capacity` (`max_queued = 2/project`, `max_run = 1/project`)
> that means one running job plus one queued is the ceiling, and it is shared with
> everyone on your allocation. The practical consequence is circular and worth saying
> plainly: **you cannot pre-queue a successor on `capacity` while the parent runs**, so
> chain via `preemptable` (dependencies work fine across queues) and reserve
> `capacity` for the long leg.

---

## 6. Why a job is not starting — diagnose, never resubmit

`preemptable` runs **only on nodes `prod` is not using**, so its throughput tracks
machine load: on a busy day it starts nothing for half a day; on a quiet one dozens of
jobs run. State it as *unpredictable start latency*, never as "never starts".

**Before you ever resubmit a job that "isn't starting", spend 30 seconds:**

```bash
qstat -x -f <jobid> | grep -E "^ +comment|^ +eligible_time|^ +job_state"
qstat -Q                                    # Que vs Run counts per queue
```

⚠ Anchor that grep with `^ +eligible_time` — a bare `grep eligible_time` matches
`Resource_List.wfp_eligible_time_exp` first and hands you an exponent, not a wait time.

Then read `comment`:

| comment | meaning | what actually helps |
|---|---|---|
| `Insufficient amount of resource: queue_tags` | **No nodes are available to this queue at all.** | **Nothing you control.** Not walltime, not node count, not resubmitting. Switch queue or wait. |
| `Not Running: Insufficient amount of resource: <other>` | a specific resource you requested is unavailable | reduce that resource |
| `job_state = H` | held, or waiting on a dependency | check `depend` |

And read `eligible_time`: if it is **large and growing**, your job *is* eligible and
simply is not getting nodes — a **supply** problem, not a priority one.

> **Resubmitting resets `eligible_time` to zero and makes things strictly worse.**
> Requested walltime is not the lever people assume it is: the `wfp` priority terms
> only rank *runnable* jobs against each other. They cannot conjure a free node.

---

## 7. Building the Python environment

Do this **once, on a login node** (installs on compute nodes work, but they burn
allocation). Login nodes reach PyPI through the ALCF proxy; if `pip` hangs, export it:

```bash
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
```

(The same variables work from a compute node — outbound *does* work there, via the
proxy only.)

### 7.1 Route A — `cray-python` + venv  ← recommended for pure CPU work

The Cray PE Python is the leanest base for CPU-only jobs: no CUDA stack, numpy/scipy
linked against `cray-libsci`, and **`mpi4py` already built against `cray-mpich`** —
which matters, because a `pip install mpi4py` builds against whatever MPI it finds and
will not use the Slingshot fabric properly.

```bash
module load cray-python                     # 3.12.12 (verified 2026-09-22)
python3 -m venv --system-site-packages <path>/cpuenv
source <path>/cpuenv/bin/activate
pip install --upgrade pip
pip install joblib threadpoolctl psutil h5py netCDF4 xarray zarr numba numexpr tqdm
```

`--system-site-packages` is the important flag: it inherits the module's
numpy/scipy/**mpi4py**/pandas/dask instead of rebuilding them. Ships out of the box
(verified): `numpy 2.3.5`, `scipy 1.16.3`, `mpi4py 4.1.1`, `pandas 2.3.3`,
`dask 2025.11.0`, `pyyaml`. Absent, so pip them: `h5py`, `netCDF4`, `joblib`,
`psutil`, `numexpr`, `numba`, `xarray`, `zarr`, `threadpoolctl`.

In the job script, re-run **both** lines — `module load cray-python` then `source
.../activate`. A venv without its base module on `PATH` is not reproducible.

### 7.2 Route B — the ALCF base conda + venv

Use this if you also want PyTorch/CUDA-capable packages available (e.g. a CPU-only
preprocessing step inside a mostly-GPU project, sharing one env).

```bash
module use /soft/modulefiles
module load conda                 # → the current default; `module avail conda` to list
conda activate base               # python + numpy now on PATH
python -m venv --system-site-packages <path>/cpuenv
source <path>/cpuenv/bin/activate
pip install <your extras>
```

Available as of 2026-09-22: `conda/2025-09-25` (Python 3.12.11, numpy 2.2.6),
`conda/2026-09-17` (Python 3.13.15), plus `-aws-nccl-*` variants that matter only for
multi-node **GPU** collectives — irrelevant to CPU-only work, and they have their own
pitfalls. Pin a dated version in your script rather than bare `module load conda`, so a
site-side default bump cannot silently change your interpreter underneath you.

> ⚠️ **`module load conda` alone does not put `python` on `PATH`** — it adds `condabin`.
> You must follow it with `conda activate base`. A script that tests `which python`
> between the two lines will wrongly conclude the module is broken.
>
> ⚠️ **This module has broken before**, cluster-side: for about a week in August 2026
> every `conda/*` modulefile pinned Cray PE versions that no longer existed, and every
> job in flight died with `conda: command not found`. It is working again (verified on
> a login node *and* a compute node, 2026-09-22 — §12). The lesson is to **pin a
> version and have a fallback route**, because this dependency is not under your
> control.

### 7.3 Route C — your own Miniforge

Full control, no module dependency, immune to site-side PE churn. Costs you ~2 GB and
the maintenance.

```bash
curl -L -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/mf.sh -b -p <path>/miniforge3
source <path>/miniforge3/etc/profile.d/conda.sh
conda create -y -n cpuenv python=3.12 numpy scipy pandas joblib threadpoolctl h5py netCDF4 xarray dask
conda activate cpuenv
```

If you need MPI here, install `mpi4py` against the system MPI rather than pulling a
conda MPI (`MPICC=cc pip install --no-binary=mpi4py mpi4py` with `cray-mpich` loaded),
or stay on Route A.

### 7.4 Five environment traps

1. **Never run `module load` inside a pipe, a subshell, or a command substitution.**
   A pipeline runs its left-hand side in a subshell, so every `PATH` and shell-function
   change the modulefile made is **discarded the instant the pipe ends** — while the
   module's own output still scrolls past, looking like success:
   ```bash
   module load conda 2>&1 | tail -3     # ← WRONG: prints fine, changes nothing
   conda activate base                  # → conda: command not found
   ```
   This cost a probe job while writing this handoff, and the symptom is indistinguishable
   from the August module breakage. Load modules as **plain statements**; if you want to
   check the result, test afterwards with `command -v conda`. The same applies to
   `conda activate` and `source .../activate`.
2. **Never `pip install --user`** a dependency that other people are meant to share.
   ALCF home directories are mode `0700`, so `~/.local` packages are readable by
   exactly one account. A colleague running your identical script gets
   `ModuleNotFoundError` while it works perfectly for you. Put shared envs in **project
   space** (`/eagle/projects/<project>/...`) and `chmod -R a+rX` them.
3. **Set `PYTHONNOUSERSITE=1` in every job script.** It removes `~/.local` from
   `sys.path`, which is how you prove an env is reproducible for anyone else. Note it
   does **not** block `PYTHONPATH` — those entries still outrank site-packages and can
   shadow a version you care about.
4. **Do not put your env in `$HOME`.** Home quota is small and home is not the fast
   filesystem. Project space on `eagle` is the right home for envs *and* you must then
   remember to declare `filesystems=home:eagle`.
5. **Verify the env from a job, not from the login node.** Login nodes have a different
   module state and are not a reliable place to test imports. A one-line `debug` job
   that does `python -c "import ..."` and echoes a PASS token is worth the five
   minutes.

---

## 8. Threads: the numbers that control CPU parallelism

Python's GIL means a CPU-bound pure-Python workload scales with **processes**, not
threads. But the libraries underneath (BLAS, OpenMP, FFT) spawn their *own* threads,
and by default each one tries to use **all 64 logical CPUs**. Combine 32 worker
processes with a numpy that opens 64 BLAS threads each and you have asked the kernel
for 2048 runnable threads on 32 cores. This is the single most common cause of
"parallel" CPU jobs on Polaris running *slower* than serial.

> **The budget rule:** `n_processes × threads_per_process ≤ 32` (physical cores) for
> compute-bound work. Go to 64 only when the work is latency- or I/O-bound.

Set every one of these explicitly. **An unset variable does not mean one thread.**
Measured here with everything unset: `threadpool_info()` reports OpenBLAS at
**`num_threads = 64`**. Thirty-two workers each opening that pool is a request for 2048
threads on 32 cores.

| variable | controls | typical CPU-only setting |
|---|---|---|
| `OMP_NUM_THREADS` | OpenMP: numpy/scipy BLAS, scikit-learn, numexpr, many C extensions | `1` when you fan out with processes; `32` for one big threaded process |
| `OPENBLAS_NUM_THREADS` | OpenBLAS specifically (overrides `OMP_*`) | same as above |
| `MKL_NUM_THREADS` | Intel MKL (conda numpy often links MKL) | same as above |
| `NUMEXPR_NUM_THREADS` | numexpr / pandas `eval` | same |
| `VECLIB_MAXIMUM_THREADS` | Accelerate-style backends | same |
| `NUMBA_NUM_THREADS` | numba `parallel=True` | set deliberately |
| `OMP_PLACES` | thread↔core mapping | `cores` |
| `OMP_PROC_BIND` | thread pinning policy | `close` (pack) or `spread` (use all NUMA domains) |

PyTorch on CPU has no environment variable of its own — it reads `OMP_NUM_THREADS` at
import and is otherwise controlled through the API:
`torch.set_num_threads(n)` (intra-op) and `torch.set_num_interop_threads(n)`, the latter
of which must be called **before** any other torch work.

A safe preamble for a process-parallel job:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_PLACES=cores OMP_PROC_BIND=close
```

…and then parallelise with processes. For the opposite shape (one process, big arrays),
set them all to `32` and use a single worker.

**Count cores the right way inside Python.** `os.cpu_count()` reports 64 regardless of
what you were actually given; `len(os.sched_getaffinity(0))` reports what this process
may actually run on — which is what you want when a launcher has pinned you:

```python
import os
NCPU = len(os.sched_getaffinity(0))      # NOT os.cpu_count()
```

`threadpoolctl` is worth installing purely to see what your BLAS is really doing:

```python
from threadpoolctl import threadpool_info, threadpool_limits
print(threadpool_info())                  # which backend, how many threads
with threadpool_limits(limits=1, user_api="blas"):
    ...                                   # scoped, no env-var surgery
```

---

## 9. Libraries for multi-process / multi-node CPU work

| need | use | notes |
|---|---|---|
| Fan out over items, one node | **`concurrent.futures.ProcessPoolExecutor`** or `multiprocessing.Pool` | stdlib, no install. Default start method on Linux is `fork` — fast, but unsafe if threads or file handles are already open. Use `mp.get_context("spawn")` if you have touched a threaded library before forking. |
| Same, with nicer ergonomics | **`joblib`** (`Parallel(n_jobs=32)`) | loky backend; handles large array args via memmapping. Plays well with `threadpoolctl` — it limits BLAS threads inside workers for you. |
| Task graphs / out-of-core arrays | **`dask`** (`LocalCluster`) | `LocalCluster(n_workers=16, threads_per_worker=2, processes=True)` → 32 total. **Respect the budget rule**: `n_workers × threads_per_worker ≤ 32`, and set `OMP_NUM_THREADS=1` so each worker does not also open a BLAS pool. |
| Chunked array I/O | `xarray` + `dask`, `zarr`, `h5py`, `netCDF4` | see §11 for the Lustre/HDF5 caveats |
| **Multi-node** CPU parallelism | **`mpi4py`** | the only clean way to span nodes. Use the `cray-python` build (§7.1) or build against `cray-mpich`. Launch with `mpiexec`, never bare `python`. |
| Multi-node without MPI | `torch.distributed` with the **`gloo`** backend | works CPU-only; heavier dependency, but natural if the code is already PyTorch |
| Trivial task farming | `xargs -P 32 -n 1` over a list of commands | no Python at all; surprisingly effective for "run this binary on 5000 files" |
| JIT / vectorised inner loops | `numba` (`@njit(parallel=True)`), `numexpr`, `cython` | set `NUMBA_NUM_THREADS`; these are threads, so they count against the budget |
| Introspection | `psutil`, `threadpoolctl` | `psutil.Process().cpu_affinity()`, per-process CPU% while tuning |

### Single-node pattern

```python
import os
from concurrent.futures import ProcessPoolExecutor

NCPU = len(os.sched_getaffinity(0))
WORKERS = min(32, NCPU)          # physical cores, not 64

def work(item): ...

if __name__ == "__main__":
    items = [...]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for result in ex.map(work, items, chunksize=8):
            ...
```

Run it directly in the PBS script — on one node your job script already executes on the
allocated node, so no launcher is needed:

```bash
python my_cpu_work.py
```

### Multi-node pattern

Multiple nodes require `mpiexec` (Cray PALS). A plain `python` call only ever runs on
the first node in `$PBS_NODEFILE`.

```bash
NNODES=$(wc -l < "$PBS_NODEFILE")
RANKS_PER_NODE=32                       # one per physical core
NRANKS=$(( NNODES * RANKS_PER_NODE ))

mpiexec -n "$NRANKS" --ppn "$RANKS_PER_NODE" \
        --cpu-bind depth -d 1 \
        python my_mpi_work.py
```

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
mine = [x for i, x in enumerate(all_items) if i % size == rank]   # static split
results = comm.gather(process(mine), root=0)
```

---

## 10. CPU binding — and why you should not skip it

`mpiexec`'s `--cpu-bind` decides which cores each rank may use.

| form | meaning |
|---|---|
| `--cpu-bind depth -d N` | give each rank `N` consecutive CPUs. `ranks_per_node × N` should equal 32 (physical) or 64 (with SMT). **Measured:** `-n 4 --ppn 4 --cpu-bind depth -d 8` → rank 0 gets cpus `0-7`, rank 1 `8-15`, rank 2 `16-23`, rank 3 `24-31` — i.e. the 32 physical cores in order, one NUMA domain per rank. |
| `--cpu-bind list:0:8:16:24` | explicit per-rank CPU lists, colon-separated. Use when you want a specific rank→NUMA placement. |
| `--cpu-bind none` | ranks may float over all 64 CPUs (documented behaviour; not exercised in the probe) |
| *(omitting `--cpu-bind`)* | **NOT "unbound".** See the trap below. |

> ### ⚠ The default is one core per rank — measured, and it is the classic silent killer
> Omitting `--cpu-bind` does **not** let ranks float. PALS defaults to binding each rank
> to a **single CPU**: `mpiexec -n 4 --ppn 4 python3 -c "...sched_getaffinity..."`
> reports `ncpus 1` for every rank (measured, §12).
>
> So `mpiexec -n 4 --ppn 4` with `OMP_NUM_THREADS=8` puts **all eight OpenMP threads of
> each rank on one core** — a ~8× slowdown with no error, no warning, and a perfectly
> normal-looking log. **Whenever a rank is meant to use more than one thread, you must
> pass `--cpu-bind depth -d <threads_per_rank>` explicitly.**
>
> The reverse case bites too: unbound/misbound ranks that *do* migrate across NUMA
> domains degrade silently rather than failing. In this environment a multi-node run
> without proper depth binding measured **9×** less inter-node bandwidth than a bound
> one — `4.08` vs `36.93 GB/s` — with no diagnostic of any kind.

**Always verify the mask rather than assuming it.** One line, and it costs nothing:

```python
import os; print("rank", os.environ.get("PMI_RANK"), sorted(os.sched_getaffinity(0)))
```

Two Polaris-specific wrinkles:

* **NPS4 memory locality.** A rank bound to cores `0-7` allocates from NUMA node 0. If
  its data was produced by a rank on NUMA node 3, every access is a cross-domain hop.
  For memory-bandwidth-bound CPU work, prefer **4 ranks × 8 cores** aligned to the NUMA
  map (§2) over 32 scattered single-core ranks.
* **SMT, and why `-d 2` × 32 ranks is the wrong shape.** Binding walks logical CPUs
  `0,1,2,…` linearly, and CPUs `0-31` are the real cores while `32-63` are siblings. So
  `-d 2` with 32 ranks hands ranks 0–15 two *physical cores* each and ranks 16–31 only
  the *siblings* of those cores (measured: rank 0 → `[0,1]`, rank 1 → `[2,3]`, … rank 31
  → `[62,63]`). The ranks are unequal, and a synchronised code runs at the pace of the
  slow half. **Balanced shapes: 32 ranks × `-d 1`, 8 × `-d 4`, or 4 × `-d 8`.**

Recommended shapes for a full node, all of which land on CPUs `0-31`:

| ranks/node | `-d` | each rank gets | good for |
|---|---|---|---|
| 32 | 1 | 1 physical core | pure MPI, single-threaded ranks |
| 8 | 4 | 4 cores, half a NUMA domain | MPI + modest OpenMP |
| **4** | **8** | **8 cores = exactly one NUMA domain** | MPI + OpenMP, memory-bandwidth-bound work |
| 1 | 32 | the whole socket | one fat threaded process |

`numactl` is available on the compute nodes for finer control
(`numactl --cpunodebind=0 --membind=0 ...`) in non-MPI launches.

---

## 11. I/O notes for CPU-heavy preprocessing

* **Declare your filesystems** (`-l filesystems=home:eagle`) or the job is rejected.
* **`HDF5_USE_FILE_LOCKING=FALSE`** — HDF5 file locking on Lustre fails with
  `BlockingIOError: unable to lock file`. Set it in every script that touches `.h5`
  or NetCDF4 files.
* **Lustre hates many small files.** A Python env is tens of thousands of small files;
  if you launch hundreds of ranks that each import the same env, the metadata server
  becomes the bottleneck. Mitigations: keep rank counts modest, or stage the env/inputs
  to **`/local/scratch`** (~2.8 TB per node, node-local, wiped at job end) at job start
  and read from there.
* **Write intermediates to `/local/scratch`, final outputs to project space.** Node-local
  scratch does not survive the job — copy anything you need out before the script ends.
* **Do not use `/tmp` as scratch.** It is a **252 GB tmpfs**, i.e. RAM. A job that
  spills 100 GB of intermediates there has silently spent 100 GB of its 503 GiB and
  will OOM somewhere unrelated. `/local/scratch` is the real disk.
* **Chunk long conversions.** Measure per-unit cost first, then size chunks to ~80% of
  the queue's walltime cap, and have the driver key each chunk's success on its **PASS
  token** and stop on the first failure — otherwise you get a silent hole in the output
  dataset that surfaces weeks later.

---

## 12. Validation record

Three 1-node `debug` jobs on **2026-09-22**, each on a different node — `7643786`
(`x3001c0s13b0n0`), `7643793` (`x3001c0s19b1n0`) and `7643804` (`x3005c0s7b1n0`) — plus
`qstat -Qf` and `pbsnodes` the same day. Everything marked "measured" above comes from
these. All three started within seconds of submission, which is also the evidence for
"`debug` starts fast".

`7643804` is the **§15 appendix script run verbatim**, so the script this handoff ships
is known to work as printed, and it independently reproduced the hardware facts, the
binding masks and the BLAS scaling on a third node.

**Hardware, confirmed on-node**

```
cpu_model           = AMD EPYC 7543P 32-Core Processor
sockets / cores per socket / threads per core = 1 / 32 / 2      → 64 logical CPUs
mem_total_gib       = 503.6
numa_nodes          = 4
numa_map            = node0 0-7,32-39   node1 8-15,40-47
                      node2 16-23,48-55 node3 24-31,56-63
numactl             = /usr/bin/numactl        taskset = /usr/bin/taskset
local_scratch       = /dev/md127  2.9T total, 2.8T free
/tmp                = tmpfs 252G           ← RAM-backed, counts against node memory
bare script affinity (no mpiexec) = 64
```

**Environment routes, both green on a compute node**

```
module load conda && conda activate base  → /soft/.../mconda3/bin/python  Python 3.12.11
                                            numpy 2.2.6, BLAS = OpenBLAS
module load cray-python                   → Python 3.12.12
                                            mpi4py 4.1.1, numpy 2.3.5
```

So the August 2026 `module load conda` breakage is **resolved** — verified on both a
login node and a compute node.

**The `module load`-in-a-pipe trap, isolated across the two jobs** (same node type, same
modulefile; the only difference is the pipe):

```
job 7643786:  module load conda 2>&1 | tail -3   → conda: command not found
job 7643793:  module load conda                  → conda found; activate base → Python 3.12.11
```

**BLAS threads when you set nothing** — `threadpool_info()` with `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` all unset:

```
default: blas openblas num_threads = 64
```

i.e. **64, not 1.** That is the oversubscription trap in §8 in one line: 32 worker
processes that each open this default pool ask for 2048 threads on 32 cores.

**Threaded BLAS scaling** — `numpy` 4000×4000 matmul, one process, warm:

| `OMP_NUM_THREADS` | wall | speedup |
|---|---|---|
| 1 | 2.33 s | 1.0× |
| 8 | 0.32 s | 7.3× |
| 32 | 0.14 s | 16.6× |

**Process scaling** — 64 identical pure-Python CPU-bound tasks through
`multiprocessing.Pool`, run twice on two different nodes:

| workers | run A wall | run A speedup | run B wall | run B speedup | efficiency (B) |
|---|---|---|---|---|---|
| 1 | 13.02 s | — | 12.86 s | — | — |
| 8 | 1.98 s | 6.57× | 1.91 s | 6.72× | 84 % |
| 16 | 1.18 s | 11.05× | 0.93 s | 13.76× | 86 % |
| 32 | 0.83 s | 15.74× | 0.71 s | 18.12× | 57 % |
| **64** | **0.69 s** | **18.89×** | **0.56 s** | **23.03×** | **36 %** |

Two honest caveats. **Run-to-run spread is large** at high worker counts (15.7× vs
18.1× at 32 workers on identical code), so treat any single number as indicative, not
as a benchmark. And 64 coarse tasks over 32 workers is only 2 tasks each, so `Pool`
startup is a visible share of a 0.7 s measurement — the curve understates true scaling
in the middle rows.

What *is* stable across both runs, and is the part to act on: **near-linear to 8
workers, clearly sublinear by 32, and only ~20–27 % more throughput from the jump to
64.** That last step is SMT giving a little, not 32 extra cores.

**`mpiexec` CPU binding, measured masks**

| launch | result |
|---|---|
| `-n 4 --ppn 4 --cpu-bind depth -d 8` | rank 0 `0-7`, rank 1 `8-15`, rank 2 `16-23`, rank 3 `24-31` — **one NUMA domain per rank** |
| `-n 32 --ppn 32 --cpu-bind depth -d 1` | 32 ranks, 1 CPU each |
| `-n 32 --ppn 32 --cpu-bind depth -d 2` | rank 0 `[0,1]`, rank 1 `[2,3]`, rank 2 `[4,5]` … rank 31 `[62,63]` |
| `-n 4 --ppn 4` *(no `--cpu-bind`)* | **every rank `ncpus 1`** |
| `-n 64 --ppn 64` *(no `--cpu-bind`)* | rank *i* → CPU *i*, through rank 63 → CPU 63 |

Two things fall out of these masks:

* **Logical CPU numbering is "all physical cores first".** CPUs `0–31` are the 32
  distinct physical cores; `32–63` are their SMT siblings (sibling of *i* is *i+32*).
  That is why `-d 8` with 4 ranks lands cleanly on real cores and on NUMA boundaries.
* **`-d 2` with 32 ranks is a trap.** It spreads linearly over CPUs `0–63`, so ranks
  0–15 get two real cores each while ranks 16–31 get only SMT siblings of those same
  cores. The ranks are then *not* equal, and a synchronised code runs at the speed of
  the slow half. If you want 32 balanced ranks, use **`-d 1`** (one physical core each);
  if you want fat ranks, use **4 × `-d 8`** or **8 × `-d 4`**.

---

## 13. Pre-flight checklist

```
[ ] -A <project>, and the allocation has hours left
[ ] -l filesystems= lists every filesystem touched   (else: rejected at submit)
[ ] walltime >= 00:05:00 and sized with margin        (else: no qalter, no second chance)
[ ] queue matches walltime AND node count             (§5)
[ ] cd "$PBS_O_WORKDIR" at the top
[ ] env: module load ... && source .../activate       (both lines, in the job)
[ ] export CUDA_VISIBLE_DEVICES=""                    (CPU-only means CPU-only)
[ ] export PYTHONNOUSERSITE=1
[ ] every *_NUM_THREADS set explicitly                (§8 budget rule)
[ ] workers counted from sched_getaffinity, not cpu_count
[ ] multi-node work goes through mpiexec with --cpu-bind  (§10)
[ ] HDF5_USE_FILE_LOCKING=FALSE if touching HDF5/NetCDF
[ ] script ends with a PASS token; success keyed on the token, not rc
```

## 14. Failure quick-reference

| symptom | cause | fix |
|---|---|---|
| `qsub` rejects immediately | missing `-l filesystems=`, or walltime < 5 min, or queue node-count range violated | §4 |
| `would exceed queue ...'s per-user/per-project limit` | `debug` allows 1 job (Q or R) per user; `capacity` 1 running + 2 total per project | §5 |
| Job sits in Q with `comment: ... queue_tags` | no nodes available to that queue | wait or switch queue — **do not resubmit** (§6) |
| `conda: command not found` in the job | `module load` run inside a pipe/subshell (§7.4 trap 1), or `module load conda` without `conda activate base` (§7.2), or a site-side module break | check §7.4 trap 1 first — likeliest and least obvious |
| Threaded ranks ~8× slower than expected | `mpiexec` without `--cpu-bind`: each rank is pinned to **1 core**, so all its threads share it | §10 |
| `ModuleNotFoundError` for a colleague only | `pip install --user` into a `0700` home | §7.4 trap 2 |
| Job dies immediately at the `module`/`conda` lines | `set -e` / `set -u` active during env setup | §3 — set them after |
| OOM with no large allocation in sight | scratch written to `/tmp`, which is RAM | §11 |
| Parallel version slower than serial | thread oversubscription: workers × BLAS threads ≫ 32 | §8 |
| Only one node does any work | no `mpiexec`; the script runs on node 0 only | §9 |
| Multi-node run inexplicably slow, no error | ranks unbound | `--cpu-bind depth -d N` (§10) |
| `BlockingIOError: unable to lock file` | HDF5 locking on Lustre | `HDF5_USE_FILE_LOCKING=FALSE` |
| `rc=0` but no output produced | trusted the exit code | PASS token (§4.5) |
| `qalter: Exception in account_check hook` | `qalter` is disabled here | resubmit; use `qhold`/`qrls` to stagger (§4.3) |

---

## 15. Appendix — the probe, to re-verify any of this

Every number in §12 came from this. Cluster software rolls; when something here stops
matching reality, re-run it rather than guessing. One `debug` node, about two minutes.

```bash
#!/bin/bash -l
#PBS -N cpu_probe
#PBS -A <your_project>
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=00:25:00
#PBS -j oe
#PBS -r n

cd "${PBS_O_WORKDIR:-$PWD}"
echo "PBS_JOBID=$PBS_JOBID host=$(hostname)"
echo "nodefile lines=$(wc -l < "$PBS_NODEFILE")"

# --- env routes ---------------------------------------------------------
module use /soft/modulefiles
module load conda                    # plain statement, never piped
echo "after module load: conda=$(command -v conda || echo MISSING)"
conda activate base
echo "after activate:    python=$(command -v python || echo MISSING) $(python -V 2>&1)"

# --- hardware + scaling -------------------------------------------------
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
python - <<'PY'
import os, time, subprocess, multiprocessing as mp
sh = lambda c: subprocess.check_output(c, shell=True).decode().strip()
print("cpu_count=%d  affinity=%d  mem_gib=%.1f" % (
    os.cpu_count(), len(os.sched_getaffinity(0)),
    int(sh("awk '/MemTotal/{print $2}' /proc/meminfo"))/1048576))
print(sh("lscpu | grep -E '^Model name|^Socket|^Core|^Thread|^NUMA'"))
print(sh("df -h /local/scratch /tmp | tail -2"))
def busy(n):
    x = 0
    for _ in range(n): x = (x*1103515245 + 12345) % 2147483648
    return x
base = None
for nw in (1, 8, 16, 32, 64):
    t = time.time()
    with mp.Pool(nw) as p: p.map(busy, [2_000_000]*64)
    dt = time.time()-t; base = base or dt
    print("workers=%-3d wall=%6.2fs speedup=%.2fx" % (nw, dt, base/dt))
print("CPU_PROBE_OK")     # printed only if everything above completed
PY

# --- BLAS threading -----------------------------------------------------
for T in 1 8 32; do
  OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T python -c "
import numpy as np, time, os
a=np.random.rand(4000,4000); a@a
t=time.time(); a@a
print('OMP=%-3s matmul4000 %.2fs'%(os.environ['OMP_NUM_THREADS'], time.time()-t))"
done

# --- binding masks ------------------------------------------------------
for ARGS in "-n 4 --ppn 4 --cpu-bind depth -d 8" "-n 4 --ppn 4" "-n 32 --ppn 32 --cpu-bind depth -d 1"; do
  echo "--- mpiexec $ARGS"
  mpiexec $ARGS python3 -c "
import os
r=int(os.environ.get('PMI_RANK', os.environ.get('PALS_RANKID','0')))
if r < 4: print(' rank',r,'ncpus',len(os.sched_getaffinity(0)),sorted(os.sched_getaffinity(0))[:8])" 2>&1 | sort
done

echo "PROBE_DONE"
```

Key on the **`CPU_PROBE_OK`** token — it is printed from inside the Python block and so
means the probe body actually ran. `PROBE_DONE` and `rc=0` only mean the shell reached
the last line.
