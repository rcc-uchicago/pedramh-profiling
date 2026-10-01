Write and submit a set of small Slurm jobs, **one per arm**, that reproduce my ACE2 epoch-2
slowdown in short epochs on one GH200 node each, while recording GPU, memory and per-process
telemetry. Use the step-2 report (pasted below).

**What we know so far:**

- The rollout length is constant, and the allocator isn't thrashing (`retries=0`).
- **Every job** runs fast until its own first epoch boundary, then drops about 2×, and stays
  slow: a fresh job at epoch 1 (6.53 → 3.06) and a resumed job at epoch 19 (6.80 → 3.27).
- The job's first validation is fast too. The only heavy step between it and the slow epoch is
  inline inference (~18 min). After it come a 22 s wandb log of the inference plots, an 8 s
  checkpoint save, a 3 s reclaim and a 5 s loader re-fork.
- The logs show no torch.compile recompile-limit messages. Eager runs at ~6.1/rank anyway, so a
  fallback to eager can't explain 3.0.

What is still open: does the slow state live in the **Python process** or in the **node/job**
(memory cgroup, page cache, a long-lived wandb process)? Each job so far was also a new
allocation, so we can't tell yet. That is why every arm gets **its own job** (a clean node), and
why arm A **relaunches** once inside its allocation.

The suspects, each with an arm:

- inline inference itself;
- train and validation workers re-forked from the trainer process after inference (netCDF +
  `FME_PERSISTENT_WORKERS` unset → re-forked every epoch, `fme/ace/data_loading/getters.py:101-118`);
- **wandb**: only rank 0 logs (`fme/core/wandb.py:119`), and DDP makes every rank wait for
  rank 0. The first big media log comes right after inference, and every job starts a fresh
  wandb run. Our Polaris tests, which did *not* reproduce, ran with wandb off.

**Rules:**

- Don't modify `fme/`, my production config, my production script, or any running job.
- Create new files only, under `steptest/`.
- Each job script copies my production `#SBATCH` header and environment block **verbatim**,
  changing only `--time` (below). After that block, add
  `export WANDB_RUN_NAME=steptest_<arm>_${SLURM_JOB_ID}`, and set `logging.project=ace2_steptest`
  in the overrides, so test runs stay out of my production wandb project.
- Judge results by the `STEPTEST_RESULT` / `RATIO` / `TELEM` lines, not by exit codes.

**Helpers:** `mem_sampler.sh`, `parse_steptest.py` and `telemetry_summary.py` from `helpers/` go
in `steptest/`.

**Telemetry in every job** (start before training, kill after it):

```bash
OUT=<new dir>/steptest_<arm>_${SLURM_JOB_ID}; mkdir -p "$OUT"
bash steptest/mem_sampler.sh "$OUT/mem.tsv" 15 & S1=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,clocks.sm,clocks.max.sm,power.draw,enforced.power.limit,temperature.gpu,clocks_throttle_reasons.active,memory.used --format=csv -l 15 > "$OUT/gpu.csv" & S2=$!
( while :; do date '+%F %T'; numastat -m | egrep 'Node|MemFree|FilePages'; sleep 300; done ) > "$OUT/numastat.txt" 2>&1 & S3=$!
# per-process: instantaneous %CPU (incl. wandb-core), threads, CPU affinity, wandb dir size
( while :; do echo "## $(date '+%F %T') load=$(cut -d' ' -f1 /proc/loadavg)"
    top -b -n 2 -d 2 -u "$USER" -w 200 | awk '/^top -/{n++} n==2' | head -45
    for p in $(pgrep -u "$USER" -f 'fme'); do
      echo "aff $p ppid=$(ps -o ppid= -p $p) $(grep -h Cpus_allowed_list /proc/$p/status 2>/dev/null)"
    done | head -60
    echo "tmp $(df -h /tmp | tail -1)   wandb_dirs $(du -sh /tmp/wandb "$OUT"/*/wandb 2>/dev/null | tr '\n' ' ')"
    sleep 60; done ) > "$OUT/procs.txt" 2>&1 & S4=$!
```

**Overrides for every arm** (everything else stays as in production):

```
experiment_dir=$OUT/run   max_epochs=2   segment_epochs=2
train_loader.sample_with_replacement=4096
log_train_every_n_batches=<max(1, (4096 / train_batch) // 16)>
logging.metrics_log_dir=$OUT/run/metrics
logging.project=ace2_steptest
inference.epochs=<the slice that makes get_inference_epochs() return [1]>   # when inference is on
```

Keep inline inference at its **production** length (7300 steps, 16 ICs); don't shorten it. The
override above just runs it once, after epoch 1. A 240-step inference didn't trigger the
slowdown on Polaris. Keep my production validation and checkpoint settings too.

**Arms: one job each, all submitted together** (they're independent):

| arm | `--time` | differs from the overrides above by | if A reproduces but this arm does NOT |
|---|---|---|---|
| A_prod | 01:45:00 | `max_epochs=3` (keep `segment_epochs=2`). Run the training command, then **run the identical command again** in the same job: fme resumes and trains epoch 3 in a **fresh process on the same node** | — (see the RATIO e3 reading below) |
| B_noinf | 00:45:00 | `inference=null` | inline inference is the trigger |
| C_persist | 01:15:00 | `export FME_PERSISTENT_WORKERS=1` (workers forked once, during epoch 1, then kept) | workers re-forked from the trainer process after inference |
| E_nowandb | 01:15:00 | `logging.log_to_wandb=false` | wandb |
| D_compile *(optional; submit only if the allocation budget allows)* | 01:15:00 | `export FME_COMPILE=0` | torch.compile state |

After each job:

```
python steptest/parse_steptest.py "$OUT/run"
python steptest/telemetry_summary.py "$OUT/run/out.log" "$OUT/gpu.csv" "$OUT/mem.tsv"
head -1 "$OUT/mem.tsv"     # the memory cgroup and host
```

**Decision rule (fixed before running):** R = median samples/s/rank in epoch 2 ÷ median over the
second half of epoch 1 (`STEPTEST_RESULT`). For D, compare against D's own epoch 1, since eager
is ~7% slower.

- R ≤ 0.77 → REPRODUCED
- R ≥ 0.91 → NO_STEP
- anything in between → INCONCLUSIVE

How to read the outcomes:

- If A shows NO_STEP, the short epochs can't trigger the slowdown. Report, then go to step 4.
- If A reproduces it, the right-hand column for the arm that does **not** reproduce names the cause.
- **A's `RATIO e3/late_e1`** (the relaunched process):
  - ≥ 0.91: the slow state lives in the process. Restarting the process every epoch
    (`segment_epochs=1` in a relaunch loop) would keep every epoch fast.
  - ≤ 0.77: it outlives the process. It's node/job state (memory cgroup, page cache, a
    lingering process), so look at `mem.tsv` and `procs.txt`.
- Compare `TELEM train e1` with `TELEM train e2`:
  - GPU utilization halves while clocks stay the same: the GPU is waiting on data or on rank 0.
  - Clocks or power drop: capping.
  - FilePages rising on nodes 4/12/20/28: the file cache is filling GPU memory.
- `mem.tsv`: `cg_limit_gb` (my script has no `--mem` line), and whether `cg_events_high/max` or
  `cg_refault_file` start climbing after inference.
- In `procs.txt`, compare a sample from mid epoch 1 with one from mid epoch 2:
  - `wandb-core`/`wandb-service` %CPU;
  - the number of loader workers, and whether old ones are still alive;
  - worker %CPU (near 100% means CPU-bound, near 0% means waiting);
  - `Cpus_allowed_list` of the new workers compared with the epoch-1 workers;
  - load average compared with 288;
  - the size of `/tmp` and the wandb dir.

REPORT: per arm, the `STEPTEST_RESULT`, `RATIO`, `EPOCH`, `VAL_RATIO` and `TELEM train/val e1, e2`
lines; `cg_limit_gb`; any change in FilePages on nodes 4/12/20/28; for A, the `procs.txt`
comparison above in at most 8 lines. Then one line `VERDICT:` and one line `FILES:` listing each
`$OUT`.
