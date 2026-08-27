# ai-rossby multi-node DDP scaling — plan and prereg

Sibling of `makani_multinode_ddp_plan.md`, for the second harness in
`polaris_multinode_ddp_port_handoff.md`. Same method: **prereg → measure →
score, misses included.** Written and committed **before the first ladder job**.

Read that handoff and CHANGELOG 2026-08-23 → 2026-08-27 first; this file does not
repeat what they establish.

---

## 1. What is being measured, and on what

**Harness.** `physicsnemo_ai_rossby/examples/weather/ai_rossby/train.py`,
`model=sfno_e3sm_parity` — SfnoPlasim, **1.18 B parameters**, 101 channels on the
E3SM 1° grid (180×360), bf16, DDP only. No spatial or ensemble parallelism exists
in this harness and none is being added (handoff §4: the batch ceiling here is a
science limit, not an engineering gap).

**Env.** `$AI_ROSSBY_VENV` — **torch 2.10.0+cu129**, its own bundled NCCL, zarr 3,
`physicsnemo` installed **editable** so it imports from this checkout. Because the
torch is not 2.8.0, **every fabric pin makani measured is unverified here** and the
three probes below are a gate, not a formality (handoff §1).

**Store — the criterion and the answer.** Two zarr lineages exist (*allyears*,
awikner v3, transferred; *peryear*, our fork's SeqZarr repack), with a known
103→96 channel drift between generations. The handoff's criterion is: use the
store the existing production run trained on. Read off that run's rendered config
(`$MEMBER_ROOT/runs/ai_rossby_sfno_e3sm/sfno_e3sm_parity01/.hydra/config.yaml`):

```
dataset.zarr_path     = ${oc.env:AI_ROSSBY_DATA}/e3sm/train
dataset.val_zarr_path = ${oc.env:AI_ROSSBY_DATA}/e3sm/val
dataset.mean_path     = ${oc.env:AI_ROSSBY_DATA}/e3sm/norm/normalization_2015-2050.zarr
```

⇒ **`$AI_ROSSBY_DATA/e3sm` — the per-year lineage**, 30 train stores
(2015–2044) + 6 val (2045–2050). Cross-checked against that run's own log:
`steps_per_epoch=10950, world_size=4` ⇒ **43,800 train samples**. Every ladder arm
uses this store and only this store (handoff mistake #6: same arm, same code read
196.5 vs 114.9 ms on two different packs).

**Wrap headroom.** The largest arm needs `STEPS × GLOBAL_BATCH` = 60 × 32 = 1,920
distinct samples against 43,800 available — 22× clear. The guard still runs,
because a guard that is only correct today is not a guard.

## 2. Measurement design

**Weak scaling.** Local batch fixed at 1 sample/GPU; global batch grows 4 → 8 →
16 → 32. Per-GPU arithmetic is therefore constant and every change in step time is
communication plus load imbalance — the actual question. Strong scaling would move
the per-GPU work and confound the two.

| arm | nodes | ranks | global batch | what it isolates |
|---|---|---|---|---|
| A | 1 | 4 | 4 | baseline; NCCL never leaves NVLink |
| B | 2 | 8 | 8 | the first Slingshot hop |
| C | 4 | 16 | 16 | does the hop cost compound or saturate |
| 8n | 8 | 32 | 32 | the shape of the climb; where a prod rung would start |

**3 interleaved reps per arm** (`-v REP=`), ordered A,B,C,8n,A,B,C,8n,A,B,C,8n —
**never** AAABBBCCC. Handoff rule #16 / CHANGELOG §4.4c: two runs of an identical
config once measured 42.2% vs 37.4% for the same quantity. This is a validity
requirement, not tidiness.

**One config for the whole table** (handoff rule #4): self-built aws-ofi-nccl
**v1.21.1 + `OFI_NCCL_PROGRESS_MODEL=AUTO`** + cray libfabric 2.3.1, default NCCL
protocol (**no** Simple pin — that was an old-plugin workaround, costs ~26%
single-node on makani, and is unmeasured on this stack), `GPU_ORDER=forward`,
wandb **off**, 60 steps, one partial epoch. A row from any other configuration —
another plugin, `-v WANDB=1`, more epochs — goes to **its own CSV**, never this
one. Config lives in the log header and the launcher pins; the CSV schema is a
cross-run contract and does not grow columns for it (CLAUDE.md #10).

**Science note.** Weak scaling grows the global batch 4 → 32, and for *training*
that is a science change, not an engineering one. These arms train 60 steps for
timing and are thrown away; the derived LR is a valid miniature, not a proposal.
**Production at a new batch/LR needs jesswan's sign-off** (CLAUDE.md).

**Why the absolute numbers will not look like makani's.** makani's gradient
all-reduce is ~0.6 GB/step on a ~150 M-param model; ai-rossby's is **~4.7 GB/step**
on 1.18 B params at roughly 4× the per-step compute — ~8× the comms in absolute
terms, ~4× per unit compute. makani's scaling *efficiency* percentages must never
be quoted for this harness; §3 turns that into a falsifiable prediction instead.

## 3. Prereg — predictions recorded before the first ladder job

Written before any ai-rossby multi-node measurement exists. Scored honestly
afterwards in CHANGELOG, **including the misses**.

1. **Multi-node is not free.** Arm B's `step_med_ms` exceeds arm A's by **≥ 10%**.
   *Falsified if* B is within 5% of A.

2. **The fabric answer carries across the NCCL version.**
   `polaris_ai_rossby_ofi_plugin_matrix.pbs` returns **`C_progress_auto` and only
   `C_progress_auto`**, and `transport` reads `AWS Libfabric` on every B/C/8n row.
   *Falsified by* `UNKNOWN`, a TCP/socket fallback, or a different winning combo —
   any of which would mean the pin is NCCL-version-dependent and every makani
   fabric conclusion needs a per-venv caveat. This is the prediction most likely
   to fail for a boring reason, and the cheapest to check.

3. **The absolute penalty transfers; the ratio does not.** On the v1.21.1+AUTO
   stack makani's A→B step time jumped 115.3 → 490.7 ms, i.e. **+375 ms**, and
   then stayed roughly flat (460.5 at 4n, 545.0 at 8n) — a large, roughly constant
   cost paid whenever the fabric is touched, which CHANGELOG attributes to
   progress-model overhead rather than a bandwidth wall. If that reading is right,
   ai-rossby pays a **similar absolute** penalty on a **~4× longer step**:
   **arm B − arm A lands in 190–750 ms** (makani's +375 ms within a factor of 2),
   and ai-rossby's 2-node weak-scaling efficiency therefore lands **well above
   makani's 25%** — roughly 50–70% — for no better reason than that its step is
   longer. *Falsified if* the increase is < 190 ms or > 750 ms. A miss on the high
   side is the interesting one: it would mean the penalty scales with the 8×
   gradient volume, i.e. bandwidth after all, and predictions 4 and 6 follow it.

4. **The cliff saturates rather than compounds.** `step_med_ms` grows **≤ 25%**
   from arm C to arm 8n (makani: +18% on this stack). *Falsified by* > 25%, which
   would say the 4.7 GB all-reduce, not fixed progress overhead, sets the cost —
   and that is what would cap the climb to prod rungs.

5. **DDP overlap hides much of the 4.7 GB.** `gpu_busy_frac` stays **≥ 0.85** on
   every arm. *Falsified if* it drops below, which would place the loss *outside*
   the timed step window (waiting between steps) rather than inside it, and would
   redirect the whole investigation from NCCL tuning to the loader.

6. **Rep spread is small compared with the effect.** Per-arm `step_med_ms` spread
   across the 3 interleaved reps is **< 5%**. *Falsified if* ≥ 5% — in which case
   no arm-to-arm difference smaller than the observed spread may be claimed, and
   the ladder needs more reps before it supports anything.

Prediction 3 is the one that matters: it is the only one whose answer changes what
a prod rung would cost. Predictions 1 and 2 are gates.

## 4. How to run

```bash
# once, login node — no allocation, no GPU, no torch
python physicsnemo_ai_rossby/polaris/test_parse_ai_rossby_scaling.py   # AI_ROSSBY_SCALING_PARSE_OK

cd physicsnemo_ai_rossby        # qsub resolves the script relative to cwd

# GATE 1 — fabric, under THIS venv. Do not skip: torch 2.10 != makani's 2.8.0.
qsub                  polaris/polaris_ai_rossby_fabric_probe.pbs        # FABRIC_PROBE_OK
qsub                  polaris/polaris_ai_rossby_ofi_plugin_matrix.pbs   # OFI_MATRIX_OK
qsub -q debug-scaling polaris/polaris_ai_rossby_nccl_mn_probe.pbs       # MN_NCCL_PROBE_OK

# GATE 2 — the launcher itself, cheapest first
qsub -l select=1:system=polaris polaris/polaris_ai_rossby_multinode_scaling.pbs
qsub -l select=2:system=polaris polaris/polaris_ai_rossby_multinode_scaling.pbs

# THE LADDER — interleaved, 3 reps. One PBS dependency chain per queue
# (`qsub -W depend=afterany:<jobid> …`), submitted in one shot: login-node rules
# forbid qstat polling loops, and `debug-scaling` allows ONE queued-or-running
# job per user, so each link becomes eligible only as its predecessor ends.
qsub                  -l select=1:system=polaris                     -v REP=1 polaris/polaris_ai_rossby_multinode_scaling.pbs
qsub                  -l select=2:system=polaris                     -v REP=1 polaris/polaris_ai_rossby_multinode_scaling.pbs
qsub -q debug-scaling -l select=5:system=polaris -v TARGET_NODES=4   -v REP=1 polaris/polaris_ai_rossby_multinode_scaling.pbs
qsub -q debug-scaling -l select=9:system=polaris -v TARGET_NODES=8   -v REP=1 polaris/polaris_ai_rossby_multinode_scaling.pbs
#  ... then REP=2, then REP=3, in the same A,B,C,8n order.
```

`NNODES` is derived from `$PBS_NODEFILE`, so one script serves every node count and
a command-line `-l select` overrides the directive. `debug` allows 1–2 nodes; 3–10
needs `debug-scaling`. `TARGET_NODES=N` with `select=N+1` prunes zombie-GPU nodes
and is **mandatory at ≥8** (three distinct nodes killed four makani runs on
2026-08-26; `x3111c0s37b1n0` three times).

PASS is **`AI_ROSSBY_MN_SCALING_OK`** plus a new row in
`$MEMBER_ROOT/bench/ai_rossby_multinode_scaling.csv` — **not `rc=0`**
(CLAUDE.md #14).

## 5. What would invalidate a row, and is checked automatically

* **N independent `world_size=1` trainers.** The launcher's worst failure: without
  the PALS rank shim, `DistributedManager` warns *"Assuming this is a single
  process job"* and every rank trains alone at a plausible step time.
  `PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD=ENV` makes it a hard error, and
  the parser rejects any row whose logged `world_size` ≠ the launched rank count —
  reading that only off `train.py`'s own banner, never off anything the launcher
  echoed.
* **Ranks that died before training.** `world_size` can be right while four ranks
  are missing. Counted separately from the PALS rank labels (`ranks_reporting`).
* **A short arm.** A walltime kill or an early `max_iterations` break gives a step
  average over a different number of steps. `n_steps` must equal the requested
  `STEPS`.
* **An epoch that wraps.** Re-serving page-cached samples makes a rank look fast,
  and the wrap point moves *with the global batch*, i.e. along the scaling axis.
  Gated on the loader's own per-rank epoch length, after sharding.
* **An unnamed transport.** A row that cannot say which network carried it is not
  evidence about an interconnect. Warned on, recorded as `UNKNOWN`.
* **Schema drift in the CSV.** Columns are a cross-run contract (CLAUDE.md #10);
  appending under a changed header is refused rather than silently done.

## 6. Where this document disagrees with the handoff

The handoff says: *if you find the doc contradicting the repo, trust the repo and
log it.* Three findings.

1. **The store verifier it names is for the other lineage.** §3a says to verify
   with `physicsnemo_sfno/polaris/polaris_verify_store.pbs` (PASS
   `SEQZARR_VERIFIED`). That verifies **SeqZarr** stores; the store the production
   run actually trained on is the **per-year zarr** written by
   `tools/data/e3sm/pangu_h5_to_zarr.py`. The verifier that matches this lineage
   is `ai_rossby_variable_contract.py --check-artifacts`, which both launchers
   already run as PREFLIGHT 2 against every train store, and which checks the
   thing that actually fails silently here: channel ORDER, where
   `ClimateZarrDataset` stacks in the store's `attrs` order while the NaN fill and
   the loss are built from the model config's lists.

2. **The launcher it calls "the origin of the stale fabric block" was also dead
   for a second, unrelated reason.** `polaris_sfno_e3sm_multinode.pbs` opened with
   a bare `module load conda`, broken cluster-side since the 2026-08 PE roll
   (`polaris_pbs_notes.md` §1). The handoff's asset table does not mention it, and
   back-porting only the fabric block would have produced a script that still
   could not start. Fixed by `polaris/polaris_ai_rossby_env.sh`.

3. **"Rerun the probes per venv" needed one addition, not just a copy.** makani's
   `polaris_makani_fabric_probe.pbs` tests six `/soft` pairings and does **not**
   include the self-built v1.21.1 the handoff mandates — that stack is only
   exercised by the separate knob matrix. A probe that decides the pin has to
   contain the pin, so the ai-rossby sibling adds combo **G** (v1.21.1 + AUTO).

## 7. Status

- [x] env bootstrap that survives the broken conda modulefile
- [x] three fabric probes, siblings under this venv (not submitted)
- [x] parser + 24 tests, `AI_ROSSBY_SCALING_PARSE_OK` on a login node
- [x] ladder launcher: shim + MASTER provisioning + corrected fabric block +
      preflight/spare + wrap guard + parser guards
- [x] the two dead spots in `polaris_sfno_e3sm_multinode.pbs` fixed
- [x] prereg (this file) written and committed **before** the first ladder job
- [ ] fabric probes run, pins recorded
- [ ] 1-node then 2-node smoke green on `AI_ROSSBY_MN_SCALING_OK`
- [ ] A/B/C/8n × 3 interleaved reps on one store, one config
- [ ] prereg scored in CHANGELOG, misses included
- [ ] prod rungs — only after the above, costs surfaced first, jesswan looped on
      any batch/LR change
