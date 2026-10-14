# HANDOFF — port the makani multi-node DDP capability to PanguWeather, ai-rossby, and ACE2

Read this whole file, then CHANGELOG.md entries 2026-08-23 → 2026-08-27 (the
makani campaign this generalizes), then the makani results at repo root:
`/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/makani_bench_report.md`
(the method: prereg → measure → score, including the scored misses — §7). Work in
that repo. **Definitions used throughout:** `$MEMBER_ROOT` =
`/eagle/projects/lighthouse-uchicago/members/mehta5` (exported by
`polaris_env.sh`, which every launcher sources); "the ladder" = weak-scaling
arms **A=1 node, B=2 nodes, C=4 nodes, 8n=8 nodes**, 1 sample/GPU (global
batch = 4×nodes), 60 timed steps per arm.

Branch off `profile/pangu-polaris-profiling` — **it is UNMERGED with a PR
pending**, so your branch creates a *stacked* PR: say so in the PR body and in
CHANGELOG, and record the merge order (this repo has been burned by silent
stacking before). This document was reviewed by two independent compute-node
Fable 5 critics (job 7568383, reports under
`$MEMBER_ROOT/runs/critic_handoff/7568383/`); their findings are folded in
below — if you find the doc contradicting the repo, trust the repo and log it.

**Mission:** give PanguWeather, physicsnemo-ai-rossby, and (survey permitting)
ACE2 the same multi-node DDP capability makani now has on Polaris — measured
scaling ladder in debug queues, then prod-queue rungs — using the proven
assets below and *not* re-losing the week of failures they encode.

**The economics, with every number labeled** (all in CHANGELOG 2026-08-27):
- PanguWeather single-node production: **121.7 h measured for ~41 epochs**
  (wandb `_runtime`, run j796bp1k) ⇒ ~297 node-hours per 100-epoch schedule,
  ~12 days of runtime across queue links.
- makani 128-node production (`prod128_alldata_v2`, **v1.21.1+AUTO stack**,
  makani's ~150 M-param model, ITS OWN 100-epoch/43,800-sample schedule —
  NOT the same work as Pangu's): **1.67 h ≈ 215 node-hours**.
- Projection for Pangu at 32 nodes (unmeasured — pin with a 2-node smoke
  before repeating it): ~12 h wall, ~400 node-hours ≈ 1.3× its single-node
  cost for ~24× the speed.
⚠ Do NOT quote makani's scaling efficiency for any other harness. Makani's
gradient all-reduce is ~0.6 GB/step; Pangu's is ~4.7 GB/step on ~2× the
per-step compute — **~8× the comms in absolute terms, ~4× per unit compute**.
Wireup is measured at TWO points on the v1.21.1 stack — **18 s at 32 ranks,
299–316 s at 512 ranks** (jobs 7565972; 7566045/7566145) — budget it per job
and treat the growth law between those points as unknown.

**Queue geography (verified `qstat -Qf` this campaign, cross-checked with
CLAUDE.md's cluster table):** `debug` 1–2 nodes ≤1 h; `debug-scaling` 1–10
nodes ≤1 h, ONE queued-or-running job per user; `prod` routes by size →
`small` (~10–24), `medium` (25–99), `large` (100–496, ≤24 h); **`capacity`**
(≤168 h — the designated long single-node queue, `max_run 1` per PROJECT: it
blocks every group member, never take it without the operator) and
**`preemptable`** (≤72 h, 1–10 nodes, **10 concurrent per project** — its start
latency is load-dependent, not absent: it runs only on nodes `prod` isn't using,
so 0/9 started on 2026-08-05 and 22 were running on 2026-09-02). A >1 h job of
3–9 nodes has no clean home in the deterministic queues — restructure it
(shorter epochs, resume chains) or accept `preemptable`'s latency; since resume
is proven (2026-09-02) a preemption now costs ~1 epoch, but our launchers are
`-r n` and would be killed rather than requeued.

## 0. Ground rules (all inherited, all enforced by evidence in CHANGELOG)

1. **Prereg before the first job.** Falsifiable predictions per harness
   (mirror the plan file's §4), scored honestly afterwards, misses included.
2. **PASS tokens, never rc** (CLAUDE.md #14). Every job keys on a greppable
   token; the parser refuses rows whose logged `world_size` ≠ launched ranks.
3. **One variable per run.** The makani deadlock took 5 single-variable jobs
   to corner; any multi-knob shortcut yields an unattributable result.
4. **Measurement config is part of the row.** Plugin, NCCL_PROTO, pack, wandb
   on/off, epochs — makani's arm A legitimately measured 196.5 / 114.9 /
   144.7 / 115.3 ms under four configs (smoke pack; scaling pack + default
   proto; + Simple pin; v1.21.1 stack). Rows from different configs never
   share a table; wandb-instrumented and multi-epoch rows go to a separate
   CSV (`-v MAKANI_SCALING_CSV=` pattern). CSV schema is a contract (#10):
   config lives in the log header and launcher pins, not new columns.
5. **Science is jesswan's** (CLAUDE.md). Weak scaling grows the global batch —
   for Pangu that is 4 → 64/128/512, a regime change dwarfing makani's.
   Profiling ladders are ours; production at a new batch/LR needs her
   sign-off. Surface prod-queue costs before submitting (memory policy).
6. **Never resubmit on `queue_tags`** (#12); ordinary "Not enough free nodes"
   accrues eligibility and must be left alone.
7. Repo ops on this Lustre: move files between worktrees with `cp`, never
   `git merge`; login-node `python3` is 3.6 — repo tooling needs
   `/usr/bin/python3.11`; login-node shell rules in CLAUDE.md apply (no
   builds/compute/agents on login nodes; qsub everything).
8. **Driving the ladder within those shell rules:** interleaved reps (rule
   #16 below) across serial queues means many submit-wait cycles, and
   login-node rules forbid qstat polling loops. Use PBS dependency chains
   submitted in one shot — `qsub -W depend=afterany:<jobid> …` — one chain
   per queue, respecting `debug-scaling`'s one-job limit (each link becomes
   eligible only as its predecessor ends). A single driver script submitted
   as a job is the alternative.

## 1. The reusable assets

All paths repo-relative to
`/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/`.
All are committed on `profile/pangu-polaris-profiling` as of `a939f634`
(including the two previously-untracked ai-rossby files — a critic finding).

| asset | path | reuse mode |
|---|---|---|
| PALS rank shim (PMI_* → RANK/WORLD_SIZE/LOCAL_RANK) | `makani_sfno/polaris/polaris_rank_env.sh` ≡ `physicsnemo_ai_rossby/polaris/polaris_rank_env.sh` (executable bodies byte-identical, verified 2026-08-27 — change BOTH or neither) | verbatim |
| **MASTER_ADDR / MASTER_PORT provisioning** — the shim does NOT set these; `env://` init hard-requires them. The launcher derives `MASTER_ADDR=$(head -1 $PBS_NODEFILE)` and exports a fixed `MASTER_PORT` before `mpiexec` | inside `makani_sfno/polaris/polaris_makani_multinode_scaling.pbs` and both probe scripts | copy the block — a launcher missing it fails at `init_process_group` on the first 2-node smoke |
| corrected fabric block (pins + FI_CXI env + `--cpu-bind depth -d 8` + existence-checked dirs) | `makani_sfno/polaris/polaris_makani_multinode_scaling.pbs` §"plugin/libfabric pin" | copy the block |
| plugin pairing probe (6-combo matrix) | `makani_sfno/polaris/polaris_makani_fabric_probe.pbs` | rerun per-venv |
| plugin runtime-knob matrix | `makani_sfno/polaris/polaris_ofi_plugin_matrix.pbs` | rerun per-venv |
| app-free 16-rank/4-node NCCL probe (3-min fail-fast) | `makani_sfno/polaris/polaris_makani_nccl_mn_probe.pbs` | rerun per-venv |
| modern plugin build + install | `makani_sfno/polaris/polaris_build_aws_ofi_nccl.pbs`; installed at `$MEMBER_ROOT/sw/aws-ofi-nccl-1.21.1` | shared install, reuse directly |
| spare-node GPU preflight (`-v TARGET_NODES=N`, `-l select=N+1`) | block in the scaling launcher | copy the block |
| result parser + 11 tests (world_size / transport / schema guards) | `makani_sfno/polaris/parse_makani_scaling.py`, `…/test_parse_makani_scaling.py` | adapt log grammar per harness, keep the guards |
| wandb key parity method | `makani_sfno/src/sfno_training/trainer/wandb_diagnostics.py`; verification tool `wandb_local_history.py` (repo root); **reference datastores**: Pangu `$MEMBER_ROOT/wandb/wandb/run-20260807_150638-j796bp1k/run-j796bp1k.wandb`, ai-rossby `$MEMBER_ROOT/runs/ai_rossby_sfno_e3sm/sfno_e3sm_parity01/wandb/wandb/run-20260812_165521-03ha0747/run-03ha0747.wandb` | byte-compare generated keys against BOTH datastores |

**Reusing the probes under a different venv:** the probe scripts source
makani's env bootstrap. For ai-rossby, create siblings under
`physicsnemo_ai_rossby/polaris/` (never edit the makani ones in place — #7,
and both dirs are subtrees: keep edits minimal/contiguous) whose only delta is
the env-activation block (`$AI_ROSSBY_VENV` = `$MEMBER_ROOT/conda-envs/
ai-rossby-venv`, per `polaris_env.sh`) — the rank program and matrix logic
copy over unchanged.

**Fabric stack for ALL new-harness bring-up — there is only one choice:**

* **Self-built aws-ofi-nccl v1.21.1** (`$MEMBER_ROOT/sw/aws-ofi-nccl-1.21.1`)
  + cray libfabric 2.3.1 + **`OFI_NCCL_PROGRESS_MODEL=AUTO`** (mandatory:
  CXI refuses `fi_domain` with ENOSYS without it — the reason every /soft
  build ≥v1.9 looks broken; ALCF ticket, §4b). Correct for every traffic
  pattern tested, incl. multi-communicator; measured near-flat step time on
  makani: 490.7 / 460.5 / 545.0 ms at 2/4/8 nodes (53-ch), 492.7 ms at 8
  nodes and ~509 ms at 512 ranks (101-ch) — those are makani numbers, label
  any reuse of them as such (rule #4).
* The old `/soft/libraries/aws-ofi-nccl/v1.6.0-libfabric-1.22.0` +
  `NCCL_PROTO=Simple` stack is **historical context for makani's early
  ladder rows only (144.7/145.7/199.5/215.3 ms) — DO NOT SELECT IT for a new
  model.** Its progress engine loses small-message broadcasts **by tensor
  size** (a 384×58 DDP weight broadcast survived where 384×107 wedged, job
  7565896) — and DDP's initial parameter broadcast is exactly that traffic.
  A new model's tensor shapes make it a lottery ticket, and it deadlocked a
  1.18 B-model-sized broadcast is untested territory. If a measured
  side-by-side is ever wanted, run it AFTER the v1.21.1 ladder is green,
  as its own config.
* Any venv whose torch ≠ 2.8.0 (ai-rossby's 2.10; possibly ACE2's) bundles a
  different NCCL — **rerun all three probes under that venv before trusting
  any pin, including AUTO** (expected to hold — it is CXI-side — but measure).

## 2. Mistakes ledger — each cost real time; do not pay twice

1. **A nonexistent dir on LD_LIBRARY_PATH is silently ignored.** The inherited
   fabric block pinned a deleted libfabric and ran against another → segfault
   blamed elsewhere (7553811). Every load-bearing path gets an existence
   check with a hard exit.
2. **`cmd | tail` reports tail's exit status.** A crashed pack audit printed
   its OK token anyway (7565734). Never pipe a gating command; capture rc.
3. **Verify contracts against artifacts, not documentation.** Two wandb key
   misses shipped past unit tests built on a *description* of Pangu's scheme
   (U10 parsed as upper-air "U at 10 hPa"; hybrid level floats where Pangu
   formats nominal labels) — caught by the operator on a LIVE run, cost a
   ~110-node-hour restart. The working fix: byte-compare generated keys
   against the reference datastores listed in §1. Mandatory per harness
   before any wandb'd production job.
4. **Fast-failing sick nodes boomerang** (x3111c0s37b1n0: 3 strikes). The
   preflight+spare pattern is mandatory at ≥8 nodes.
5. **Wrap guard:** a pack smaller than `STEPS×GLOBAL_BATCH` re-serves cached
   samples and biases step time along the scaling axis. Gate on the real
   sample count read from the store.
6. **Packs/stores are not interchangeable:** same arm, same code read 196.5
   vs 114.9 ms on two packs. One store per comparison, named per row.
7. **`CUDA_LAUNCH_BLOCKING=1` legitimately deadlocks NCCL** (7554283). Use
   the flight recorder (`TORCH_NCCL_TRACE_BUFFER_SIZE`) to localize instead.
8. **Unquoted comma values in PBS `-v` fail** ("cannot send environment").
   Quoting per PBS Pro docs was NOT tested here — verify before relying on
   it, or keep overrides single-valued / baked into the launcher.
9. **qsub resolves the script relative to cwd** — `cd` to the harness dir
   first, every time.
10. **`$TMPDIR` on eagle breaks DataLoader workers** — AF_UNIX socket paths
    exceed 108 bytes. Launchers export `TMPDIR=/tmp` locally; never edit the
    shared `polaris_env.sh` for one consumer.
11. **Shared-script env vars eat qsub `-v` overrides** (`MAKANI_DATA` is
    reassigned unconditionally). Launchers take their own override names
    (`PACK=`, `OFI_PLUGIN=`).
12. **Multi-communicator setup broadcasts violate NCCL ordering** if
    interleaved per-parameter. Pure DDP (one comm) is immune; a harness that
    creates extra process groups needs the serialized-sync lesson (see
    `_serialized_sync_params` in `makani_sfno/src/sfno_training/trainer/
    plasim_trainer.py` — port the pattern, not the code).
13. **Don't diagnose scaling with rank-0 logs** — aggregate logs cannot
    answer per-rank questions; hence the parser guards and per-rank NSYS.
14. **wandb needs its four identity keys** (`wandb_name/group/project/
    entity`) — first wandb'd job died on `AttributeError: wandb_name`
    (7564377). Render per-run: name = the run's RUN_NUM, group = per-harness,
    project = `pedramh-profiling`, entity = YAML `null`. **Workflow:** runs
    are OFFLINE on the cluster (`WANDB_MODE=offline`, CLAUDE.md); panels
    become overlayable only after `wandb sync <offline-run-dir>` **from a
    login node** (compute nodes need the proxy; login has direct net + the
    user's `~/.netrc` credential). Syncing publishes — do it when the
    operator asks or at declared milestones, not silently.
15. **Per-iteration wandb logging silently kills `step=epoch` logging** —
    wandb drops out-of-order steps without error. Re-log epoch scalars flat
    (makani's `log_epoch` override is the pattern); audit a harness's
    existing `wandb.log(..., step=...)` call sites before adding
    per-iteration metrics.
16. **Interleave reps, never batch them** — two runs of an IDENTICAL config
    once measured 42.2% vs 37.4% for the same quantity (CHANGELOG §4.4c).
    A,B,C,A,B,C — not AAABBBCCC. Validity requirement, not tidiness.

## 3. Per-harness plan

### 3a. ai-rossby (physicsnemo) — smallest delta, do first

* **Env:** `$AI_ROSSBY_VENV` (`$MEMBER_ROOT/conda-envs/ai-rossby-venv`),
  **torch 2.10 + a different bundled NCCL ⇒ full probe revalidation (§1)
  before any pin is trusted.**
* **Launcher:** `physicsnemo_ai_rossby/polaris/polaris_sfno_e3sm_multinode.pbs`
  exists and is now committed — it is the ORIGIN of the stale fabric block.
  Back-port the corrected block + MASTER provisioning + preflight + parser
  guards. DistributedManager consumes the shim it already ships.
* **Data:** TWO zarr lineages exist — *allyears* (awikner v3, transferred)
  vs *peryear* (our fork's v2 SeqZarr repack), with a known 103→96 channel
  drift between generations. **Criterion: use the store the existing
  ai-rossby production run trained on** (identify it from that run's rendered
  config / startup log under `$MEMBER_ROOT/runs/ai_rossby_sfno_e3sm/`) so
  ladder rows are comparable to the only production evidence. Verify it with
  the store verifier (`physicsnemo_sfno/polaris/polaris_verify_store.pbs`,
  PASS `SEQZARR_VERIFIED`) and check wrap-guard arithmetic against its real
  sample count.
* **Ladder:** A/B/C/8n × 3 interleaved reps (definitions at top), prereg
  first, dependency-chained per §0.8.

### 3b. PanguWeather — fabric findings likely apply, but VERIFY, then a new launcher

* **Env — establish it, do not assume it:** Pangu runs the **base conda**
  (not makani's SFNO venv — but note the SFNO venv is `--system-site-packages`
  over that same base conda, so both *should* resolve to the same torch 2.8.0
  / NCCL 2.28.3 binary — which is the actual basis for reusing the fabric
  pins). **The waiver of probe revalidation is conditional:** confirm from a
  green Pangu job log that its torch/NCCL versions match 2.8.0/2.28.3; if
  they do, skip the probes and record why; if not, §3d's first checkbox
  applies in full. The known-green bootstrap lives in the Pangu equivalence
  scripts that ran 2026-08-21 despite the broken conda modulefile:
  **`polaris_equiv_baseline.pbs` / `polaris_equiv_dhconv.pbs` at repo root**
  (jobs 7551401+; see CHANGELOG 2026-08-21 and
  `prompts/pangu_polaris_loop_journal.md`). Copy that bootstrap verbatim.
  `$POLARIS_TOPUPS` (defined in `polaris_env.sh`) is the base-conda top-up
  dir (netCDF4, torch_harmonics **0.7.4**, …): Pangu NEEDS it on PYTHONPATH;
  SFNO jobs must never see it (0.7.4 would shadow the venv's 0.9.x).
* **DDP:** Pangu's trainer initializes with `init_method='env://'` — grep
  `PanguWeather/v2.0/train.py` for `init_process_group` (near line 3954 as
  of commit 4133802a). The shim + MASTER provisioning feed it; multi-node
  launch = `mpiexec … bash polaris_rank_env.sh python …` replacing the
  single-node `torchrun --standalone`.
* **Ladder config:** use the bench-instrumented `train.py` (NEVER
  `train_optimized.py` — CLAUDE.md #4) with the config family the green
  Polaris bench jobs used (`polaris_bench_pangu_plasim.pbs` at repo root
  shows the rendered-config pattern); per-GPU batch 1. ⚠ CLAUDE.md #13:
  `test.yaml` is the FULL model and OOMs despite its name.
* **Guardrails specific to Pangu:** bench instrumentation (NVTX ranges, CSV
  columns) is a frozen cross-project contract (#10) and the production
  lineage runs through existing scripts — build a NEW sibling launcher; edit
  nothing in place. PYTHONPATH must contain exactly one of {`s2s/v2.0`,
  `PanguWeather/v2.0`}.
* **Watch:** 1.18 B params ⇒ ~4.7 GB gradient all-reduce per step and a
  ~14 GB checkpoint whose rank-0 write does not parallelize — at high node
  counts checkpoint cadence can dominate epoch time; budget it explicitly.
  **First measurement: a 2-node smoke to pin step time** before believing
  any projection in this document.

### 3c. ACE2 (AI2) — SURVEY FIRST; do not assume anything makani-shaped

0. **Confirm the harness even exists here:** `ACE2_retrain/` at repo root
   (present in recent working trees; NOT one of CLAUDE.md's six enumerated
   codebases). If absent on your branch, or its addition to the roadmap
   (DESIGN §8) is unrecorded, STOP and surface to the operator — do not
   fetch code from upstream on your own authority.
1. Is there a green **single-node** Polaris baseline (env, data, PASS token)
   in CHANGELOG? Multi-node work starts from a green single-node anchor —
   makani's diagnosis chain leaned on its (7253465) constantly. If none,
   bring-up precedes this handoff for ACE2.
2. How does the vendored `fme` package initialize distributed? Find its
   `init_process_group` call: `env://` ⇒ the shim applies; anything else,
   document what does.
3. Which venv/torch — any torch ≠ 2.8.0 triggers the §1 probe revalidation.
4. Data: stores on eagle, sample counts for the wrap guard, and whether the
   HEALPix grid (the `fme.core.hpx` reorder tables are repo-tracked and
   required) changes the loader/sharding story.
5. Metrics: ACE2 has NO wandb contract. The candidate is the 102-key
   per-channel lwrmse scheme (defined in `makani_sfno/src/sfno_training/
   trainer/wandb_diagnostics.py`; provenance CHANGELOG 2026-08-26/27).
   **This is a designed human gate:** draft a one-page adoption memo
   (which ACE2 variables map to which keys, what cannot map) and HALT for
   the operator + jesswan — do not implement metrics on your own authority.

### 3d. Shared acceptance criteria (each harness, before any prod rung)

- [ ] fabric probes run under the harness's own venv and pins recorded — OR,
      Pangu only, the §3b version-match waiver satisfied and logged
- [ ] shim + MASTER provisioning + fabric block + preflight in a sibling
      launcher; smoke green on its own PASS token at 1 node, then 2
- [ ] parser guards (world_size, transport, schema) with tests, green
- [ ] prereg written and committed BEFORE the ladder
- [ ] A/B/C/8n × 3 interleaved reps on ONE store, one config (v1.21.1+AUTO)
- [ ] prereg scored in CHANGELOG, misses included
- [ ] prod rungs (16/32/…) only after the above, costs surfaced first,
      jesswan looped on any batch/LR change

## 4. What NOT to port

- makani's spatial parallelism (`h/w` groups) — the other harnesses are
  DDP-only; their batch ceiling is a science limit, not an engineering gap.
- `_serialized_sync_params` as code — port the lesson (#12) only if a
  harness creates extra comm groups.
- The Simple-protocol pin — an old-plugin workaround; unnecessary and
  unmeasured on the v1.21.1 stack.

## 4b. The ALCF ticket — DRAFT it; the operator files it

Filing is outward-facing communication under the user's identity: **prepare
the ticket body and hand it to the operator; do not submit it yourself.**
Contents, all evidenced in CHANGELOG 2026-08-26/27: (a) every
`/soft/libraries/aws-ofi-nccl` build ≥v1.9 fails `fi_domain` with ENOSYS
against libfabric 2.3.1 — including ALCF's own 2025-09 rebuilds — because the
CXI provider requires auto progress and newer plugin defaults don't request
it; one-line fix `OFI_NCCL_PROGRESS_MODEL=AUTO` (matrix job 7563894); (b) the
surviving v1.6.0 build's LL protocol deadlocks ≥3 nodes and its progress
engine drops small broadcasts by tensor size (app-free repro:
`makani_sfno/polaris/polaris_makani_nccl_mn_probe.pbs`); (c) zombie-GPU
nodes: **x3111c0s37b1n0** (3 strikes), **x3201c0s1b1n0**, **x3109c0s1b0n0**.

## 5. Definition of done

Three CHANGELOG entries (one per harness) each containing: the fabric
matrix result (or Pangu's logged waiver), a 4-arm ladder table with 3
interleaved reps, scored prereg, and — for any harness taken to production
scale — the same artifacts makani has: pinned-RUN_NUM resumable launcher,
checkpoint policy, wandb parity evidence byte-checked against the §1
datastores, and a checkpoint-usage doc modeled on
`makani_sfno/docs/2026-08-27_prod128_alldata_checkpoint_usage.md`.
