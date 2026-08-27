# HANDOFF — port the makani multi-node DDP capability to PanguWeather, ai-rossby, and ACE2

Read this whole file, then CHANGELOG.md entries 2026-08-23 → 2026-08-27 (the
makani campaign this generalizes), then `makani_multinode_ddp_plan.md` (the
method: prereg → measure → score, including the scored misses). Work in
`/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling`.
Branch off `profile/pangu-polaris-profiling` — **note it is UNMERGED with a PR
pending**, so your branch creates a *stacked* PR: say so in the PR body and in
CHANGELOG, and record the merge order (this repo has been burned by silent
stacking before).

**Mission:** give PanguWeather, physicsnemo-ai-rossby, and ACE2 the same
multi-node DDP capability makani now has on Polaris — measured scaling ladder
(1→8 nodes in debug queues), then prod-queue rungs — using the proven assets
below and *not* re-losing the week of failures they encode. The prize, from
makani's measured result: a ~290-node-hour / ~12-day single-node production
schedule becomes overnight at 32 nodes for ~1.3× the compute (CHANGELOG
2026-08-27: 128-node production in 1.7 h ≈ 215 node-hours). ⚠ Makani's ~96%
weak-scaling efficiency 8→128 nodes is a MAKANI measurement — a ~150 M-param
model whose gradient all-reduce is ~0.6 GB/step. Pangu's is ~4.7 GB/step on
~2× the compute; do not quote makani's efficiency for any other harness until
that harness's own 2-node smoke exists. Also budget wireup: measured 18 s at
32 ranks → ~300–316 s at 512 (once per job, grows superlinearly-ish).

**Queue geography (verified `qstat -Qf` this campaign):** `debug` 1–2 nodes
≤1 h; `debug-scaling` 1–10 nodes ≤1 h, ONE queued-or-running job per user;
`prod` routes by size → `small` (~10–24), `medium` (25–99), `large` (100–496,
≤24 h). Settle everything at ≤10 nodes; buy prod rungs only with a settled
config and surfaced cost.

## 0. Ground rules (all inherited, all enforced by evidence in CHANGELOG)

1. **Prereg before the first job.** Write falsifiable predictions per harness
   (mirror `makani_multinode_ddp_plan.md` §4) and score them honestly,
   including misses. Makani's prereg caught its own wrong assumptions twice.
2. **PASS tokens, never rc** (CLAUDE.md #14). Every job keys on a greppable
   token; a parser refuses rows whose logged `world_size` ≠ launched ranks
   (the N-solo-trainers failure mode produces *plausible* numbers).
3. **One variable per run.** The makani deadlock took 5 single-variable jobs
   to corner (`degenerate groups → Ring → app-free probe → RX_MATCH → Simple`);
   every multi-knob shortcut attempted during that chase would have produced
   an unattributable result.
4. **Measurement config is part of the row.** Plugin, NCCL_PROTO, pack, wandb
   on/off, epochs — makani's arm A legitimately measured 196.5 / 114.9 /
   144.7 / 115.3 ms under four configs. Rows from different configs never
   share a table; wandb-instrumented and multi-epoch rows go to a separate
   CSV (`-v MAKANI_SCALING_CSV=` pattern). CSV schema is a contract (#10):
   config goes in the log header and launcher pins, not new columns.
5. **Science is jesswan's** (CLAUDE.md). Weak scaling grows the global batch —
   for Pangu that is 4 → 64/128/512, a regime change dwarfing makani's.
   Profiling ladders are ours; any production run at a new batch/LR needs her
   sign-off. Surface prod-queue costs before submitting (memory policy).
6. **Never resubmit on `queue_tags`** (#12); ordinary "Not enough free nodes"
   accrues eligibility and must be left alone.
7. Repo ops on this Lustre: move files between worktrees with `cp`, never
   `git merge` (memory: git-hangs-on-polaris-lustre); login-node `python3` is
   3.6 — repo tooling needs `/usr/bin/python3.11`; login-node shell rules in
   CLAUDE.md apply (no builds/compute, qsub everything).

## 1. The reusable assets (all committed on `profile/pangu-polaris-profiling`)

| asset | path (under `makani_sfno/polaris/` unless noted) | reuse mode |
|---|---|---|
| PALS rank shim (PMI_* → RANK/WORLD_SIZE/LOCAL_RANK) | `polaris_rank_env.sh` (body must stay byte-identical to `physicsnemo_ai_rossby/polaris/polaris_rank_env.sh` — change BOTH or neither) | verbatim |
| corrected fabric block (pins + FI_CXI env + `--cpu-bind depth -d 8` + **existence-checked dirs**) | inside `polaris_makani_multinode_scaling.pbs` (§"plugin/libfabric pin") | copy the block |
| plugin pairing probe (6-combo matrix) | `polaris_makani_fabric_probe.pbs` | rerun per-venv |
| plugin runtime-knob matrix | `polaris_ofi_plugin_matrix.pbs` | rerun per-venv |
| app-free 16-rank/4-node NCCL probe (fail-fast, 3-min timeout) | `polaris_makani_nccl_mn_probe.pbs` | rerun per-venv |
| modern plugin build (v1.21.1 vs libfabric 2.3.1) | `polaris_build_aws_ofi_nccl.pbs`; installed at `$MEMBER_ROOT/sw/aws-ofi-nccl-1.21.1` | shared install, reuse directly |
| spare-node GPU preflight (`-v TARGET_NODES=N`, `-l select=N+1`) | block in the scaling launcher | copy the block |
| result parser + 11 tests (world_size / transport / schema guards) | `parse_makani_scaling.py`, `test_parse_makani_scaling.py` | adapt log grammar per harness, keep the guards |
| wandb key parity method (byte-compare vs reference datastores) | `src/sfno_training/trainer/wandb_diagnostics.py` + CHANGELOG 2026-08-27 "cont." | Pangu/ai-rossby already own the contract; ACE2 must adopt it |

**The two fabric stacks, measured (torch 2.8.0 / NCCL 2.28.3 — revalidate for
any other torch, §3):**

* `/soft/.../v1.6.0-libfabric-1.22.0` + cray libfabric 2.3.1 + `NCCL_PROTO=Simple`
  — fast for pure DDP (makani ladder 144.7/145.7/199.5/215.3 ms at 1/2/4/8
  nodes) **but its progress engine loses small-message broadcasts by TENSOR
  SIZE** (a 384×58 weight broadcast survived where 384×107 wedged — job
  7565896). "Works at N ranks with model X" does NOT imply model Y: treat any
  new model's first ≥3-node run as unproven regardless of makani's greens.
* self-built v1.21.1 + `OFI_NCCL_PROGRESS_MODEL=AUTO` — correct everywhere
  (incl. multi-communicator / spatial traffic), near-flat step time to 512
  ranks. **Default for production; the ONLY acceptable stack for anything
  with more than one active communicator.** The AUTO knob is mandatory: CXI
  refuses `fi_domain` (ENOSYS) without it — this is why every /soft build
  ≥v1.9 looks broken, and belongs in the ALCF ticket.

## 2. Mistakes ledger — each cost real time; do not pay twice

1. **A nonexistent dir on LD_LIBRARY_PATH is silently ignored.** The inherited
   fabric block pinned a deleted libfabric (`2.2.0rc1`) and ran against 2.3.1
   → segfault blamed elsewhere (7553811). *Every* load-bearing path gets an
   existence check with a hard exit.
2. **`cmd | tail` reports tail's exit status.** A crashed pack audit printed
   `PACK_ALLDATA_OK` anyway (7565734). Never pipe a gating command; capture
   rc directly.
3. **Verify contracts against artifacts, not documentation.** Two wandb key
   misses shipped past unit tests built on my *description* of Pangu's scheme
   (U10 parsed as "U at 10 hPa"; hybrid level floats where Pangu formats
   nominal labels — caught by the operator on a LIVE production run, cost a
   ~110-node-hour restart). The fix that works: byte-compare generated keys
   against the reference RUN DATASTORES (`wandb_local_history.py` /
   DataStore). Do this for every harness before any wandb'd production job.
4. **Fast-failing sick nodes boomerang.** `CUDA device busy` nodes return to
   the free pool in seconds and the scheduler re-deals them (x3111c0s37b1n0:
   3 strikes). The preflight+spare pattern is mandatory at ≥8 nodes and
   non-negotiable at 100+.
5. **Wrap guard:** a pack smaller than `STEPS×GLOBAL_BATCH` re-serves cached
   samples and biases step time *along the scaling axis*. Gate on the real
   sample count read from the pack (the guard refused two arms correctly and
   exposed that the "1-year" pack was actually a 400-sample smoke).
6. **Packs are not interchangeable:** same arm, same code read 196.5 vs
   114.9 ms on two packs (I/O regime). Every arm of a comparison shares one
   pack, named in the row's log.
7. **`CUDA_LAUNCH_BLOCKING=1` legitimately deadlocks NCCL** — it is NOT a
   safe IMA localizer for distributed code (7554283 wedged silently). Use
   `TORCH_NCCL_TRACE_BUFFER_SIZE` (flight recorder) instead — it named the
   faulting collective twice.
8. **PBS `-v` cannot carry commas** in values (`NCCL_PROTO=LL,LL128,Simple`
   fails as "cannot send environment"). Single-valued overrides only, or bake
   the set into the launcher.
9. **qsub resolves the script relative to cwd** — every submission `cd`s to
   the harness dir first (two "script file:: No such file" failures).
10. **`$TMPDIR` on eagle breaks DataLoader workers** — AF_UNIX socket paths
    exceed 108 bytes (`pymp-*/listener-*`). Launchers export `TMPDIR=/tmp`
    locally; never edit the shared `polaris_env.sh` for one consumer.
11. **Env vars set unconditionally by shared scripts eat qsub `-v` overrides**
    (`MAKANI_DATA` via `_pick()`). Give launchers their own override names
    (`PACK=`, `OFI_PLUGIN=`) and document why.
12. **Multi-communicator setup broadcasts violate NCCL ordering** if
    interleaved per-parameter (makani's own upstream comment concedes it).
    Pure DDP (one comm) is immune; anything creating extra process groups
    needs the serialized-sync pattern (`_serialized_sync_params`) — necessary
    hygiene, though NOT sufficient against the old plugin's progress bug.
13. **Don't diagnose scaling with rank-0 logs** — makani logs from rank 0
    only; the parser's world_size check and per-rank `NSYS` capture exist
    because aggregate logs cannot answer per-rank questions.
14. **wandb needs its four identity keys** (`wandb_name/group/project/entity`)
    — makani's configs never carried them and the first wandb'd job died on
    `AttributeError: wandb_name` (7564377). Render them per-run; project is
    pinned to `pedramh-profiling` (landing in the shared project is what makes
    the panels overlayable at all).
15. **Per-iteration wandb logging silently kills `step=epoch` logging** —
    auto-incremented steps run ahead and wandb DROPS out-of-order epoch logs
    without error. Re-log epoch scalars flat (makani's `log_epoch` override is
    the pattern); any harness adopting per-iteration metrics must audit its
    existing `wandb.log(..., step=...)` call sites first.
16. **Interleave reps, never batch them** — two runs of an IDENTICAL config
    once measured 42.2% vs 37.4% for the same quantity (CHANGELOG §4.4c).
    Three back-to-back reps of arm A then three of arm B confounds arm with
    time-of-day/allocation; A,B,C,A,B,C,... does not. This is a validity
    requirement, not tidiness.

## 3. Per-harness plan

### 3a. ai-rossby (physicsnemo) — smallest delta, do first

* **Env:** its own venv, **torch 2.10 + a different bundled NCCL. Every
  plugin behavior above was measured under 2.8.0's NCCL — REVALIDATE before
  trusting any pin**: run the pairing probe + knob matrix + app-free mn probe
  under the ai-rossby venv (minutes each). Expect AUTO to still be required
  (it's a CXI-side constraint) but *measure* it.
* **Launcher:** `physicsnemo_ai_rossby/polaris/polaris_sfno_e3sm_multinode.pbs`
  already exists — it is the ORIGIN of the stale fabric block. Back-port the
  corrected block + preflight + parser guards. DistributedManager consumes
  the shim it already ships (verify body still byte-matches makani's copy).
* **Data:** existing zarr stores — but there are TWO lineages (fact inlined
  here because it lives only in session memory otherwise): the *allyears*
  store (awikner v3, transferred) and the *peryear* SeqZarr repack (our
  fork's v2), with a known 103→96 channel drift between generations. Confirm
  WHICH store, verify it (`SEQZARR_VERIFIED` tooling exists), and check the
  wrap-guard arithmetic against its real sample counts before measuring
  anything.
* **Ladder:** A/B/C/8n × 3 interleaved reps, prereg first. `debug` 1–2 nodes,
  `debug-scaling` 3–10 (max 10 nodes / 1 h / 1 job per user, verified).

### 3b. PanguWeather — same env as makani, new launcher required

* **Env:** base-conda torch 2.8.0 — the fabric findings apply AS MEASURED,
  no revalidation. `PANGU_*` knobs, `$POLARIS_TOPUPS` on PYTHONPATH (never
  for SFNO — but Pangu NEEDS it; copy the env bootstrap from an existing
  green Pangu PBS script verbatim, per CLAUDE.md §Smokes). ⚠ Verify the env
  FIRST against Pangu's most recent green jobs (7551401+ ran fine on
  2026-08-21 despite the broken conda modulefile — whatever bootstrap those
  used is the one to copy; check whether ALCF has since fixed
  `module load conda` before assuming either way). Data: Pangu reads the full
  51,100-sample archive directly, so the wrap guard is generous — but state
  the arithmetic in the prereg anyway.
* **DDP:** `PanguWeather/v2.0/train.py:3954` inits with `env://` — the shim
  feeds it verbatim. Multi-node launch = `mpiexec … bash polaris_rank_env.sh
  python …` replacing single-node `torchrun --standalone`.
* **Guardrails specific to Pangu:** its bench instrumentation (NVTX ranges,
  CSV columns) is a frozen cross-project contract (#10) and its production
  lineage runs through existing scripts — build a NEW sibling launcher; edit
  nothing in place. `train.py` vs `train_optimized.py` inversion trap (#4).
  PYTHONPATH must contain exactly one of {`s2s/v2.0`, `PanguWeather/v2.0`}.
* **Watch:** 1.18 B params ⇒ ~4.7 GB gradient all-reduce per step (~8× makani
  per unit compute) and a ~14 GB checkpoint whose rank-0 write does not
  parallelize — at high node counts checkpoint cadence dominates epoch time;
  budget it explicitly. First measurement: a 2-node smoke to pin step time
  before believing any projection.

### 3c. ACE2 (AI2, `ACE2_retrain/`) — SURVEY FIRST; do not assume makani's shape

ACE2's Polaris status is NOT established in the CHANGELOG the way the other
harnesses' is. Before any multi-node work, a survey job must answer, with
evidence in the CHANGELOG:
1. Is there a green **single-node** Polaris baseline (env, data, PASS token)?
   Multi-node work starts from a green single-node run — makani's did
   (7253465) and every diagnosis leaned on that anchor. If none exists,
   bring-up comes first and this handoff pauses for ACE2.
2. How does the vendored `fme` package initialize distributed? (Find the
   `init_process_group` call and its init method; if `env://`, the shim
   applies; if not, document what does.)
3. Which venv/torch — and therefore whether the fabric matrix needs a rerun
   (any torch ≠ 2.8.0 does).
4. Data: what stores exist on eagle, sample counts for the wrap guard, and
   whether the HEALPix grid changes the loader's sharding story (note the
   `fme.core.hpx` reorder tables are repo-tracked and required).
5. Metrics: ACE2 has NO wandb contract yet. Decide explicitly (operator +
   jesswan) whether it adopts the 102-key lwrmse scheme — its variable set
   differs, so expect the makani situation: same *scheme*, shared keys only
   where physics overlaps, byte-verified against the reference datastores
   (mistake #3) before any production logging claim.

### 3d. Shared acceptance criteria (each harness, before any prod rung)

- [ ] fabric matrix run under the harness's own venv; pins recorded
- [ ] shim + fabric block + preflight in a sibling launcher; smoke green on
      its own PASS token at 1 node, then 2
- [ ] parser guards (world_size, transport, schema) with tests, green
- [ ] prereg written and committed BEFORE the ladder
- [ ] A/B/C/8n × 3 interleaved reps on ONE pack/store, one config
- [ ] prereg scored in CHANGELOG, misses included
- [ ] prod rungs (16/32/…) only after the above, costs surfaced first,
      jesswan looped on any batch/LR change

## 4. What NOT to port

- makani's spatial parallelism (`h/w` groups) — Pangu/ai-rossby/ACE2 are
  DDP-only codebases; their batch ceiling is real and is a science limit.
- `_serialized_sync_params` as code — it patches makani's trainer; port the
  *lesson* (mistake #12) only if a harness creates extra comm groups.
- The Simple-protocol pin as a default — it is an old-plugin workaround with
  a measured cost; on the v1.21.1 stack it is unnecessary.

## 4b. The ALCF ticket — file it, it is part of this work

Nobody has filed it yet. Contents, all evidenced in CHANGELOG 2026-08-26/27:
(a) every `/soft/libraries/aws-ofi-nccl` build ≥v1.9 fails `fi_domain` with
ENOSYS against libfabric 2.3.1 — including ALCF's own 2025-09 rebuilds —
because the CXI provider requires auto progress and newer plugin defaults
don't request it; the one-line fix is `OFI_NCCL_PROGRESS_MODEL=AUTO`
(measured: matrix job 7563894); (b) the surviving v1.6.0 build's LL protocol
deadlocks ≥3 nodes and its progress engine drops small broadcasts by tensor
size (app-free reproduction: `polaris_makani_nccl_mn_probe.pbs`); (c) three
nodes with zombie-GPU state (`CUDA device busy` at init): **x3111c0s37b1n0**
(3 strikes), **x3201c0s1b1n0**, **x3109c0s1b0n0**.

## 5. Definition of done

Three CHANGELOG entries (one per harness) each containing: the revalidated
fabric matrix result, a 4-point ladder table with reps, scored prereg, and —
for any harness taken to production scale — the same artifacts makani has:
pinned-RUN_NUM resumable launcher, checkpoint policy, wandb parity evidence,
and a checkpoint-usage doc modeled on
`makani_sfno/docs/2026-08-27_prod128_alldata_checkpoint_usage.md`.
