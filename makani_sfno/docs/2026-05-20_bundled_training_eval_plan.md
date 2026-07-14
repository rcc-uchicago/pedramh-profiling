# Bundled training + eval in a single SLURM job

Date: 2026-05-20
Author: Zhixing Liu (with Claude)
Status: PLAN v3.1 — approved after third Codex pass (doc nits only)

## Revision history

- **v1 (2026-05-20)**: initial plan.
- **v2 (2026-05-20)**: addresses first Codex review. Key changes: explicit
  `RUN_DIR` derivation contract; prelude converted to a returning function
  (no `exit` paths when sourced); explicit per-stage failure reporting via
  `bundled_eval_status.txt` and `TRAIN=OK EVAL=FAIL` status mail; per-submit
  walltime table grounded in actual `-t` values; collision guard tightened
  to `ALLOW_RERUN=1`; baseline (gh / GH200) deferred from first rollout;
  added `submit_zgplev_group_clone.slurm` to the production list; clarified
  per-submit `BUNDLED_EVAL` defaults.
- **v3 (2026-05-20)**: addresses second Codex review. Key changes:
  (1) errexit-safe call sites for every helper-invoked command using
  `if cmd; then rc=0; else rc=$?; fi`; (2) eval tail budget raised to
  1.7–2.7 h to account for the first-eval climatology build
  (`submit_eval_score.slurm:5,18` confirms ~30–60 min, stored under each
  fresh `$OUT_ROOT` with no cache); (3) fallback status file in
  `logs/bundled_eval_status_${SLURM_JOB_ID}.txt` for early-skip / pre-prelude
  failures; (4) standalone `submit_eval.sh` behaviour change called out
  intentionally; (5) rollout smoke comparison uses two distinct `RUN_TAG`s;
  (6) h100 partition timelimit is **unlimited** per `sinfo`, not 24h —
  proposed walltimes remain valid.
- **v3.1 (2026-05-20)**: doc nits from third Codex pass (approval review).
  Corrected CPU-stage cost estimate from "~12 min" to ~45–75 min (climatology
  dominates) in §3 and §5; wording cleanup so every reference to
  `$OUT_ROOT/bundled_eval_status.txt` is qualified by "once the prelude
  resolves `$OUT_ROOT`" or paired with the always-on fallback log per §4.9.

## 1. Motivation

H100 queue waits at Stampede3 have grown to multi-hour / next-morning ranges
(e.g. 2026-05-20 17:00 PT → 2026-05-21 07:00 estimated start for two pending
`sfno_eval_inf` jobs). Because we always evaluate the canonical EMA checkpoint
after training a production-class run, splitting training and eval into
separate SLURM submissions pays this queue wait twice.

Goal: when a production training submit finishes, run the full eval pipeline
**inside the same allocation**, so the eval starts in seconds instead of hours.
Skip eval cleanly when the run did not produce a new canonical checkpoint;
report eval failure separately from training failure.

## 2. Decisions

From the 2026-05-20 interview + Codex review:

| Question | Decision |
|---|---|
| Eval scope inside the job | **Full eval** (inference → score → report → figures), all 4 stages |
| Apply to which submits | h100 production: full, v10 group_clone, v11, v11_clip, v10/v11_clip warm-start. **Baseline (gh / GH200) deferred.** |
| Failure handling, gating | **Skip eval if no new ckpt this run**; gate score/report/figures on earlier stages; eval failure does NOT fail the SLURM job, but is reported separately |
| Eval MODE | `nwp` (12 ICs/yr × 8 yr × K=56) |
| Warm-start chain timing | **Only the final chunk** runs eval (operator sets `BUNDLED_EVAL=1` only on final chunk) |
| "New ckpt" detection | **EMA ckpt mtime newer than `$JOB_START_EPOCH`** captured before the training step |
| Status reporting | Write `$OUT_ROOT/bundled_eval_status.txt` once the prelude succeeds, plus an always-on fallback `logs/bundled_eval_status_${SLURM_JOB_ID}.txt` for early-skip / pre-prelude paths (see §4.9); final mail/log line says `TRAIN=OK EVAL=OK|FAIL|SKIP` |
| Re-use of existing `$OUT_ROOT` | **Never silently** — require `ALLOW_RERUN=1` (closes Codex Q2) |

## 3. Findings from the code audit

1. **Training is single-node** (`-N 1`, `--nproc_per_node=4`). The "wasted
   multi-node alloc during eval tail" concern is moot.

2. **Eval inference is single-node h100** (`scripts/submit_eval_inference.slurm`,
   `-p h100 -N 1`, walltime 06:00:00, docstring "~1.5 h actual" for 96 NWP +
   climate). NWP-only is likely closer to ~1 h but assume up to ~1.5 h.

3. **Eval score / report / figures normally run on skx-dev** (CPU-only).
   Running them inside an h100 job burns h100 SUs on CPU work — and this
   is **not negligible**: score is ~35–65 min (climatology build dominates;
   see §4.5 + `submit_eval_score.slurm:5,18`), plus ~7 min for report +
   figures. Total CPU-stage h100 SU spend is ~45–75 min, not ~12 min as
   the v1 plan estimated. Still an acceptable trade-off vs queue waits
   (potentially 12+ hours), but a meaningful cost that motivates the
   deferred climatology-cache optimisation in §4.5.

4. **`scripts/submit_eval.sh` already has `BLOCKER_JOB_ID` afterok chaining**
   (lines 230–234). That is the "re-queue after training" path — NOT what
   we want, since the eval inference job would re-enter the h100 queue and
   lose the saved wait.

5. **Per-submit run-dir is implicit.** Each training submit defines
   `EXP_DIR` and `FULL_TPL` (and resolves the trainer run dir later as
   `$EXP_DIR/<config>/0`). They do NOT export `RUN_DIR`. Under `set -u`,
   bundled-eval would fail. Contract: each affected submit must export
   `CONFIG_NAME` and `RUN_NUM` (default 0), and `bundled_eval.sh` derives
   `RUN_DIR="$EXP_DIR/$CONFIG_NAME/$RUN_NUM"`.
   - `submit_zgplev_baseline.slurm` already defines `CONFIG_NAME` (line 26).
   - The h100 production submits define `FULL_TPL` but not `CONFIG_NAME`;
     we'll add the latter without removing the former (used elsewhere by
     the YAML rendering step).

6. **Prelude carve-out cannot use `exit`.** `submit_eval.sh` has `exit 2`
   (line 75, git missing) and `exit 3` (line 162, collision guard). If
   sourced from the training submit, these would terminate the whole SLURM
   job after training has already succeeded. **Convert to function with
   `return` codes**; the bundled path inspects the return and reports
   `EVAL=FAIL` without killing the training job.

7. **Status trap already sends success mail on exit 0.** `slurm_helpers.sh`
   `sfno_install_status_trap` (line ~135) sends mail with the exit code; if
   we silence eval failures with `set +e`, the status mail reads "success"
   even when eval crashed. **Mitigation:** explicit status artifacts
   (`$OUT_ROOT/bundled_eval_status.txt` once the prelude resolves
   `$OUT_ROOT`; always-on fallback `logs/bundled_eval_status_${SLURM_JOB_ID}.txt`
   for the pre-prelude paths — see §4.9) + override the status mail's body
   to include `TRAIN=OK EVAL=<status>`.

8. **Per-submit walltimes vary widely** (see table in §4.5). The "two
   generic bullets" in v1 was wrong; v2 is per-submit.

9. **Partition mix.** Baseline runs on **gh** (GH200), not h100 — eval
   inference is hardcoded `-p h100`, so bundled-eval on baseline would
   either need a GH200 inference path or runs as-is on GH200 (untested).
   **Defer baseline** from first rollout; revisit after GH200 validation.

10. **Collision guard is narrower than v1 claimed.** `submit_eval.sh:145`
    only rejects an existing `OUT_ROOT` when the recorded `CKPT` path
    differs. Same CKPT path updated by a resumed run on the same day → same
    RUN_TAG → stale artifacts mix silently. v2 tightens to `ALLOW_RERUN=1`
    required to reuse any existing `$OUT_ROOT`.

## 4. Design

### 4.1 Run-dir derivation contract

Each affected training submit MUST, before sourcing `bundled_eval.sh`,
export:

```
RUN_NUM="${RUN_NUM:-0}"
CONFIG_NAME="<this submit's config>"   # e.g. plasim_sim52_zgplev_full
RUN_DIR="$EXP_DIR/$CONFIG_NAME/$RUN_NUM"
export RUN_NUM CONFIG_NAME RUN_DIR
```

`bundled_eval.sh` does NOT guess; if `RUN_DIR` is unset it logs an explicit
`SKIP_NO_RUN_DIR` to the always-on fallback log
`logs/bundled_eval_status_${SLURM_JOB_ID}.txt` (the per-`$OUT_ROOT` status
file does not exist yet at this point — see §4.9) and skips eval. This
keeps the helper strict under `set -u` and avoids implicit coupling
between submit scripts.

The existing `RESUME_CKPT_DIR` derivation inside each submit (e.g.
`submit_zgplev_full.slurm:126`) is kept for the resume-detection logic but
also rewritten to use `$RUN_DIR` so there's a single source of truth.

### 4.2 Prelude refactor

Carve `scripts/submit_eval.sh` lines ~38–212 into:

- `scripts/submit_eval_prelude.sh` — defines a single shell function
  `submit_eval_compute_env`. The function:
  - Resolves `RUN_DIR`, `CKPT`, `MODE`, `TEST_HOLDOUT`, `TRAIN_DIR`,
    `PACKAGER_TEST_SRC`, `TRACK`, the 3 SHAs, `RUN_TAG`, `OUT_ROOT`,
    `BENCHMARK_5410_OUT_ROOT`.
  - Writes `$OUT_ROOT/provenance.txt` on success.
  - Returns `0` on success, `2` if a required CLI tool is missing,
    `3` if the collision guard trips (see §4.7).
  - Exports the resolved env on success only.
  - Never calls `exit`.

- `scripts/submit_eval.sh` — sources the prelude, calls
  `submit_eval_compute_env`, exits non-zero if it returns non-zero, and
  otherwise proceeds with the existing `sbatch` chain.

**Intentional behaviour change for standalone path.** The tightened collision
guard (§4.7) changes `submit_eval.sh`'s historical behaviour: a re-run that
previously reused an `$OUT_ROOT` whose recorded `CKPT` path matched will now
abort unless `ALLOW_RERUN=1` is set. This is intentional safety hardening
motivated by the 2026-05-12 v11 EMA / gbhpo40 collision and the
resumed-run-same-CKPT-path hole identified in v2 review. Users of the
standalone path must add `ALLOW_RERUN=1` to any workflow that legitimately
re-evaluates into an existing dir.

### 4.3 Inline stage runners

For each of the 4 stage SLURM scripts
(`submit_eval_inference.slurm`, `submit_eval_score.slurm`,
`submit_eval_report.slurm`, `submit_eval_figures.slurm`), extract the
actual `python …` invocation + env-var defaulting into a sibling
sourceable shell:

```
scripts/eval_run_inference_inline.sh   # NEW — `set -e`, returns non-zero on failure
scripts/eval_run_score_inline.sh       # NEW
scripts/eval_run_report_inline.sh      # NEW
scripts/eval_run_figures_inline.sh     # NEW
```

Each SLURM script remains valid by sourcing its inline counterpart, so the
standalone 4-job chain and the bundled flow run **identical stage bodies**.

### 4.4 Bundled-eval helper

`src/sfno_training/bundled_eval.sh`. **All helper-invoked commands use
`if cmd; then rc=0; else rc=$?; fi` so the parent training script's
`set -e` does NOT short-circuit before `rc` is captured.** A parallel
fallback status file in `logs/bundled_eval_status_${SLURM_JOB_ID}.txt`
captures early-skip / pre-prelude failures, where `$OUT_ROOT` does not
yet exist.

```
_bundled_log_fallback() {
    # Append to a logs/-rooted file so early-skip + prelude-fail paths
    # leave a discoverable artefact even when $OUT_ROOT is unknown.
    local f="$REPO_ROOT/logs/bundled_eval_status_${SLURM_JOB_ID:-noslurm}.txt"
    echo "$1" >> "$f"
}

bundled_eval_maybe_run() {
    local ema ema_mtime rc
    _bundled_log_fallback "ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID:-noslurm}"

    if [ "${BUNDLED_EVAL:-0}" != "1" ]; then
        echo "[bundled-eval] BUNDLED_EVAL=${BUNDLED_EVAL:-0} — skip"
        BUNDLED_EVAL_STATUS="SKIP_DISABLED"
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS"
        return 0
    fi

    if [ -z "${RUN_DIR:-}" ]; then
        echo "[bundled-eval] RUN_DIR unset (submit script must export it) — skip"
        BUNDLED_EVAL_STATUS="SKIP_NO_RUN_DIR"
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS"
        return 0
    fi

    ema="$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar"
    ema_mtime=$(stat -c %Y "$ema" 2>/dev/null || echo 0)
    if [ "$ema_mtime" -le "${JOB_START_EPOCH:-0}" ]; then
        echo "[bundled-eval] EMA ckpt not refreshed this run — skip"
        BUNDLED_EVAL_STATUS="SKIP_NO_NEW_CKPT"
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS ema=$ema mtime=$ema_mtime job_start=${JOB_START_EPOCH:-0}"
        return 0
    fi

    # Errexit-safe: capture rc, do not let set -e abort us mid-helper.
    source "$REPO_ROOT/scripts/submit_eval_prelude.sh"
    if submit_eval_compute_env; then rc=0; else rc=$?; fi
    if [ "$rc" -ne 0 ]; then
        echo "[bundled-eval] submit_eval_compute_env returned $rc — skip"
        BUNDLED_EVAL_STATUS="FAIL_PRELUDE_$rc"
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS"
        return 0
    fi

    # From here, $OUT_ROOT exists and the per-OUT_ROOT status file is canonical.
    local status_file="$OUT_ROOT/bundled_eval_status.txt"
    : > "$status_file"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
    echo "slurm_job_id=${SLURM_JOB_ID:-noslurm}" >> "$status_file"
    _bundled_log_fallback "status=STARTED out_root=$OUT_ROOT"

    # Each call wrapped to survive parent set -e. Gate downstream stages.
    if bash "$REPO_ROOT/scripts/eval_run_inference_inline.sh"; then rc=0; else rc=$?; fi
    echo "inference_rc=$rc" >> "$status_file"
    if [ "$rc" -ne 0 ]; then
        BUNDLED_EVAL_STATUS="FAIL_INFERENCE"
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS out_root=$OUT_ROOT"
        return 0
    fi

    if bash "$REPO_ROOT/scripts/eval_run_score_inline.sh"; then rc=0; else rc=$?; fi
    echo "score_rc=$rc" >> "$status_file"
    if [ "$rc" -ne 0 ]; then
        BUNDLED_EVAL_STATUS="FAIL_SCORE"
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS out_root=$OUT_ROOT"
        return 0
    fi

    if bash "$REPO_ROOT/scripts/eval_run_report_inline.sh"; then rc=0; else rc=$?; fi
    echo "report_rc=$rc" >> "$status_file"
    if [ "$rc" -ne 0 ]; then
        BUNDLED_EVAL_STATUS="FAIL_REPORT"
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS out_root=$OUT_ROOT"
        return 0
    fi

    if bash "$REPO_ROOT/scripts/eval_run_figures_inline.sh"; then rc=0; else rc=$?; fi
    echo "figures_rc=$rc" >> "$status_file"
    if [ "$rc" -ne 0 ]; then
        BUNDLED_EVAL_STATUS="FAIL_FIGURES"
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS out_root=$OUT_ROOT"
        return 0
    fi

    echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
    BUNDLED_EVAL_STATUS="OK"
    _bundled_log_fallback "status=OK out_root=$OUT_ROOT"
    return 0
}
```

Key points:
- The helper **never exits non-zero** on eval failure — training success
  must remain visible.
- Every call site is errexit-safe via `if cmd; then rc=0; else rc=$?; fi`,
  so a stage failure under `set -e` cannot abort the helper before `rc` is
  captured.
- Early-skip / pre-prelude paths write to
  `logs/bundled_eval_status_${SLURM_JOB_ID}.txt`. Once `$OUT_ROOT` is
  resolved, the per-OUT_ROOT file becomes canonical, but the fallback file
  continues to receive one-line breadcrumbs for grep-ability.
- `BUNDLED_EVAL_STATUS` is read by the status-mail wrapper (§4.6).

### 4.5 Per-submit walltime table

Eval tail budget (NWP only, K=56, 96 ICs) — **calibrated to user-observed
NWP-only times, 2026-05-20**:
- Inference (NWP only, ~5,376 forward passes vs the docstring's
  17,016 including climate): **~30 min** (the docstring's ~1.5 h estimate
  covers NWP+climate together)
- Score (climatology build + scoring): **~35–40 min** — climatology build
  is the I/O-bound portion (`submit_eval_score.slurm:5,18`), stored under
  each fresh `$OUT_ROOT` with no cache. Because §4.7's `ALLOW_RERUN=1`
  policy makes every bundled run land in a fresh `$OUT_ROOT`, **every
  bundled eval pays this climatology cost**.
- Report + figures: ~5–7 min
- **Total: ~1 h 10 min. Budget 1 h 15 min for headroom.**

| Submit | Partition | Current `-t` | Required `-t` | Notes |
|---|---|---|---|---|
| submit_zgplev_full.slurm | h100 | 06:00:00 | **07:15:00** | small warm-start chunk; +1h15m tail |
| submit_zgplev_group_clone.slurm | h100 | 17:00:00 | **18:15:00** | v10 production line |
| submit_zgplev_group_clone_v11.slurm | h100 | 18:30:00 | **19:45:00** | |
| submit_zgplev_group_clone_v11_clip.slurm | h100 | 18:30:00 | **19:45:00** | |
| submit_zgplev_group_clone_v10_warmstart.slurm | h100 | 18:30:00 | **19:45:00** | warm-start chunk; eval only on final chunk |
| submit_zgplev_group_clone_v11_clip_warmstart.slurm | h100 | 18:30:00 | **19:45:00** | warm-start chunk; eval only on final chunk |
| submit_zgplev_baseline.slurm | **gh** (GH200) | 24:00:00 | **DEFERRED** | excluded from first rollout (§4.8) |

**Partition cap.** The reviewer confirmed via `sinfo` that the h100
partition timelimit is reported as **infinite**, not 24h. All proposed
values are therefore safely under any operative cap. (Operational
guidance: TACC may still enforce a softer queue policy at scheduler level;
21:30:00 is well-behaved and matches existing 18:30:00 submits' shape.)

**Future optimisation (deferred, not in this plan):** climatology could be
materialised once into a shared cache (e.g.
`$WORK2/SFNO_Climate_Emulator/results/sfno_eval/_climatology_cache/<TRAIN_DIR_SHA>.nc`)
and symlinked into each `$OUT_ROOT/baselines/`. That would cut ~30–60 min
off every bundled eval. Out of scope here because it touches
`compute_climatology.py` semantics and warrants its own review.

### 4.6 Order of operations + status-mail integration

```
JOB_START_EPOCH=$(date +%s)         # capture BEFORE training
export JOB_START_EPOCH
CONFIG_NAME="<config>"               # e.g. plasim_sim52_zgplev_full
RUN_NUM="${RUN_NUM:-0}"
RUN_DIR="$EXP_DIR/$CONFIG_NAME/$RUN_NUM"
export CONFIG_NAME RUN_NUM RUN_DIR

source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap                  # existing; sends mail on exit
... preflight, training, scan_for_nans (unchanged) ...
TRAIN_RC=$?
export TRAIN_RC

# Bundled eval. Sets BUNDLED_EVAL_STATUS but does NOT exit non-zero on eval failure.
source "$REPO_ROOT/src/sfno_training/bundled_eval.sh"
bundled_eval_maybe_run

# Compose a final status line in the mail body. Patch slurm_helpers.sh so
# sfno_send_status_mail's body includes both TRAIN_RC and BUNDLED_EVAL_STATUS,
# e.g. "TRAIN=OK EVAL=FAIL_INFERENCE (see $OUT_ROOT/bundled_eval_status.txt)".
```

The status trap continues to send `END` mail with the script's overall exit
code. We override the body via a `sfno_status_body_extra` env var that
includes the bundled-eval verdict so a failed eval is loud in the mail
even when the SLURM job itself exited 0.

### 4.7 Tightened collision guard

In `submit_eval_compute_env`:

```
if [ -e "$OUT_ROOT" ]; then
    if [ "${ALLOW_RERUN:-0}" != "1" ]; then
        echo "[submit_eval_prelude] FATAL: $OUT_ROOT already exists." >&2
        echo "Set ALLOW_RERUN=1 to overwrite (and move/keep the prior dir yourself if you need to)." >&2
        return 3
    else
        echo "[submit_eval_prelude] ALLOW_RERUN=1 — proceeding into existing $OUT_ROOT" >&2
    fi
fi
```

This is strictly stronger than the existing CKPT-path-only check
(`submit_eval.sh:145–164`). It also closes the resumed-run + same-CKPT-path
hole the v1 plan glossed over. The existing CKPT-path mismatch branch is
preserved as additional defensive logging inside the `ALLOW_RERUN=1` path.

Note that `submit_eval_prelude` writes `provenance.txt` AFTER this check,
so the guard sees the directory but only relies on its presence.

### 4.8 Per-submit BUNDLED_EVAL defaults

| Submit | Top-of-script line | Effect |
|---|---|---|
| submit_zgplev_full.slurm | `: "${BUNDLED_EVAL:=1}"` | Bundled eval ON by default (production) |
| submit_zgplev_group_clone.slurm | `: "${BUNDLED_EVAL:=1}"` | ON (production v10) |
| submit_zgplev_group_clone_v11.slurm | `: "${BUNDLED_EVAL:=1}"` | ON (production v11) |
| submit_zgplev_group_clone_v11_clip.slurm | `: "${BUNDLED_EVAL:=1}"` | ON (production v11_clip) |
| submit_zgplev_group_clone_v10_warmstart.slurm | `: "${BUNDLED_EVAL:=0}"` | OFF by default — user sets `BUNDLED_EVAL=1` for final chunk only |
| submit_zgplev_group_clone_v11_clip_warmstart.slurm | `: "${BUNDLED_EVAL:=0}"` | OFF by default — final chunk only |
| submit_zgplev_baseline.slurm | (unchanged) | DEFERRED until GH200 eval-inference validated |
| All HPO / smoke / short / tiny submits | (unchanged, helper not sourced) | Not touched |

A "loud" log line near the top of every affected submit prints the
resolved `BUNDLED_EVAL` value so it's visible in `*.out`, e.g.:

```
echo "[bundled-eval] BUNDLED_EVAL=$BUNDLED_EVAL (set to 1 to enable, 0 to disable)"
```

This addresses the v1 contradiction (Codex Medium finding 2): production
submits opt **in** by default; warm-start chunks opt **out** by default;
sweep / smoke submits are untouched. There is no longer a "default behaviour
unchanged" claim — production submits' default behaviour changes intentionally.

### 4.9 Failure-reporting summary

Artefacts:

1. `$OUT_ROOT/bundled_eval_status.txt` — machine-readable per-stage rc
   table. Present whenever the prelude succeeded (i.e. `$OUT_ROOT` was
   established).
2. `logs/bundled_eval_status_${SLURM_JOB_ID}.txt` — fallback / breadcrumb
   file. Present for **every** bundled run, including the early-skip and
   prelude-failure paths where `$OUT_ROOT` does not exist. Contents are
   one short line per state transition.
3. `$OUT_ROOT/provenance.txt` — written by the prelude on success.
4. Status mail body — augmented with `TRAIN=… EVAL=…` line.

Three resume paths if eval fails halfway:

1. Score onward: `RUN_TAG=<existing> ALLOW_RERUN=1 sbatch scripts/submit_eval_score.slurm`
   (and downstream).
2. Re-run inference only: same with `submit_eval_inference.slurm`.
3. Full re-chain: `ALLOW_RERUN=1 RUN_TAG=<existing> scripts/submit_eval.sh`.

## 5. Risks and trade-offs

1. **h100 SUs spent on CPU eval stages (~45–75 min, dominated by the
   ~30–60 min climatology build).** Acceptable given queue-wait savings
   (potentially 12+ hours), but more material than the v1 plan's "~12 min"
   estimate suggested. A shared climatology cache (§4.5 future
   optimisation) would cut this in half.

2. **Inference walltime variance.** Mitigated by the per-submit walltime
   bumps in §4.5. If a run wall-times out during eval, training is still
   safely checkpointed and the user can run the standalone 4-job chain.

3. **Operator must remember to set `BUNDLED_EVAL=1` on the final warm-start
   chunk.** Mitigation: loud log line + the warm-start submits' top-of-file
   comment block documents this.

4. **Eval failure visibility.** Mitigated by the per-stage status artifacts
   defined in §4.9 (`$OUT_ROOT/bundled_eval_status.txt` once `$OUT_ROOT`
   is established; always-on `logs/bundled_eval_status_${SLURM_JOB_ID}.txt`
   for pre-prelude paths) + status-mail body augmentation. Training
   success remains the SLURM exit code; eval failure is a parallel signal.

5. **Baseline / GH200 deferred.** If we discover we want baseline bundled
   too, follow-up plan: validate `submit_eval_inference.slurm` body runs
   correctly on GH200, then enable baseline.

## 6. Affected files (planned diff surface)

NEW:
- `scripts/submit_eval_prelude.sh`
- `scripts/eval_run_inference_inline.sh`
- `scripts/eval_run_score_inline.sh`
- `scripts/eval_run_report_inline.sh`
- `scripts/eval_run_figures_inline.sh`
- `src/sfno_training/bundled_eval.sh`

MODIFIED:
- `scripts/submit_eval.sh` — sources prelude function; preserves existing CLI
- `scripts/submit_eval_inference.slurm` — sources `eval_run_inference_inline.sh`
- `scripts/submit_eval_score.slurm` — sources `eval_run_score_inline.sh`
- `scripts/submit_eval_report.slurm` — sources `eval_run_report_inline.sh`
- `scripts/submit_eval_figures.slurm` — sources `eval_run_figures_inline.sh`
- `src/sfno_training/slurm_helpers.sh` — status-mail body extras for `BUNDLED_EVAL_STATUS`
- `src/sfno_training/submit_zgplev_full.slurm` — walltime + `CONFIG_NAME` + bundled hook
- `src/sfno_training/submit_zgplev_group_clone.slurm` — same
- `src/sfno_training/submit_zgplev_group_clone_v11.slurm` — same
- `src/sfno_training/submit_zgplev_group_clone_v11_clip.slurm` — same
- `src/sfno_training/submit_zgplev_group_clone_v10_warmstart.slurm` — same, `BUNDLED_EVAL` default 0
- `src/sfno_training/submit_zgplev_group_clone_v11_clip_warmstart.slurm` — same, `BUNDLED_EVAL` default 0

NOT TOUCHED in first rollout:
- `src/sfno_training/submit_zgplev_baseline.slurm` (gh / GH200)
- All HPO sweep submits (`submit_zgplev_gbhpo40_*`, `submit_zgplev_group_clone_v11_gb32_*`)
- All smoke / short / tiny submits

## 7. Closed questions (from Codex review)

- **Q1: Should bundled eval be allowed to fail the training SLURM job?**
  **No.** Training success remains visible via SLURM exit code; eval status
  is reported via the two-tier status artifacts in §4.9
  (`$OUT_ROOT/bundled_eval_status.txt` once `$OUT_ROOT` is established;
  always-on `logs/bundled_eval_status_${SLURM_JOB_ID}.txt`) and the
  status-mail body suffix (`TRAIN=OK EVAL=FAIL_<stage>`). See §4.6, §4.9.

- **Q2: Should repeated evals of the same CKPT ever reuse an OUT_ROOT?**
  **No, default refuse.** `ALLOW_RERUN=1` required to reuse, regardless of
  whether the recorded CKPT path matches. See §4.7.

## 8. Open questions remaining for review

1. ~~h100 queue walltime cap~~ — **resolved in v3**: reviewer confirmed
   `sinfo` reports the h100 timelimit as infinite. All proposed values
   (max 21:30:00) are safely under any operative cap.

2. **5410 benchmark overlay** (`BENCHMARK_5410_OUT_ROOT` default in
   `submit_eval.sh:208`). Assume bundled flow uses the same default — i.e.
   the prelude function exports it identically. OK to confirm.

3. **eval-sfno-own skill update.** The skill prose describes the 4-job
   chain explicitly. We'll need a "bundled eval" subsection so future
   invocations know which path produced a given `$OUT_ROOT` (and how to
   resume from `bundled_eval_status.txt` + the
   `logs/bundled_eval_status_*` fallback file).

## 9. Rollout plan

1. Refactor `submit_eval.sh` → prelude function + inline runners. Sanity
   check: standalone `scripts/submit_eval.sh` flow with `ALLOW_RERUN=1`
   produces an identical `provenance.txt` and the 4-job chain still works.
   Update any existing personal workflow that relied on
   same-CKPT-implicit-reuse to add `ALLOW_RERUN=1`.
2. Add `bundled_eval.sh` + status-mail body augmentation. Unit-test the
   helper's skip paths (`SKIP_DISABLED`, `SKIP_NO_RUN_DIR`,
   `SKIP_NO_NEW_CKPT`, `FAIL_PRELUDE_3`). Verify the
   `logs/bundled_eval_status_*` fallback file is written for each.
3. Wire into **one** submit first: `submit_zgplev_full.slurm`. Run a small
   warm-start chunk with `BUNDLED_EVAL=1` and verify all 4 stages complete
   on the same allocation.
4. **Smoke comparison** against the standalone 4-job chain. Per §4.7 the
   bundled and standalone runs MUST use **separate `RUN_TAG`s** (or the
   second invocation must move/clear the first `$OUT_ROOT`). Suggested
   approach: pass `RUN_TAG=<auto>_bundled` to the bundled run and
   `RUN_TAG=<auto>_standalone` to the comparison standalone run, then
   diff `report.md` and the scorecard CSV between the two `$OUT_ROOT`s.
5. Roll out to the remaining 4 h100 production submits.
6. Update the eval-sfno-own skill prose.
7. (Future) Validate eval inference on GH200 and enable baseline.
8. (Future, optional) Shared climatology cache to remove ~30–60 min per
   bundled eval (deferred — §4.5).
