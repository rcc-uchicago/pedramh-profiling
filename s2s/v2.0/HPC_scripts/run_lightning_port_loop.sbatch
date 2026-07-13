#!/bin/bash
# Autonomous driver for the S2S -> PyTorch Lightning port (SNFO-mirrored).
# Pure orchestrator on --partition=build (RCC, has egress for the headless `claude` CLI): runs the loop + the
# light agent gauntlet + git, and submits ALL gate-deciding GPU smoke tests as NESTED sbatch jobs to
# --partition=pedramh-gpu (--account=pi-pedramh). The compute nodes are offline (pre-built env, WANDB_MODE=offline,
# no pip at job time), so Claude must NOT run there — it orchestrates from here and reads the smoke .out/.err back.
# Survives build's 6h QOS MaxWall by chaining a fresh job ~10 min before the kill; resumes losslessly because the
# driver commits after every green phase and re-orients from git state (prompt §1).
#
# Drives prompts/_cluster_autonomous_lightning_port_loop.md, whose §10 emits the sentinels
# STAGE_LANDED | PORT_COMPLETE | BLOCKED. The greps below MUST match those names.
# Each phase is delegated to the .claude/agents/lightning-porter.md subagent via the Agent tool.
#
# Submit from the repo root:  mkdir -p logs && sbatch v2.0/HPC_scripts/run_lightning_port_loop.sbatch
# Stop:  scancel <jobid>  (also scancel any queued chained job; rm _scratch/.lport_started for a fresh start)
# Right-size after run 1:  seff <jobid>  /  sacct -j <jobid> -o JobID,MaxRSS,Elapsed
#
#SBATCH --job-name=s2s-lightning-port
#SBATCH --output=logs/s2s-lightning-port-%j.log
#SBATCH --open-mode=append
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00                # 6h = build's QOS MaxWall; USR1 chain spans longer runs.
#SBATCH --signal=B:USR1@600            # SIGUSR1 to the batch shell ~10 min before the kill

set -uo pipefail
set -m                                 # monitor mode: each `cmd &` gets its OWN process group, so the trap
                                       # can kill claude's whole tree (CLI + git + Agent children)

REPO="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && git rev-parse --show-toplevel)"
SELF="$REPO/v2.0/HPC_scripts/run_lightning_port_loop.sbatch"   # stable path — NOT $0 (Slurm runs a spooled copy)
cd "$REPO"
mkdir -p logs _scratch

# Branch guard: land the port on the lightning-port branch, NEVER main / bench-instrumentation. The autonomous
# agent commits to HEAD and the sbatch may be launched from any branch — force it up front and FAIL LOUD.
BASELINE_BRANCH="lightning-port"
if [[ "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" != "$BASELINE_BRANCH" ]]; then
  git checkout "$BASELINE_BRANCH" 2>/dev/null || git checkout -b "$BASELINE_BRANCH" 2>/dev/null || {
    echo "[FATAL] $(date) cannot switch to $BASELINE_BRANCH (dirty tree?) — refusing to run on $(git rev-parse --abbrev-ref HEAD 2>/dev/null) to avoid committing to the wrong branch." >&2; exit 1; }
  echo "[branch] $(date) switched to $BASELINE_BRANCH"
fi
FF_FILE="$REPO/_scratch/.lport_fastfail"
fastfail=$(cat "$FF_FILE" 2>/dev/null || echo 0); [[ "$fastfail" =~ ^[0-9]+$ ]] || fastfail=0  # load EARLY so the USR1 trap sees a fastfail storm even during env setup

# --- heartbeat: every ~30 min, tail the latest per-iteration claude log into --output so a long no-commit
#     iteration (e.g. a GPU smoke in the queue) isn't mistaken for a stall ---
HEARTBEAT_PID=""
heartbeat() {
  while :; do
    sleep 1800
    local latest; latest=$(ls -t logs/lport-iter-*.log 2>/dev/null | head -1)
    if [[ -n "$latest" ]]; then
      echo "[heartbeat] $(date) — tail of $latest:"; tail -n 25 "$latest" 2>/dev/null || true
    else
      echo "[heartbeat] $(date) — no iter log yet (between iterations / nested smoke in flight)"
    fi
  done
}
stop_heartbeat() { [[ -n "${HEARTBEAT_PID:-}" ]] && kill "$HEARTBEAT_PID" 2>/dev/null || true; HEARTBEAT_PID=""; }

# --- helpers (defined BEFORE the trap can fire) ---
stop_child() { stop_heartbeat; [[ -n "${CLAUDE_PID:-}" ]] || return 0; kill -TERM -"$CLAUDE_PID" 2>/dev/null || true; wait "$CLAUDE_PID" 2>/dev/null || true; }
resubmit_self() {
  mkdir -p "$REPO/logs"; local rc; sbatch "$SELF"; rc=$?
  if (( rc == 0 )); then echo "[chain] $(date) resubmitted OK"
  else echo "[FATAL] $(date) sbatch resubmit FAILED rc=$rc — chain BROKEN (check partition/QOS time limits)." >&2; fi
}
chain_on_signal() {
  trap '' USR1
  echo "[chain] $(date) USR1 received — stopping child before resubmit"
  stop_child
  if (( ${fastfail:-0} >= 5 )); then echo "[stop] $(date) fastfail storm — NOT resubmitting"; exit 1; fi
  resubmit_self; exit 0
}
trap chain_on_signal USR1

# Orchestrator env: this node only runs the `claude` CLI + git + the Agent gauntlet. It does NOT need the torch
# stack (the nested GPU smokes activate their own env, §4 of the prompt). It DOES need miniforge so Phase 0 can
# provision the Lightning env on /project (build has internet; compute nodes do not).
# TODO(operator): add any site module loads your build node needs for git/python here.
module load python/miniforge-25.3.0 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"          # keep the node-based `claude` CLI on PATH
# SNFO template checkout the porter mirrors — exported so the porter (and the preflight) resolve it. Cluster
# default is the shared copy; if SNFO ever moves, change ONLY this line (or set SNFO_DIR in the environment).
export SNFO_DIR="${SNFO_DIR:-/project/pedramh/shared/anthonyz}"
command -v claude >/dev/null 2>&1 || { echo "[FATAL] $(date) 'claude' not on PATH (expected ~/.local/bin) — cannot run headless loop." >&2; exit 1; }
command -v git    >/dev/null 2>&1 || { echo "[FATAL] $(date) 'git' not on PATH." >&2; exit 1; }

heartbeat & HEARTBEAT_PID=$!
echo "[heartbeat] $(date) — started (pid $HEARTBEAT_PID), tailing latest logs/lport-iter-*.log every ~30 min"

PROMPT="prompts/_cluster_autonomous_lightning_port_loop.md"

# --- preflight: delegate to the committed test — (A) required files present, (B) the commit-on-pass /
#     revert-on-fail guardrail behaves, (D) prompt<->sbatch sentinels consistent. FATAL if it fails, rather
#     than failing mid-loop. Run it by hand any time: bash v2.0/HPC_scripts/preflight_test.sh ---
if ! CLAUDE_PROJECT_DIR="$REPO" bash "$REPO/v2.0/HPC_scripts/preflight_test.sh"; then
  echo "[FATAL] $(date) preflight_test.sh failed (see [FAIL] above) — fix it (scp missing files / register the hook) before running." >&2
  stop_heartbeat; exit 1
fi

# Loop control. Health is judged by PROGRESS (a new git commit), not by exit code or wall-clock. Counters persist
# across chained jobs, reset on any progress/success:
#   FF_FILE = consecutive failing-no-progress iterations (auth/env)  -> BLOCK at 5
#   TH_FILE = consecutive API throttles with no progress             -> BLOCK at MAX_THROTTLE
#   NP_FILE = consecutive iterations with no new commit              -> BLOCK at MAX_NOPROGRESS
MAX_ITERS=200
MAX_THROTTLE=12
MAX_NOPROGRESS=25
TH_FILE="$REPO/_scratch/.lport_throttle"
NP_FILE="$REPO/_scratch/.lport_noprogress"
fastfail=$(cat "$FF_FILE" 2>/dev/null || echo 0);   [[ "$fastfail" =~ ^[0-9]+$ ]]   || fastfail=0
throttle=$(cat "$TH_FILE" 2>/dev/null || echo 0);   [[ "$throttle" =~ ^[0-9]+$ ]]   || throttle=0
noprogress=$(cat "$NP_FILE" 2>/dev/null || echo 0); [[ "$noprogress" =~ ^[0-9]+$ ]] || noprogress=0
reset_breakers(){ fastfail=0; throttle=0; echo 0 > "$FF_FILE"; echo 0 > "$TH_FILE"; }
is_throttle(){ grep -qiE 'temporarily limiting requests|rate.?limit(ed|ing)?|overloaded_error|"type" *: *"(overloaded|rate_limit)|too many requests|(status|http|code)[^0-9]{0,6}(429|529)\b' "$1"; }

# Nested-job registry: the porter appends each smoke's id (`sbatch --parsable`) to this file (prompt §4), so the
# orchestrator waits on EXACTLY the jobs THIS loop launched — name/folder/--chdir-agnostic — and an unrelated job
# from your account never stalls it. nested_jobs() returns the in-flight (RUNNING/PENDING) registry ids, prunes
# finished ones, and FALLS BACK to a folder scope (jobs whose %Z is under $REPO, minus this orchestrator) when the
# registry is empty (e.g. the porter has not recorded an id yet).
NESTED_FILE="$REPO/_scratch/.lport_nested_jobs"
nested_jobs() {
  local mine keep="" j csv
  mine=" $(squeue --me -h -o '%i' 2>/dev/null | tr '\n' ' ') "          # my still-listed jobs (any state)
  if [[ -s "$NESTED_FILE" ]]; then
    while read -r j; do [[ "$j" =~ ^[0-9]+$ && "$mine" == *" $j "* ]] && keep+="$j "; done < "$NESTED_FILE"
    if [[ -n "${keep// /}" ]]; then
      printf '%s\n' $keep > "$NESTED_FILE"                              # prune finished ids out of the registry
      csv=$(echo $keep | tr ' ' ',')
      squeue -h -t RUNNING,PENDING -o '%i' -j "$csv" 2>/dev/null | tr '\n' ' '
      return
    fi
    : > "$NESTED_FILE"                                                  # all recorded ids finished → clear, fall back
  fi
  squeue --me -h -t RUNNING,PENDING -o '%i|%j|%Z' 2>/dev/null \
    | awk -F'|' -v self="${SLURM_JOB_NAME:-s2s-lightning-port}" -v repo="$REPO" \
          '$2 != self && ($3 == repo || index($3, repo "/") == 1) { print $1 }' \
    | tr '\n' ' '
}

for ((i=1; i<=MAX_ITERS; i++)); do
  # If a NESTED GPU smoke (a job THIS loop submitted) is still in flight, WAIT before spending a claude iteration
  # (a queued smoke makes no commits; re-invoking claude to poll would waste API calls AND trip the no-progress
  # guard on a healthy wait). nested_jobs() returns only jobs THIS loop launched — the porter's recorded ids, or a
  # folder-scoped fallback — so an UNRELATED job from your account never makes the loop wait on it. Cap the wait.
  wait_start=$SECONDS
  while [[ -n "${SLURM_JOB_ID:-}" ]]; do
    nested=$(nested_jobs)
    [[ -z "${nested// /}" ]] && break
    if (( SECONDS - wait_start > 5400 )); then
      echo "[wait] $(date) nested smoke(s) [$nested] still in flight after 1.5h — proceeding to a diagnosing iteration, NOT spinning" >&2; break
    fi
    echo "[wait] $(date) nested GPU smoke in flight: $nested — polling every 120s (not a stall)"
    sleep 120
  done

  LOG="logs/lport-iter-${SLURM_JOB_ID:-local}-$i.log"
  head_before=$(git rev-parse HEAD 2>/dev/null || echo none)
  if [[ -f _scratch/.lport_started ]]; then
    claude -p --continue "Continue the loop per $PROMPT. Orient from git state (§1); resume the current phase." \
      --model claude-opus-4-8 --effort max --dangerously-skip-permissions --max-turns 400 > "$LOG" 2>&1 &
  else
    touch _scratch/.lport_started
    claude -p "$(cat "$PROMPT")" \
      --model claude-opus-4-8 --effort max --dangerously-skip-permissions --max-turns 400 > "$LOG" 2>&1 &
  fi
  CLAUDE_PID=$!; wait "$CLAUDE_PID"; rc=$?

  # Anchored sentinels scanned over the WHOLE log (must match prompt §10).
  if grep -qE '^=== LOOP: PORT_COMPLETE ===$' "$LOG"; then echo "[done] $(date) S2S Lightning port complete"; stop_heartbeat; rm -f "$FF_FILE" "$TH_FILE"; exit 0; fi
  if grep -qE '^=== LOOP: BLOCKED ===$' "$LOG";       then echo "[stop] $(date) driver signalled BLOCKED"; stop_heartbeat; exit 1; fi
  if grep -qE '^=== LOOP: STAGE_LANDED ===$' "$LOG"; then
    echo "[stage] $(date) STAGE_LANDED — fresh session for the next phase"
    rm -f _scratch/.lport_started; reset_breakers; noprogress=0; echo 0 > "$NP_FILE"; continue
  fi

  head_after=$(git rev-parse HEAD 2>/dev/null || echo none)
  if [[ "$head_after" != "$head_before" && "$head_after" != none ]]; then
    reset_breakers; noprogress=0; echo 0 > "$NP_FILE"; continue
  fi
  noprogress=$((noprogress+1)); echo "$noprogress" > "$NP_FILE"
  if (( noprogress >= MAX_NOPROGRESS )); then
    echo "[stop] $(date) $noprogress iterations: no commit AND no nested smoke in flight — genuinely stalled. BLOCKED" >&2; stop_heartbeat; exit 1
  fi
  if (( rc == 0 )); then reset_breakers; continue; fi
  if is_throttle "$LOG"; then
    throttle=$((throttle+1)); echo "$throttle" > "$TH_FILE"
    if (( throttle > MAX_THROTTLE )); then
      echo "[stop] $(date) API throttled ${throttle}x with no progress — API down or request malformed. BLOCKED" >&2; stop_heartbeat; exit 1
    fi
    wait_s=$(( 60 * throttle )); (( wait_s > 300 )) && wait_s=300
    echo "[throttle] $(date) iter $i rate-limited (retry $throttle/$MAX_THROTTLE) — sleeping ${wait_s}s"
    sleep "$wait_s"; continue
  fi
  fastfail=$((fastfail+1)); echo "$fastfail" > "$FF_FILE"
  echo "[warn] $(date) iter $i failed (rc=$rc) with no git progress — fastfail=$fastfail/5" >&2
  if (( fastfail >= 5 )); then
    echo "[stop] $(date) 5 failing iterations with no progress — NOT resubmitting (check auth/env). BLOCKED" >&2; stop_heartbeat; exit 1
  fi
done
echo "[chain] $(date) iteration cap ($MAX_ITERS) reached, work unfinished — refreshing process"
trap '' USR1
stop_heartbeat
resubmit_self
exit 0
