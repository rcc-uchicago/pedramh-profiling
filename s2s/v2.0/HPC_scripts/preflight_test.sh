#!/usr/bin/env bash
# Preflight test for the autonomous Lightning-port loop. The sbatch runs this before the first `claude` call;
# you can also run it by hand:  bash v2.0/HPC_scripts/preflight_test.sh
#
# It gates the run on three checks (exit 0 only if all pass; non-zero + a [FAIL ...] line otherwise):
#   (A) FILES — every prompt/agent/hook/settings asset the loop spawns is present on THIS node. The .claude/*
#       agents + hooks are gitignored and ship via scp; a missing one silently breaks the run (a gauntlet
#       subagent that does not exist, an unenforced hook). Warns (does not fail) if the block-dangerous
#       PreToolUse hook is not registered in settings — that is expected on a dev box, required on the cluster.
#   (B) GUARDRAIL — the loop's commit discipline behaves: a PASSING phase commits and advances the last-green
#       floor; a FAILING phase reverts the WIP back to the last-green floor (git reset --hard, no commit leaks).
#       Exercised in an ISOLATED temp repo via mktemp — it never touches the real working tree.
#   (D) SENTINELS — the terminal sentinels are consistent between the driver prompt (which emits them) and the
#       sbatch (which greps them); drift here would strand the loop.
set -uo pipefail

REPO="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[[ -n "$REPO" && -d "$REPO" ]] || { echo "[FAIL] cannot resolve repo root — set CLAUDE_PROJECT_DIR or run inside the repo" >&2; exit 1; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

# ---------- (A) required files ----------
REQUIRED=(
  "prompts/_cluster_autonomous_lightning_port_loop.md"
  "v2.0/HPC_scripts/run_lightning_port_loop.sbatch"
  ".claude/agents/lightning-porter.md"
  ".claude/agents/drift-auditor.md"
  ".claude/agents/s2s-code-reviewer.md"
  ".claude/agents/s2s-code-reviewer-critic.md"
  ".claude/tools/_hook_block_dangerous_cmds.sh"
  ".claude/tools/_hook_check_s2s_invariants.sh"
  ".claude/tools/check_s2s_invariants.sh"
  ".claude/settings.local.json"
)
miss=(); for f in "${REQUIRED[@]}"; do [[ -r "$REPO/$f" ]] || miss+=("$f"); done
if (( ${#miss[@]} )); then
  printf '[FAIL] (A) %d required file(s) missing on %s — scp them (see manifest):\n' "${#miss[@]}" "$(hostname)" >&2
  printf '   MISSING: %s\n' "${miss[@]}" >&2
  exit 1
fi
echo "[OK]   (A) all ${#REQUIRED[@]} required prompt/agent/hook/settings files present"
grep -q "_hook_block_dangerous_cmds.sh" "$REPO/.claude/settings.local.json" 2>/dev/null \
  || echo "[WARN] (A) settings.local.json does not register the block-dangerous PreToolUse hook — guardrail backstop INACTIVE (expected on a dev box; required on the cluster)" >&2

# ---------- (C) SI template checkout present (the porter mirrors it) ----------
SI_DIR="${SI_DIR:-/project/pedramh/shared/anthonyz}"
[[ -d "$SI_DIR" ]] || fail "(C) SI template dir not found: SI_DIR=$SI_DIR — set SI_DIR or fix the path (the porter mirrors this checkout)"
si_miss=()
for f in train.py modules/train_module.py data/datamodule.py common/bench_callback.py environment.yml; do
  [[ -r "$SI_DIR/$f" ]] || si_miss+=("$f")
done
if (( ${#si_miss[@]} )); then
  printf '[FAIL] (C) SI_DIR=%s is missing template file(s):\n' "$SI_DIR" >&2
  printf '   MISSING: %s\n' "${si_miss[@]}" >&2
  exit 1
fi
echo "[OK]   (C) SI template present at $SI_DIR"

# ---------- (B) commit-on-pass / revert-on-fail guardrail, in an ISOLATED temp repo ----------
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" config user.email preflight@test; git -C "$tmp" config user.name preflight
printf 'base\n' > "$tmp/work"; git -C "$tmp" add -A; git -C "$tmp" commit -qm "green-floor-0"
green="$(git -C "$tmp" rev-parse HEAD)"

# B1 — a PASSING phase commits and advances the last-green floor.
printf 'phase-1 change\n' >> "$tmp/work"; smoke_ok=true
$smoke_ok && { git -C "$tmp" add -A; git -C "$tmp" commit -qm "phase-1 [green]"; }
[[ "$(git -C "$tmp" rev-list --count HEAD)" -eq 2 ]] || fail "(B) commit-on-pass: a passing phase produced no commit"
green="$(git -C "$tmp" rev-parse HEAD)"

# B2 — a FAILING phase reverts the WIP to the last-green floor: tree clean, HEAD unchanged, WIP gone, no commit.
printf 'broken WIP that fails its smoke\n' >> "$tmp/work"; smoke_ok=false
$smoke_ok || git -C "$tmp" reset --hard -q "$green"
[[ -z "$(git -C "$tmp" status --porcelain)" ]]      || fail "(B) revert-on-fail: working tree not clean after reset"
[[ "$(git -C "$tmp" rev-parse HEAD)" == "$green" ]] || fail "(B) revert-on-fail: HEAD moved off the green floor"
grep -q "broken WIP" "$tmp/work"                    && fail "(B) revert-on-fail: broken WIP survived the reset"
[[ "$(git -C "$tmp" rev-list --count HEAD)" -eq 2 ]] || fail "(B) revert-on-fail: a commit leaked"
echo "[OK]   (B) commit-on-pass advances the green floor; revert-on-fail (reset --hard to last green) restores it"

# ---------- (D) sentinel consistency: prompt emits == sbatch greps ----------
P="$REPO/prompts/_cluster_autonomous_lightning_port_loop.md"
S="$REPO/v2.0/HPC_scripts/run_lightning_port_loop.sbatch"
for sent in STAGE_LANDED PORT_COMPLETE BLOCKED; do
  grep -q "=== LOOP: ${sent} ===" "$P" || fail "(D) prompt never emits sentinel '${sent}'"
  grep -q "LOOP: ${sent} ===" "$S"     || fail "(D) sbatch never greps sentinel '${sent}'"
done
echo "[OK]   (D) terminal sentinels {STAGE_LANDED,PORT_COMPLETE,BLOCKED} consistent prompt<->sbatch"

echo "PREFLIGHT PASS"
