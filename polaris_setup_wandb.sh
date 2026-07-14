#!/bin/bash -l
# ============================================================================
# One-time Weights & Biases setup for experiment tracking on Polaris.
#
#     bash polaris_setup_wandb.sh            # PASS = "WANDB_OK"
#
# Run it once on a LOGIN node. Afterwards, any job can log live:
#     qsub -v WANDB_MODE=online <script>
#
# CAN COMPUTE NODES REACH W&B? YES — verified on-node, not assumed (job 7253810,
# x3206c0s7b1n0): https://api.wandb.ai/healthz -> HTTP 200 through the ALCF proxy that
# `module load conda` exports (https_proxy=http://proxy.alcf.anl.gov:3128). A DIRECT
# connection fails (HTTP 000). Per ALCF's own docs, the proxy is the ONLY route out:
#     http_proxy / https_proxy / ftp_proxy = http://proxy.alcf.anl.gov:3128
# So the module must be loaded before anything talks to W&B — every polaris_*.pbs does.
#
# SECRETS: your API key goes in ~/.netrc (mode 0600, written by `wandb login`) or in
# $WANDB_API_KEY. NEVER put it in a script, a config, or anything git tracks — CLAUDE.md #8.
# ALCF homes are mode 0700, which is exactly right for a secret: it is the one thing that
# SHOULD be readable by you alone. (Contrast $POLARIS_TOPUPS — shared code/deps must NOT
# live in a private home; see polaris_setup_base_topups.sh.)
# ============================================================================
set -uo pipefail

if [[ "$(hostname)" != *login* ]]; then
    echo "ERROR WRONG_NODE: run this on a Polaris LOGIN node."
    echo "  It needs an interactive prompt for your API key. The RUNS themselves reach W&B"
    echo "  fine from compute nodes via the proxy — it is only this login step that wants a TTY."
    exit 2
fi

module use /soft/modulefiles
module load conda
conda activate base

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/polaris_env.sh" || exit 2

echo "=== W&B setup ==="
echo "  python      : $(command -v python)"
echo "  wandb       : $(python -c 'import wandb; print(wandb.__version__)' 2>/dev/null || echo MISSING)"
echo "  WANDB_DIR   : ${WANDB_DIR}"
echo "  WANDB_PROJECT: ${WANDB_PROJECT}"
echo "  https_proxy : ${https_proxy:-<unset — module load conda should set this>}"

if ! python -c "import wandb" 2>/dev/null; then
    echo "ERROR WANDB_MISSING: the base conda should already provide wandb."
    exit 2
fi

# ---- 1) key ----------------------------------------------------------------
# Precedence mirrors wandb's own: $WANDB_API_KEY beats ~/.netrc.
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "  using \$WANDB_API_KEY from the environment"
elif grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "  found an existing key in ~/.netrc"
else
    echo
    echo "  No API key yet. Get one from https://wandb.ai/authorize (sign in first),"
    echo "  then paste it at the prompt. It is stored in ~/.netrc, readable only by you."
    echo
    wandb login || { echo "ERROR WANDB_LOGIN_FAILED"; exit 2; }
fi

# ---- 2) prove it actually works, against the real backend -------------------
# A key that is present is not a key that works. Ask the API who we are.
python - <<'PY' || { echo "ERROR WANDB_VERIFY_FAILED"; exit 2; }
import os, sys
os.environ["WANDB_MODE"] = "online"          # force a real call, not a no-op
import wandb
try:
    # NB: `viewer` is a PROPERTY in wandb 0.22, not a method. Calling it raises
    # "TypeError: 'User' object is not callable" — which looks like an auth failure but is
    # actually auth SUCCEEDING. Do not add parentheses back.
    v = wandb.Api(timeout=20).viewer
    print("  authenticated as: %s (entity: %s)" % (
        getattr(v, "username", "?"), getattr(v, "entity", "?")))
except Exception as e:
    print("  ERROR: %s: %s" % (type(e).__name__, str(e)[:150]))
    print("  If this says api_key not configured, the login above did not stick.")
    print("  If it is a network error, check that `module load conda` set https_proxy.")
    sys.exit(2)
PY

# ---- 3) prove the netrc is not world-readable ------------------------------
if [ -f "${HOME}/.netrc" ]; then
    perm=$(stat -c '%a' "${HOME}/.netrc")
    if [ "${perm}" != "600" ]; then
        echo "  WARNING: ~/.netrc is mode ${perm}; wandb expects 600. Fixing."
        chmod 600 "${HOME}/.netrc"
    fi
    echo "  ~/.netrc mode: $(stat -c '%a' "${HOME}/.netrc") (private — correct for a secret)"
fi

cat <<'EOF'

To log a run live, add WANDB_MODE=online to the submit:
    qsub -v WANDB_MODE=online PanguWeather/v2.0/HPC_scripts/polaris_train_e3sm_sfno.pbs

Runs land in the project "pedramh-profiling" (override with -v WANDB_PROJECT=<name>).

Jobs stay OFFLINE by default. That is deliberate: an offline run cannot fail, stall, or
leak on a network hiccup, and a preempted job leaves a clean local record. To upload an
offline run afterwards, from a LOGIN node:
    wandb sync <your member dir>/wandb/offline-run-*

EOF
echo "WANDB_OK"
