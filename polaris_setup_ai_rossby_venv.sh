#!/bin/bash
# polaris_setup_ai_rossby_venv.sh — build the ai-rossby (PanguPlasim) venv on Polaris.
#
# Run ONCE, on a LOGIN node (needs outbound network; compute nodes only have the
# proxy and this wastes allocation):
#
#     bash polaris_setup_ai_rossby_venv.sh
#
# ai-rossby CANNOT reuse $SFNO_VENV. It needs torch>=2.10 (that venv has 2.8) and
# zarr>=3 (v2 there), and its pyproject pins torch <2.11 for a real multi-GPU DDP
# regression. So it gets its own, built by `uv` — which ships with the ALCF conda
# module, so nothing is bootstrapped.
#
#   -> ${MEMBER_ROOT}/conda-envs/ai-rossby-venv   (YOUR member dir; override with
#      POLARIS_AI_ROSSBY_VENV=<dir>)
#
# ⚠ Like the SFNO venv, this one installs physicsnemo EDITABLE — the venv imports
#   `physicsnemo` from whichever checkout `uv sync` ran in. Build it from YOUR OWN
#   clone. polaris_pangu_plasim.pbs hard-fails (AI_ROSSBY_WRONG_CHECKOUT) rather
#   than let a run silently execute someone else's working tree.
#
# Wheels only — no source builds. torch_harmonics and wandb come from the
# sfno-extras / utils-extras groups; xarray / zarr>=3 / netCDF4 / h5py / cftime /
# hydra are already core deps in the ai-rossby fork, so no datapipes-extras needed.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${REPO}/physicsnemo_ai_rossby"

# `conda activate base` is REQUIRED, not decoration: `module load conda` alone
# puts neither `python` nor `uv` on PATH — it only registers the shell function.
# (`set -u` stays off through this block; conda's activation script trips it.)
module use /soft/modulefiles
module load conda
conda activate base

set -u
# shellcheck source=polaris_env.sh
source "${REPO}/polaris_env.sh"

if [ ! -f "${PROJECT}/pyproject.toml" ]; then
    echo "ERROR AI_ROSSBY_TREE_MISSING: ${PROJECT}/pyproject.toml"
    echo "  The subtree is not vendored. See polaris_pbs_notes.md §6b."
    exit 2
fi

command -v uv >/dev/null 2>&1 || {
    echo "ERROR UV_MISSING: \`uv\` is not on PATH after \`module load conda\`."
    exit 2
}

VENV="${POLARIS_AI_ROSSBY_VENV:-${MEMBER_ROOT}/conda-envs/ai-rossby-venv}"

# uv writes the env at UV_PROJECT_ENVIRONMENT instead of <project>/.venv — keeping
# a multi-GB tree out of the git checkout, and on eagle rather than the 45 GB home.
export UV_PROJECT_ENVIRONMENT="${VENV}"
export UV_CACHE_DIR="${MEMBER_ROOT}/uv_cache"
mkdir -p "${UV_CACHE_DIR}" "$(dirname "${VENV}")"

echo "--- ai-rossby venv build ---"
echo "  project = ${PROJECT}"
echo "  venv    = ${VENV}"
echo "  uv      = $(command -v uv) ($(uv --version))"

cd "${PROJECT}"
uv sync --extra cu129 --extra sfno-extras --extra utils-extras

# --- verification: the imports that actually gate the run -------------------
# Key on the printed token, not on exit code alone (CLAUDE.md).
"${VENV}/bin/python" - <<'PY'
import sys

failures = []


def check(label, fn):
    try:
        print(f"  {label}: {fn()}")
    except Exception as exc:                       # noqa: BLE001 — report, don't raise
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"  {label}: FAILED — {type(exc).__name__}: {exc}")


import torch  # noqa: E402

print("--- ai-rossby venv verification ---")
print(f"  python: {sys.version.split()[0]}")
print(f"  torch:  {torch.__version__} (cuda {torch.version.cuda})")
print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

# The two modules the whole bring-up hangs on.
check("datapipes.plasim", lambda: __import__(
    "physicsnemo.experimental.datapipes.plasim", fromlist=["PlasimClimateDatapipe"]
).__name__)
check("models.pangu_plasim", lambda: __import__(
    "physicsnemo.experimental.models.pangu_plasim", fromlist=["PanguPlasimLegacy"]
).__name__)

# Editable install must resolve to THIS checkout, not a stale/shared one.
import physicsnemo  # noqa: E402

print(f"  physicsnemo from: {physicsnemo.__file__}")

for mod in ("zarr", "xarray", "netCDF4", "h5py", "cftime", "wandb", "hydra", "torch_harmonics"):
    check(mod, lambda m=mod: __import__(m).__version__ if hasattr(__import__(m), "__version__") else "ok")

import zarr  # noqa: E402

if int(zarr.__version__.split(".")[0]) < 3:
    failures.append(f"zarr {zarr.__version__} < 3 — the ai-rossby stores are zarr v3")

if failures:
    print("ERROR AI_ROSSBY_VENV_VERIFY_FAILED")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)

# torch.cuda is False on a login node (no GPU) — that is expected and NOT a failure.
# The PBS smoke asserts it on-node. Say so rather than printing a scary False.
if not torch.cuda.is_available():
    print("  (note: no GPU on a login node — CUDA is asserted on-node by the PBS smoke)")

print("AI_ROSSBY_VENV_OK")
PY

echo
echo "Activate with:  source ${VENV}/bin/activate"
echo "Or set:         export POLARIS_AI_ROSSBY_VENV=${VENV}"
