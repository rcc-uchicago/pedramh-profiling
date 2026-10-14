#!/bin/bash -l
# ============================================================================
# makani port M1: build sfno-venv-main BESIDE sfno-venv (makani_port/HANDOFF_worker.md §2, §3).
#
# Sibling of polaris_setup_sfno_venv.sh, which stays untouched and keeps building the old venv.
# Same bootstrap (a --system-site-packages venv on the ALCF base conda, so torch 2.8 is the base
# conda's, not reinstalled). Differences, each deliberate:
#   * makani      @ upstream main a0aa4c4fe5c4... (full sha), --no-deps
#   * torch_harmonics from source at the SAME commit the old venv has (2edb24ed...), not "latest"
#   * physicsnemo NON-editable, from `git archive` of physicsnemo_sfno at the repo HEAD the build
#                 runs from, so no edit to physicsnemo_sfno/ can reach this venv (and the old venv's
#                 editable install is not touched)
#   * zarr>=3     (makani main requires it); every other runtime dep pinned to the old venv's version
#
# Never touches sfno-venv: refuses a target that resolves to it, and the M1 job hashes the old
# venv's makani dist-info before and after. Run INSIDE polaris/polaris_makani_port_m1.pbs (a compute
# node, through the ALCF proxy) -- not on the login node (pid cap; makani_port/_papercuts.md).
# PASS = final line "SFNO_VENV_MAIN_BUILT". The import/provenance gate (VENV_MAIN_OK) is the job's.
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${REPO}/polaris_env.sh" || exit 2
VENV="${SFNO_VENV_MAIN:-${MEMBER_ROOT}/conda-envs/sfno-venv-main}"
OLD_VENV="${MEMBER_ROOT}/conda-envs/sfno-venv"
MAKANI_SHA=a0aa4c4fe5c40207d4fcc3da61d4c656a4ea0346
TH_SHA=2edb24edcdcfa34abcb4885691818540defb4ee6

if [ "$(readlink -f "${VENV}")" = "$(readlink -f "${OLD_VENV}")" ]; then
    echo "ERROR VENV_MAIN_IS_OLD_VENV: ${VENV}"; exit 2
fi
if [ -e "${VENV}" ]; then
    if [ "${REBUILD:-0}" != "1" ]; then
        echo "ERROR VENV_MAIN_EXISTS: ${VENV} (set REBUILD=1 to replace it)"; exit 2
    fi
    case "$(basename "${VENV}")" in
        sfno-venv-main*) rm -rf "${VENV}" ;;
        *) echo "ERROR VENV_MAIN_NAME_UNEXPECTED: refusing to remove ${VENV}"; exit 2 ;;
    esac
fi

module use /soft/modulefiles
module load conda
conda activate base

export OMP_NUM_THREADS=1
export MAX_JOBS="${MAX_JOBS:-16}"
export TORCH_CUDA_ARCH_LIST=8.0
export HDF5_USE_FILE_LOCKING=FALSE
export PIP_CACHE_DIR="${MEMBER_ROOT}/pip_cache"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"

echo "=== creating venv (inherits base conda torch) : ${VENV} ==="
python -m venv --system-site-packages "${VENV}" || exit 3
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONNOUSERSITE=1
python -c "import sys, torch; print('venv python:', sys.executable); print('inherited torch:', torch.__version__, torch.version.cuda)"

echo "=== bootstrap build tooling ==="
pip install --no-cache-dir -U pip setuptools wheel || exit 3

echo "=== torch_harmonics FROM SOURCE @ ${TH_SHA} (same commit as sfno-venv) ==="
pip install --no-cache-dir --no-build-isolation --no-deps \
    "torch_harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@${TH_SHA}" || exit 3

echo "=== makani @ ${MAKANI_SHA} (--no-deps protects torch 2.8) ==="
pip install --no-cache-dir --no-deps "makani @ git+https://github.com/NVIDIA/makani.git@${MAKANI_SHA}" || exit 3

PN_SRC="$(mktemp -d)"
PN_TREE="$(git -C "${REPO}" rev-parse HEAD:physicsnemo_sfno)"
echo "=== physicsnemo NON-editable from physicsnemo_sfno tree ${PN_TREE} (repo HEAD $(git -C "${REPO}" rev-parse HEAD)) ==="
git -C "${REPO}" archive HEAD:physicsnemo_sfno | tar -x -C "${PN_SRC}" || exit 3
pip install --no-cache-dir --no-deps "${PN_SRC}" || exit 3
echo "${PN_TREE}" > "${VENV}/physicsnemo_sfno_tree.txt"
rm -rf "${PN_SRC}"

echo "=== runtime deps: old venv's versions, except zarr>=3 ==="
pip install --no-cache-dir "warp-lang==1.15.0" "s3fs==2026.6.0" "treelib==1.8.0" "netCDF4==1.7.4" \
    "moviepy==2.2.1" "tensorly==0.9.0" "tensorly-torch==0.5.0" "mlflow==3.14.0" "zarr>=3" || exit 3
pip install --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist \
    "nvidia-dali-cuda120==2.2.0" || exit 3

echo "venv: ${VENV}"
echo "SFNO_VENV_MAIN_BUILT makani=${MAKANI_SHA} torch_harmonics=${TH_SHA} physicsnemo_tree=${PN_TREE}"
