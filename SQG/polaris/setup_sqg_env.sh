#!/bin/bash -l
# setup_sqg_env.sh -- build the SQG profiling env (a pip copy of Midway's wdenv) on a Polaris LOGIN node.
#   bash -l SQG/polaris/setup_sqg_env.sh            # from the repo root; ~5 GB of wheels, 10-20 min
# Login node because pip needs the network (compute nodes have no direct route). Installs only -- it never
# imports jax/torch here (CLAUDE.md #3; the 09-24 ruling: no venv Python on login nodes). The import check is
# the first debug job, SQG/polaris/sqg_env_probe.pbs (SQG_ENV_OK + SQG_IMPORT_OK).
# The CUDA libraries (cuFFT, cuDNN, cuBLAS, nvcc/ptxas) come from the nvidia-*-cu12 wheels, so the run-time PBS
# scripts load no module at all; conda is used here only to get a python 3.10 interpreter (wdenv is cp310).
# Prints SQG_ENV_BUILT <python> on success.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MEMBERS=/eagle/projects/lighthouse-uchicago/members
if [ -z "${MEMBER_ROOT:-}" ]; then                     # as polaris_env.sh: the member dir is not always $USER
    if [ -w "${MEMBERS}/${USER}" ]; then MEMBER_ROOT="${MEMBERS}/${USER}"
    else for d in "${MEMBERS}"/*/; do [ -O "$d" ] && [ -w "$d" ] && { MEMBER_ROOT="${d%/}"; break; }; done; fi
fi
[ -n "${MEMBER_ROOT:-}" ] && [ -w "$MEMBER_ROOT" ] || { echo "ERROR SQG_NO_MEMBER_ROOT: ${MEMBER_ROOT:-<unset>}"; exit 2; }
ENV="${SQG_ENV:-${MEMBER_ROOT}/conda-envs/sqg-env}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${MEMBER_ROOT}/pip_cache}"    # NOT $HOME (small quota)
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
# The login node caps each user at 256 pids and threads count: keep every tool single-threaded.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       CONDA_DEFAULT_THREADS=1 CONDA_FETCH_THREADS=1 CONDA_REPODATA_THREADS=1 CONDA_VERIFY_THREADS=1 \
       CONDA_EXECUTE_THREADS=1
echo "repo=$REPO member_root=$MEMBER_ROOT env=$ENV pip_cache=$PIP_CACHE_DIR"

if [ -x "${ENV}/bin/python" ]; then
    SRC="existing"
else
    # Interpreter, in the prompt's order (PROFILING_PROMPT.md §2c). `module load conda` works again: ALCF fixed
    # the modulefile's dead pins on 2026-09-16.
    module use /soft/modulefiles
    if module load conda >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
        SRC="module-conda"
    else
        source /soft/applications/conda/2025-09-25/mconda3/etc/profile.d/conda.sh
        SRC="conda.sh"
    fi
    if conda create -y -p "$ENV" python=3.10; then
        SRC="${SRC}+conda-create-py3.10"
    else
        echo "WARN SQG_CONDA_CREATE_FAILED: falling back to a venv on the base conda python"
        rm -rf "$ENV"
        python3 -m venv "$ENV"                          # never --system-site-packages
        SRC="${SRC}+venv-base-python"
    fi
fi
echo "interpreter source: $SRC -> $("$ENV/bin/python" -V 2>&1)"

PIP=("$ENV/bin/python" -m pip --disable-pip-version-check)
"${PIP[@]}" install -q --progress-bar off -U pip
if ! "${PIP[@]}" install --progress-bar off -r "$REPO/SQG/polaris/requirements-sqg.txt" \
        -c "$REPO/SQG/polaris/requirements-wdenv.txt"; then
    echo "WARN SQG_CONSTRAINTS_FAILED: retrying with the top-level pins only (no -c)"
    "${PIP[@]}" install --progress-bar off -r "$REPO/SQG/polaris/requirements-sqg.txt"
fi
"${PIP[@]}" freeze > "$ENV/sqg-env-freeze.txt"
for pkg in jax jaxlib jax-cuda12-plugin pyqg-jax torch timm numpy nvidia-cufft-cu12 nvidia-cudnn-cu12 \
           nvidia-cuda-nvcc-cu12; do
    python3 - "$ENV/sqg-env-freeze.txt" "$pkg" <<'EOF'
import sys
f, pkg = sys.argv[1], sys.argv[2].lower()
hits = [l.strip() for l in open(f) if l.lower().replace("_", "-").startswith(pkg + "==")]
print("  " + (hits[0] if hits else f"ERROR SQG_PKG_MISSING {pkg}"))
EOF
done
echo "SQG_ENV_BUILT $ENV/bin/python (source: $SRC; freeze: $ENV/sqg-env-freeze.txt)"
