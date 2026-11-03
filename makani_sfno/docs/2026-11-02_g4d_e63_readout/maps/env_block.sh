# Env bootstrap copied verbatim from polaris_g4d63_ckpt_prep.pbs (itself from polaris_climate_run.pbs).
# Sourced by the maps PBS scripts, which are submitted from eval-protocol/makani_sfno/.
cd "${PBS_O_WORKDIR}" || exit 2
MAKANI_ROOT="$(cd "${PBS_O_WORKDIR}" && pwd)"
if [ ! -d "${MAKANI_ROOT}/src/sfno_inference" ]; then
    MAKANI_ROOT="$(cd "${PBS_O_WORKDIR}/.." && pwd)"
fi
REPO="$(cd "${MAKANI_ROOT}/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO}/polaris_env.sh" || exit 2
# shellcheck disable=SC1091
source "${MAKANI_ROOT}/polaris/polaris_makani_env.sh" || exit 2
if [ ! -x "${SFNO_VENV}/bin/python" ]; then
    echo "ERROR SFNO_VENV_MISSING: ${SFNO_VENV}"
    exit 3
fi
# shellcheck disable=SC1091
source "${SFNO_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MAKANI_ROOT}/src:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="${SFNO_VENV}/bin/python"
M=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/g4d_e63_2026-11-02/maps
E=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval
