#!/bin/bash -l
# ============================================================================
# One-time setup: an ISOLATED venv for the SFNO frameworks (makani + physicsnemo).
#
# WHY ISOLATED: makani 0.2.0 imports `precompute_latitudes` from
# torch_harmonics.quadrature, which is PUBLIC only in torch_harmonics 0.9.x
# (0.7.4 / 0.8.0 expose only the private `_precompute_latitudes`). But the 0.9.1
# *wheel* ships an `attention/_C.so` built against a different torch ABI and dies
# on torch 2.8 with `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv`,
# so 0.9.x must be built FROM SOURCE against the local torch.
# The base conda must KEEP torch_harmonics 0.7.4 — the GREEN PanguWeather-SFNO
# (job 7252271) and SI (job 7252700) smokes run on it and both use torch_harmonics.
# Hence: base conda = 0.7.4 (greens), this venv = 0.9.x-from-source (makani).
#
# HOW: a `--system-site-packages` venv layered on the ALCF base conda, so it
# INHERITS the CUDA-12.9-matched torch 2.8 (no 2.5 GB reinstall) and only the
# SFNO-specific packages are installed/overridden here. venvs disable the user
# site-packages, so the base's --user torch_harmonics 0.7.4 is NOT visible inside.
#
# RUN ON A LOGIN NODE (compute nodes have no outbound network):
#     bash polaris_setup_sfno_venv.sh
# PASS = the final line "SFNO_VENV_OK".
# The makani / physicsnemo polaris_*.pbs scripts activate this venv.
# ============================================================================
set -uo pipefail

VENV=/eagle/projects/lighthouse-uchicago/members/mehta5/conda-envs/sfno-venv
REPO=/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling
MAKANI_PIN=c97043086e60d44a3adc3bede9a6b3dc71f5005d   # README-mandated pin (0.2.0 wheel lacks the
                                                      # cache_unpredicted_features clone fix)

module use /soft/modulefiles
module load conda
conda activate base

# Be polite to the shared login node (it throttles thread-hungry builds:
# "OMP: System error #11: Resource temporarily unavailable").
export OMP_NUM_THREADS=1
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=8.0        # A100 sm80
export HDF5_USE_FILE_LOCKING=FALSE
export PIP_CACHE_DIR=/eagle/projects/lighthouse-uchicago/members/mehta5/pip_cache

echo "=== creating venv (inherits base conda torch) : ${VENV} ==="
python -m venv --system-site-packages "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# CRITICAL: a --system-site-packages venv re-enables the USER site (~/.local), and
# site.py adds user-site BEFORE the venv's own site-packages — so the base env's
# --user torch_harmonics 0.7.4 / makani would SHADOW this venv's 0.9.x and the job
# dies with "cannot import name 'precompute_latitudes'". PYTHONNOUSERSITE=1 drops the
# user site; the conda base (SYSTEM site) still provides torch/h5py/xarray/einops/timm.
export PYTHONNOUSERSITE=1

python -c "import sys, torch; print('venv python:', sys.executable); print('inherited torch:', torch.__version__, torch.version.cuda)"

# python3.12 venvs do NOT ship setuptools, and --no-build-isolation builds need it.
echo "=== bootstrap build tooling in the venv ==="
pip install --no-cache-dir -U pip setuptools wheel

echo "=== torch_harmonics 0.9.x FROM SOURCE (ABI-matched _C + public precompute_latitudes) ==="
pip install --no-cache-dir --no-build-isolation --no-deps \
    "torch_harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git"

echo "=== makani (pinned) + physicsnemo (local tree, editable) — --no-deps protects torch 2.8 ==="
pip install --no-cache-dir --no-deps "makani @ git+https://github.com/NVIDIA/makani.git@${MAKANI_PIN}"
pip install --no-cache-dir --no-deps -e "${REPO}/physicsnemo_sfno"

echo "=== remaining runtime deps (not in base conda / not visible via user-site) ==="
pip install --no-cache-dir warp-lang s3fs treelib netCDF4 h5netcdf "zarr<3" moviepy
pip install --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist \
    nvidia-dali-cuda120
# makani import-time deps. NOTE: the module `tltorch` ships as the PyPI package
# **tensorly-torch** — `pip install tltorch` fails with "No matching distribution".
pip install --no-cache-dir tensorly tensorly-torch numba

echo "=== VERIFY ==="
python - <<'PY'
import importlib, sys
ok = True
for m in ["torch", "torch_harmonics", "makani", "physicsnemo", "warp", "nvidia.dali",
          "h5py", "netCDF4", "zarr", "tensorly", "tltorch"]:
    try:
        mod = importlib.import_module(m)
        print("  OK   %-16s %s" % (m, getattr(mod, "__version__", "")))
    except Exception as e:
        ok = False
        print("  FAIL %-16s %s: %s" % (m, type(e).__name__, str(e)[:90]))
try:
    from torch_harmonics.quadrature import precompute_latitudes  # noqa: F401
    print("  OK   precompute_latitudes (makani's hard requirement)")
except Exception as e:
    ok = False
    print("  FAIL precompute_latitudes: %s" % e)

# Provenance gate: torch_harmonics/makani MUST resolve from the venv, not from the
# base env's --user site (~/.local), which still holds 0.7.4 for the Pangu/SI smokes.
import os
import torch_harmonics, makani
venv = os.environ.get("VIRTUAL_ENV", "")
for mod in (torch_harmonics, makani):
    src = getattr(mod, "__file__", "") or ""
    if not src.startswith(venv):
        ok = False
        print("  FAIL %s resolves from %s (expected inside %s) — user-site shadowing"
              % (mod.__name__, src, venv))
    else:
        print("  OK   %s resolves from the venv" % mod.__name__)
sys.exit(0 if ok else 1)
PY
rc=$?
if [ ${rc} -ne 0 ]; then
    echo "ERROR SFNO_VENV_FAILED (see FAIL lines above)"
    exit 1
fi
echo "venv: ${VENV}"
echo "SFNO_VENV_OK"
