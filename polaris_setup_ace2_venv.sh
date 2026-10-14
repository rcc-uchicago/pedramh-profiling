#!/bin/bash
# polaris_setup_ace2_venv.sh — build the ACE2 (`fme`) venv on Polaris.
#
# Run ONCE, on a LOGIN node (this is the one thing that must run there: it needs
# outbound network, and the `conda` module exports the ALCF proxy on login):
#
#     bash polaris_setup_ace2_venv.sh
#
#   -> ${MEMBER_ROOT}/conda-envs/fme-venv   (YOUR member dir; override with
#      POLARIS_ACE2_VENV=<dir>)
#
# PASS = the line `ACE2_VENV_OK`. Key on that, never on the exit code.
#
# WHY A NEW VENV AND NOT ONE OF THE EIGHT ALREADY THERE
#   None of them carries `fme` (checked 2026-09-02: no site-packages/fme in any
#   of ai-rossby-venv, decrypto-serve, makani-h5py-overlay, marshal-train,
#   pangu-gcshim, pangu-shim, polaris-topups, sfno-venv). ACE2's Midway script
#   activates /project/rcc/mehta5/envs/fme, which is a MIDWAY path. This is the
#   day-one blocker the handoff §4 names.
#
# ⚠ THE TORCH VERSION IS A DELIBERATE CHOICE, NOT A DEFAULT.
#   fme's own constraints.txt pins torch==2.7.1 "version matches torch in Docker
#   image", and its Dockerfile builds on pytorch/pytorch:2.7.1-cuda12.8. That pin
#   is NOT applied here, on purpose:
#
#     * The Polaris fabric stack (aws-ofi-nccl 1.21.1 + libfabric 2.3.1 +
#       OFI_NCCL_PROGRESS_MODEL=AUTO) has been measured under exactly two torch
#       builds on this machine — makani's 2.8.0/NCCL 2.28.3 and ai-rossby's
#       2.10.0/NCCL 2.27.5 — and gave the same answer under both.
#     * The multi-node handoff §4 requires: "any torch != 2.8.0 => re-run the
#       three fabric probes under it". Choosing 2.10.0 satisfies that with the
#       ai-rossby campaign's existing probe results instead of three new jobs.
#     * 2.7.1 would satisfy neither and would cost those three jobs.
#
#   Override with ACE2_TORCH=<version> if fme turns out to need 2.7.1 — and if you
#   do, say so in the CHANGELOG and re-run the probes, because the ladder's
#   transport pin is then unmeasured.
#
# NOT INSTALLED, each for a stated reason:
#   * requirements-healpix.txt — needs --no-build-isolation and compiles; our
#     config is the lat/lon SFNO (spatial_dimensions defaults to latlon) and never
#     imports the healpix path.
#   * analysis-deps.txt — post-hoc notebook tooling, not the training path.
#   * apex — NOT needed despite the config's `optimizer_type: FusedAdam`. In this
#     vendored fme, FusedAdam means `torch.optim.AdamW(..., fused=True)`
#     (fme/core/optimization.py) and emits a DeprecationWarning. The handoff §4
#     lists apex as a trap; at this commit it is not one.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACE_DIR="${REPO}/ACE2_retrain/ace_exp"
ACE2_TORCH="${ACE2_TORCH:-2.10.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"

# ⚠ NOT `module load conda`. That is BROKEN cluster-side since the 2026-08 Cray
# PE roll (dead cray-hdf5-parallel/1.14.3.5 and gcc-native/14.2 pins), and the
# first version of this script died on it in 3 seconds:
#     Lmod has detected the following error: The following module(s) are unknown
# The sibling setup scripts (polaris_setup_ai_rossby_venv.sh,
# polaris_setup_sfno_venv.sh) still open that way and are dead for the same
# reason — TODO P0-10. polaris_ace2_env.sh already carries the fallback, so this
# script uses it rather than growing a fourth copy of it.
#
# polaris_env.sh must come first: ACE2_VENV is derived from MEMBER_ROOT there.
# (`set -u` stays off through this block; conda's activation script trips it.)
# shellcheck source=polaris_env.sh
source "${REPO}/polaris_env.sh"
# shellcheck disable=SC1091
source "${REPO}/ACE2_retrain/polaris/polaris_ace2_env.sh" || exit 2
echo "env source: ${ACE2_ENV_SOURCE}"

set -u

if [ ! -f "${ACE_DIR}/pyproject.toml" ]; then
    echo "ERROR ACE2_TREE_MISSING: ${ACE_DIR}/pyproject.toml"
    exit 2
fi

command -v uv >/dev/null 2>&1 || {
    echo "ERROR UV_MISSING: \`uv\` is not on PATH after \`module load conda\`."
    exit 2
}

# Same default as polaris_ace2_env.sh derives, and the same override name, so the
# builder and every launcher cannot disagree about where the venv is.
VENV="${ACE2_VENV:-${MEMBER_ROOT}/conda-envs/fme-venv}"
export UV_CACHE_DIR="${MEMBER_ROOT}/uv_cache"
mkdir -p "${UV_CACHE_DIR}" "$(dirname "${VENV}")"

echo "--- ACE2 (fme) venv build ---"
echo "  project = ${ACE_DIR}"
echo "  venv    = ${VENV}"
echo "  uv      = $(command -v uv) ($(uv --version))"
echo "  torch   = ${ACE2_TORCH} from ${TORCH_INDEX}"

# include-system-site-packages stays FALSE (uv's default), so nothing resolves out
# of the base conda. That is what lets polaris_ace2_env.sh skip makani's torch
# repair and h5py overlay entirely.
uv venv --python "$(command -v python)" "${VENV}"

# Torch FIRST, from the CUDA index. Installing fme first would let uv resolve
# torch off PyPI (a different CUDA build, and possibly a different version), and
# then this line would be a no-op that looks like it worked.
uv pip install --python "${VENV}/bin/python" \
    --index-url "${TORCH_INDEX}" "torch==${ACE2_TORCH}"

TORCH_BEFORE="$("${VENV}/bin/python" -c 'import torch; print(torch.__version__)')"
echo "  torch after step 1: ${TORCH_BEFORE}"

# EDITABLE, matching the upstream Dockerfile's `pip install -e .` — and load-
# bearing here: ace2_nvtx.py and ace2_telemetry.py monkeypatch fme by attribute
# path, so the installed fme must be the checkout those files were read against.
# The verification below asserts exactly that.
uv pip install --python "${VENV}/bin/python" -e "${ACE_DIR}"

TORCH_AFTER="$("${VENV}/bin/python" -c 'import torch; print(torch.__version__)')"
if [ "${TORCH_AFTER}" != "${TORCH_BEFORE}" ]; then
    echo "ERROR ACE2_TORCH_MOVED: installing fme replaced torch ${TORCH_BEFORE} -> ${TORCH_AFTER}"
    echo "  The replacement almost certainly came from PyPI (CPU or a different CUDA"
    echo "  build), so every fabric pin in polaris_ace2_env.sh is now unmeasured."
    exit 3
fi

# --- verification: the imports that actually gate the run -------------------
"${VENV}/bin/python" - "${ACE_DIR}" <<'PY'
import os
import sys

ace_dir = os.path.realpath(sys.argv[1])
failures = []


def check(label, fn):
    try:
        print(f"  {label}: {fn()}")
    except Exception as exc:                       # noqa: BLE001 — report, don't raise
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"  {label}: FAILED — {type(exc).__name__}: {exc}")


import torch  # noqa: E402

print("--- ACE2 venv verification ---")
print(f"  python: {sys.version.split()[0]}")
print(f"  torch:  {torch.__version__} (cuda {torch.version.cuda}, nccl "
      f"{'.'.join(map(str, torch.cuda.nccl.version()))})")

import fme  # noqa: E402

print(f"  fme from: {fme.__file__}")
if not os.path.realpath(fme.__file__).startswith(ace_dir):
    failures.append(
        f"fme resolves to {fme.__file__}, not the checkout at {ace_dir} — the "
        "monkeypatch harnesses would patch a different tree than they were written "
        "against"
    )

# The four modules the bring-up hangs on, in the order they fail.
check("fme.ace.train", lambda: __import__("fme.ace.train", fromlist=["main"]).__name__)
check("fme.ace.validate_config",
      lambda: __import__("fme.ace.validate_config", fromlist=["main"]).__name__)
check("stepper.single_module",
      lambda: __import__("fme.ace.stepper.single_module",
                         fromlist=["TrainStepper"]).__name__)
check("SFNO builder",
      lambda: __import__("fme.ace.models.modulus.sfnonet",
                         fromlist=["SphericalFourierNeuralOperatorNet"]).__name__)

# Every attribute ace2_telemetry.py patches. A rename upstream turns telemetry
# silently off; catching it here costs nothing.
def _hooks():
    from fme.ace.data_loading.gridded_data import GriddedData
    from fme.ace.stepper.single_module import TrainStepper
    from fme.core.generics.trainer import Trainer
    from fme.core.optimization import NullOptimization, Optimization

    missing = [
        name for obj, name in (
            (Trainer, "train_one_epoch"),
            (GriddedData, "alternate_shuffle"),
            (TrainStepper, "train_on_batch"),
            (Optimization, "step_scheduler"),
            (NullOptimization, "step_scheduler"),
        ) if not hasattr(obj, name)
    ]
    if missing:
        raise AttributeError("fme moved: " + ", ".join(missing))
    return "all 5 present"


check("telemetry hook points", _hooks)

for mod in ("torch_harmonics", "xarray", "zarr", "netCDF4", "h5netcdf", "h5py",
            "dacite", "cftime", "wandb", "tensorly", "einops", "cartopy",
            "matplotlib"):
    check(mod, lambda m=mod: getattr(__import__(m), "__version__", "ok"))

import torch_harmonics as th  # noqa: E402

if th.__version__ != "0.8.0":
    failures.append(
        f"torch_harmonics {th.__version__} != 0.8.0 — fme pins it exactly, and the "
        "Midway profile (which the Polaris numbers are compared against) was taken "
        "on 0.8.0"
    )

if failures:
    print("ERROR ACE2_VENV_VERIFY_FAILED")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)

# torch.cuda is False on a login node (no GPU) — expected, NOT a failure. The PBS
# smoke asserts it on-node.
if not torch.cuda.is_available():
    print("  (note: no GPU on a login node — CUDA is asserted on-node by the PBS smoke)")

print("ACE2_VENV_OK")
PY

echo
echo "Activate with:  source ${VENV}/bin/activate"
echo "Or set:         export POLARIS_ACE2_VENV=${VENV}"
echo "Next:           qsub -l select=1:system=polaris ACE2_retrain/polaris/polaris_ace2_train.pbs"
