#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Environment bootstrap for Makani on Polaris that does NOT depend on the
# broken `conda` modulefile.  SOURCE it; do not execute it.
#
#     source "<repo>/makani_sfno/polaris/polaris_makani_env.sh" || exit 2
#
# WHY THIS FILE EXISTS
#   Every Makani PBS script in this repo opens with `module load conda`.  Since
#   the 2026-08 Cray PE roll that module cannot load: /soft/modulefiles/conda/
#   2025-09-25.lua declares
#       depends_on("cray-hdf5-parallel/1.14.3.5")   -- only 1.14.3.9 installed
#       depends_on("gcc-native/14.2")               -- only 14 installed
#   so Lmod aborts with "The following module(s) are unknown" and `conda` never
#   lands.  Re-confirmed still broken on a login node 2026-08-21.
#   -> polaris_pbs_notes.md §1 BLOCKER for the history and the workarounds
#      already tried and rejected.
#
#   This file reconstructs, by hand, what that modulefile did, with the two dead
#   version pins replaced by the versions that ARE installed, and then repairs
#   the two dangling DT_NEEDED entries in the base conda's torch 2.8.0.
#
# IT PREFERS THE OFFICIAL PATH.  `module load conda` is attempted FIRST; the
# manual reconstruction is the fallback.  When ALCF fixes the modulefile this
# script goes back to the sanctioned env on its own, and it always reports which
# of the two it used (MAKANI_ENV_SOURCE) -- the two are not guaranteed identical
# and no table should mix them without saying so.
#
# WHAT IT DOES NOT DO
#   It does not activate the venv (callers keep control of PYTHONNOUSERSITE
#   ordering).  Required order -- polaris_env.sh comes FIRST:
#       source polaris_env.sh            # repo paths: MEMBER_ROOT, SFNO_VENV, caches
#       source polaris_makani_env.sh     # this file: interpreter + libs
#       source "${SFNO_VENV}/bin/activate"
#
#   That is the reverse of what polaris_env.sh's own header suggests ("source it
#   AFTER module load conda"), and it is safe: at source time polaris_env.sh is
#   pure path arithmetic -- its only `python` call sits inside the
#   polaris_require_topups() function body and does not run until called.
#   The order MATTERS because this file's two writable-state paths (the mpi shim
#   dir and the h5py overlay) are anchored on $MEMBER_ROOT.  Sourced first, they
#   silently fall back to $HOME, which on ALCF is a small quota'd filesystem and
#   the wrong home for a package overlay -- so an unset MEMBER_ROOT is an error
#   here, not a default.

# Deliberately no `set -e`: this file is sourced, and killing the caller's shell
# on a probe failure would hide the diagnostics printed below.

MAKANI_ENV_SOURCE="unset"
_mk_env_fail=0

if [ -z "${MEMBER_ROOT:-}" ]; then
    echo "ERROR MAKANI_ENV_NO_MEMBER_ROOT: source polaris_env.sh BEFORE this file."
    echo "  Without it the mpi shim dir and the h5py overlay would land in \$HOME."
    _mk_env_fail=1
fi

module use /soft/modulefiles 2>/dev/null

# ---- attempt 1: the sanctioned module ---------------------------------------
if module load conda >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
    MAKANI_ENV_SOURCE="module-conda"
    conda activate base
else
    # ---- attempt 2: reconstruct 2025-09-25.lua by hand ----------------------
    MAKANI_ENV_SOURCE="manual-reconstruction"

    # The modulefile's five depends_on lines. Two pins no longer exist; the
    # substitutes are the only installed versions, NOT a preference:
    #   cray-hdf5-parallel/1.14.3.5 -> 1.14.3.9  (soversion 200 -> 310, see below)
    #   gcc-native/14.2             -> 14        (affects CC/CXX for JIT only)
    if ! module load PrgEnv-gnu craype-x86-milan cudnn/9.13.0 gcc-native/14 \
                     cray-hdf5-parallel/1.14.3.9 >/dev/null 2>&1; then
        echo "ERROR MAKANI_ENV_MODULES_FAILED: the substitute module set no longer loads either."
        echo "  Tried: PrgEnv-gnu craype-x86-milan cudnn/9.13.0 gcc-native/14 cray-hdf5-parallel/1.14.3.9"
        echo "  The PE has rolled again. Re-derive the set from /soft/modulefiles/conda/2025-09-25.lua"
        echo "  and update this file AND polaris_pbs_notes.md §1 together."
        _mk_env_fail=1
    fi

    # Its setenv/prepend_path lines, copied value-for-value.
    export CC=/usr/bin/gcc-14
    export CXX=/usr/bin/g++-14
    export TORCH_CUDA_ARCH_LIST=8.0          # A100 = sm80
    export MPICH_GPU_SUPPORT_ENABLED=1
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export http_proxy=http://proxy.alcf.anl.gov:3128

    _mk_cuda_home=/soft/compilers/cudatoolkit/cuda-12.9.1
    export CUDA_HOME="${_mk_cuda_home}"
    export CUDA_PATH="${_mk_cuda_home}"
    export CUDA_TOOLKIT_BASE="${_mk_cuda_home}"
    export PATH="${_mk_cuda_home}/bin:${PATH}"
    export LD_LIBRARY_PATH="${_mk_cuda_home}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${_mk_cuda_home}/lib64:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH=/soft/libraries/trt/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=/soft/libraries/nccl/nccl_2.28.3-1+cuda12.9_x86_64/lib:${LD_LIBRARY_PATH}

    # Initialize conda from the install directory. The INSTALL is intact -- only
    # the modulefile that points at it is broken.
    _mk_conda_dir=/soft/applications/conda/2025-09-25/mconda3
    if [ ! -r "${_mk_conda_dir}/etc/profile.d/conda.sh" ]; then
        echo "ERROR MAKANI_ENV_CONDA_GONE: ${_mk_conda_dir} no longer holds conda.sh."
        echo "  The base conda install itself has moved -- no longer a modulefile problem."
        _mk_env_fail=1
    else
        # shellcheck disable=SC1091
        source "${_mk_conda_dir}/etc/profile.d/conda.sh"
        conda activate base
    fi
fi

# ---- repair torch 2.8.0's two dangling DT_NEEDED entries --------------------
# `ldd .../torch/lib/libtorch_global_deps.so` reports two "not found" in a bare
# environment. torch's _load_global_deps CATCHES the OSError and falls through
# to _preload_cuda_deps, so these may well be non-fatal -- but a caught
# exception is not a measurement, and a half-loaded libtorch is not something to
# take into a scaling study. Both are resolvable:
#
#  * libcudart.so.13 -- present in the cuda-13.0.1 toolkit. Purely a path.
#  * libmpi_gnu_123.so.12 -- polaris_pbs_notes.md §1 recorded this as "not
#    present anywhere". That is true of the FILENAME and false of the LIBRARY:
#    `_123` was the old cray-mpich's way of spelling the gcc-12.3 build, and
#    mpich/9.1.0 ships that exact build as libmpi_gnu.so.12 under .../gnu/12.3/.
#    SONAME and soversion both stay 12, so a symlink under the old name is a
#    rename, not an ABI substitution. (The modulefile's own last line names this
#    soname in a commented-out PyTorch hotfix.)
#    Contrast cray-hdf5-parallel, where 1.14.3.5's libhdf5_parallel_gnu_123.so.200
#    became 1.14.3.9's ...gnu.so.310: soversion 200 -> 310 IS an ABI break and
#    must not be symlinked. That is why h5py needs the overlay below instead.
export LD_LIBRARY_PATH="/soft/compilers/cudatoolkit/cuda-13.0.1/lib64:${LD_LIBRARY_PATH}"

MAKANI_LIB_SHIMS="${MAKANI_LIB_SHIMS:-${MEMBER_ROOT}/lib-shims/polaris-mpich-compat}"
_mk_mpi_real="${CRAY_MPICH_DIR:-/opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3}/lib/libmpi_gnu.so.12"
if [ -e "${_mk_mpi_real}" ]; then
    if [ ! -e "${MAKANI_LIB_SHIMS}/libmpi_gnu_123.so.12" ]; then
        mkdir -p "${MAKANI_LIB_SHIMS}" &&
            ln -sfn "${_mk_mpi_real}" "${MAKANI_LIB_SHIMS}/libmpi_gnu_123.so.12"
    fi
    export LD_LIBRARY_PATH="${MAKANI_LIB_SHIMS}:${LD_LIBRARY_PATH}"
else
    echo "WARN MAKANI_ENV_NO_MPI_SHIM: ${_mk_mpi_real} absent; libmpi_gnu_123.so.12 stays unresolved."
    echo "  Not necessarily fatal (torch catches it) but say so in any table from this run."
fi

# ---- h5py overlay -----------------------------------------------------------
# makani/utils/metric.py:19 does a bare `import h5py`, so h5py is on the import
# path of EVERY makani entrypoint -- including --enable_synthetic_data runs, so
# synthetic data does not dodge it. The base conda's h5py is linked against the
# deleted hdf5 1.14.3.5 (libhdf5_parallel_gnu_123.so.200) and cannot be shimmed
# (see above), so it is replaced by a PyPI wheel, which vendors its own libhdf5
# under h5py.libs and needs no Cray hdf5 at all.
#
# Same minimal-overlay pattern as $POLARIS_TOPUPS, and minimal for the same
# reason: PYTHONPATH outranks site-packages, so an overlay holding anything else
# could shadow the venv's torch / torch_harmonics -- the exact failure CLAUDE.md
# and the Makani PBS scripts already guard against. h5py ONLY.
# Build it once with polaris_setup_makani_h5py_overlay.sh.
MAKANI_H5PY_OVERLAY="${MAKANI_H5PY_OVERLAY:-${MEMBER_ROOT}/conda-envs/makani-h5py-overlay}"
if [ -d "${MAKANI_H5PY_OVERLAY}" ]; then
    export PYTHONPATH="${MAKANI_H5PY_OVERLAY}:${PYTHONPATH:-}"
    export MAKANI_H5PY_OVERLAY
fi

export MAKANI_ENV_SOURCE MAKANI_LIB_SHIMS

makani_env_report() {
    echo "MAKANI_ENV_SOURCE      = ${MAKANI_ENV_SOURCE}"
    echo "python                 = $(command -v python || echo '<none>')"
    echo "CUDA_HOME              = ${CUDA_HOME:-<unset>}"
    echo "mpi shim dir           = ${MAKANI_LIB_SHIMS}"
    echo "h5py overlay           = ${MAKANI_H5PY_OVERLAY:-<absent, base conda h5py will fail>}"
    local _t _missing
    # Search sys.path rather than sysconfig's purelib: purelib is the VENV's
    # site-packages, but torch lives in the BASE conda's, so the old form printed
    # "<could not locate ...>" on every run -- a check that silently answers
    # nothing is worse than no check. Still does NOT import torch: this must stay
    # callable on a login node (CLAUDE.md #3).
    _t="$(python -c 'import os,sys
for d in sys.path:
    p = os.path.join(d, "torch", "lib", "libtorch_global_deps.so")
    if os.path.exists(p):
        print(p); break' 2>/dev/null)"
    if [ -n "${_t}" ] && [ -e "${_t}" ]; then
        _missing="$(ldd "${_t}" 2>/dev/null | grep -c 'not found')"
        echo "libtorch dangling libs = ${_missing}"
        if [ "${_missing}" != "0" ]; then
            ldd "${_t}" 2>/dev/null | grep 'not found' | sed 's/^/    still unresolved: /'
        fi
    else
        echo "libtorch dangling libs = <could not locate libtorch_global_deps.so>"
    fi
}

return ${_mk_env_fail} 2>/dev/null || true
