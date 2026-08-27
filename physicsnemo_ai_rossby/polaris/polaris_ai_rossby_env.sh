#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Environment bootstrap for ai-rossby on Polaris that does NOT depend on the
# broken `conda` modulefile.  SOURCE it; do not execute it.
#
#     source "<repo>/physicsnemo_ai_rossby/polaris/polaris_ai_rossby_env.sh" || exit 2
#
# WHY THIS FILE EXISTS
#   polaris_sfno_e3sm.pbs and polaris_sfno_e3sm_multinode.pbs both open with a
#   bare `module load conda`.  That worked when they were written (2026-08-14)
#   and has been BROKEN cluster-side since the 2026-08 Cray PE roll:
#   /soft/modulefiles/conda/2025-09-25.lua declares
#       depends_on("cray-hdf5-parallel/1.14.3.5")   -- only 1.14.3.9 installed
#       depends_on("gcc-native/14.2")               -- only 14 installed
#   so Lmod aborts and `conda` never lands.  -> polaris_pbs_notes.md §1 BLOCKER
#   (which also lists the three workarounds already tried and rejected).
#
#   makani hit this first and solved it in polaris_makani_env.sh.  This file is
#   the ai-rossby sibling of that one, and the DELTA IS DELIBERATE — it is NOT a
#   copy with a variable renamed:
#
#     * NO torch-2.8 DT_NEEDED repair.  makani runs the BASE CONDA's torch 2.8.0,
#       whose libtorch_global_deps.so has two dangling libs.  ai-rossby runs its
#       own venv's torch 2.10.0+cu129, which ships the whole nvidia stack as
#       wheels under site-packages and has no such gap.
#     * NO h5py overlay.  That exists because makani/utils/metric.py does a bare
#       `import h5py` against the base conda's hdf5-1.14.3.5-linked build.
#       ai-rossby reads zarr and never imports the base conda's h5py — the venv
#       is built with include-system-site-packages=false (pyvenv.cfg), so base
#       site-packages is not even on its import path.
#     * NO /soft/libraries/nccl prepend.  makani wants the system NCCL 2.28.3 its
#       torch was built against.  Prepending it here would put NCCL 2.28.3 ahead
#       of the NCCL the ai-rossby venv bundles (nvidia/nccl/lib/libnccl.so.2),
#       i.e. it would silently change the communication library under a scaling
#       measurement.  The whole point of §1 of the multi-node handoff is that
#       every pin gets re-measured under THIS venv, not inherited.
#
# IT PREFERS THE OFFICIAL PATH.  `module load conda` is attempted FIRST; the
# manual reconstruction is the fallback.  When ALCF fixes the modulefile this
# script returns to the sanctioned env on its own, and it always reports which of
# the two it used (AI_ROSSBY_ENV_SOURCE) — the two are not guaranteed identical
# and no table should mix them without saying so.
#
# WHAT IT DOES NOT DO
#   It does not activate the venv (callers keep control of PYTHONNOUSERSITE
#   ordering).  Required order — polaris_env.sh comes FIRST, because
#   AI_ROSSBY_VENV is defined there:
#       source polaris_env.sh              # repo paths: MEMBER_ROOT, AI_ROSSBY_VENV
#       source polaris_ai_rossby_env.sh    # this file: interpreter + libs
#       source "${AI_ROSSBY_VENV}/bin/activate"

# Deliberately no `set -e`: this file is sourced, and killing the caller's shell
# on a probe failure would hide the diagnostics printed below.

AI_ROSSBY_ENV_SOURCE="unset"
_ar_env_fail=0

if [ -z "${MEMBER_ROOT:-}" ]; then
    echo "ERROR AI_ROSSBY_ENV_NO_MEMBER_ROOT: source polaris_env.sh BEFORE this file."
    echo "  AI_ROSSBY_VENV is derived from MEMBER_ROOT there; without it the caller"
    echo "  would activate an empty path and fall back to whatever python is first."
    _ar_env_fail=1
fi

module use /soft/modulefiles 2>/dev/null

# ---- attempt 1: the sanctioned module ---------------------------------------
if module load conda >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
    AI_ROSSBY_ENV_SOURCE="module-conda"
    conda activate base
else
    # ---- attempt 2: reconstruct 2025-09-25.lua by hand ----------------------
    AI_ROSSBY_ENV_SOURCE="manual-reconstruction"

    # The modulefile's depends_on lines. Two pins no longer exist; the
    # substitutes are the only installed versions, NOT a preference:
    #   cray-hdf5-parallel/1.14.3.5 -> 1.14.3.9
    #   gcc-native/14.2             -> 14
    if ! module load PrgEnv-gnu craype-x86-milan cudnn/9.13.0 gcc-native/14 \
                     cray-hdf5-parallel/1.14.3.9 >/dev/null 2>&1; then
        echo "ERROR AI_ROSSBY_ENV_MODULES_FAILED: the substitute module set no longer loads either."
        echo "  Tried: PrgEnv-gnu craype-x86-milan cudnn/9.13.0 gcc-native/14 cray-hdf5-parallel/1.14.3.9"
        echo "  The PE has rolled again. Re-derive the set from /soft/modulefiles/conda/2025-09-25.lua"
        echo "  and update this file, polaris_makani_env.sh AND polaris_pbs_notes.md §1 together."
        _ar_env_fail=1
    fi

    # The modulefile's setenv/prepend_path lines, copied value-for-value, minus
    # the NCCL prepend (see the header).
    export CC=/usr/bin/gcc-14
    export CXX=/usr/bin/g++-14
    export TORCH_CUDA_ARCH_LIST=8.0          # A100 = sm80
    export MPICH_GPU_SUPPORT_ENABLED=1
    # The proxy is load-bearing for wandb: compute nodes have NO direct route and
    # reach api.wandb.ai only through it (polaris_env.sh, job 7253810).
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export http_proxy=http://proxy.alcf.anl.gov:3128

    _ar_cuda_home=/soft/compilers/cudatoolkit/cuda-12.9.1
    export CUDA_HOME="${_ar_cuda_home}"
    export CUDA_PATH="${_ar_cuda_home}"
    export CUDA_TOOLKIT_BASE="${_ar_cuda_home}"
    export PATH="${_ar_cuda_home}/bin:${PATH}"
    export LD_LIBRARY_PATH="${_ar_cuda_home}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${_ar_cuda_home}/lib64:${LD_LIBRARY_PATH}"

    # Initialize conda from the install directory. The INSTALL is intact — only
    # the modulefile that points at it is broken. Activating base is parity with
    # the green production run (7368536), not a dependency: the venv sets
    # include-system-site-packages=false, so nothing imports out of base.
    _ar_conda_dir=/soft/applications/conda/2025-09-25/mconda3
    if [ ! -r "${_ar_conda_dir}/etc/profile.d/conda.sh" ]; then
        echo "ERROR AI_ROSSBY_ENV_CONDA_GONE: ${_ar_conda_dir} no longer holds conda.sh."
        echo "  The base conda install itself has moved — no longer a modulefile problem."
        echo "  The venv's interpreter is a symlink into that same tree (pyvenv.cfg home=),"
        echo "  so this is fatal, not cosmetic."
        _ar_env_fail=1
    else
        # shellcheck disable=SC1091
        source "${_ar_conda_dir}/etc/profile.d/conda.sh"
        conda activate base
    fi
fi

export AI_ROSSBY_ENV_SOURCE

ai_rossby_env_report() {
    echo "AI_ROSSBY_ENV_SOURCE   = ${AI_ROSSBY_ENV_SOURCE}"
    echo "python                 = $(command -v python || echo '<none>')"
    echo "CUDA_HOME              = ${CUDA_HOME:-<unset>}"
    echo "AI_ROSSBY_VENV         = ${AI_ROSSBY_VENV:-<unset>}"
}

return ${_ar_env_fail} 2>/dev/null || true
