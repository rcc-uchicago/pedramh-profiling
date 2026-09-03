#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Environment bootstrap for ACE2 (ai2cm `fme`) on Polaris that does NOT depend on
# the broken `conda` modulefile.  SOURCE it; do not execute it.
#
#     source "<repo>/ACE2_retrain/polaris/polaris_ace2_env.sh" || exit 2
#
# WHY THIS FILE EXISTS
#   `module load conda` has been BROKEN cluster-side since the 2026-08 Cray PE
#   roll: /soft/modulefiles/conda/2025-09-25.lua declares
#       depends_on("cray-hdf5-parallel/1.14.3.5")   -- only 1.14.3.9 installed
#       depends_on("gcc-native/14.2")               -- only 14 installed
#   so Lmod aborts and `conda` never lands.  -> polaris_pbs_notes.md §1 BLOCKER
#
#   This is the third sibling of polaris_makani_env.sh / polaris_ai_rossby_env.sh.
#   It is closest to the ai-rossby one, and the DELTAS ARE DELIBERATE:
#
#     * NO torch-2.8 DT_NEEDED repair and NO h5py overlay.  Those exist because
#       makani runs the BASE CONDA's torch and imports the base conda's h5py.
#       ACE2 runs its own venv (pyvenv.cfg include-system-site-packages=false),
#       which ships the whole nvidia stack plus its own h5py/netCDF4 wheels.
#     * NO /soft/libraries/nccl prepend, same reason as ai-rossby: it would put
#       the system NCCL ahead of the one this venv's torch was built against and
#       silently change the communication library under a scaling measurement.
#     * IT DOES pin the fabric stack (below).  ai-rossby's launcher does that
#       inline in its .pbs; ACE2 has two launchers (1-node and multi-node) that
#       must agree bit-for-bit on the transport or their rows are not comparable,
#       so the pin lives here, once.
#
# IT PREFERS THE OFFICIAL PATH.  `module load conda` is attempted FIRST; the
# manual reconstruction is the fallback.  It always reports which of the two ran
# (ACE2_ENV_SOURCE) -- the two are not guaranteed identical and no table should
# mix them without saying so.
#
# ORDER MATTERS.  polaris_env.sh comes FIRST, because MEMBER_ROOT is defined
# there and ACE2_VENV is derived from it:
#     source polaris_env.sh              # repo paths: MEMBER_ROOT, TMPDIR, wandb
#     source polaris_ace2_env.sh         # this file: interpreter + libs + fabric
#     source "${ACE2_VENV}/bin/activate"
#
# KNOBS
#   POLARIS_ACE2_VENV=<dir>   force a specific venv (default $MEMBER_ROOT/conda-envs/fme-venv)
#   OFI_PLUGIN=<dir>          another aws-ofi-nccl build -- ITS OWN CSV, never the ladder's
#   OFI_LIBFABRIC=<dir>       another libfabric
#   PROGRESS_MODEL=           set empty to leave OFI_NCCL_PROGRESS_MODEL unset

# Deliberately no `set -e`: this file is sourced, and killing the caller's shell
# on a probe failure would hide the diagnostics printed below.

ACE2_ENV_SOURCE="unset"
_ace2_env_fail=0

if [ -z "${MEMBER_ROOT:-}" ]; then
    echo "ERROR ACE2_ENV_NO_MEMBER_ROOT: source polaris_env.sh BEFORE this file."
    echo "  ACE2_VENV is derived from MEMBER_ROOT there; without it the caller"
    echo "  would activate an empty path and fall back to whatever python is first."
    _ace2_env_fail=1
fi

export ACE2_VENV="${POLARIS_ACE2_VENV:-${MEMBER_ROOT:-/nonexistent}/conda-envs/fme-venv}"

module use /soft/modulefiles 2>/dev/null

# ---- attempt 1: the sanctioned module ---------------------------------------
if module load conda >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
    ACE2_ENV_SOURCE="module-conda"
    conda activate base
else
    # ---- attempt 2: reconstruct 2025-09-25.lua by hand ----------------------
    ACE2_ENV_SOURCE="manual-reconstruction"

    # The modulefile's depends_on lines. Two pins no longer exist; the
    # substitutes are the only installed versions, NOT a preference:
    #   cray-hdf5-parallel/1.14.3.5 -> 1.14.3.9
    #   gcc-native/14.2             -> 14
    if ! module load PrgEnv-gnu craype-x86-milan cudnn/9.13.0 gcc-native/14 \
                     cray-hdf5-parallel/1.14.3.9 >/dev/null 2>&1; then
        echo "ERROR ACE2_ENV_MODULES_FAILED: the substitute module set no longer loads either."
        echo "  Tried: PrgEnv-gnu craype-x86-milan cudnn/9.13.0 gcc-native/14 cray-hdf5-parallel/1.14.3.9"
        echo "  The PE has rolled again. Re-derive the set from /soft/modulefiles/conda/2025-09-25.lua"
        echo "  and update this file, polaris_ai_rossby_env.sh, polaris_makani_env.sh AND"
        echo "  polaris_pbs_notes.md §1 together."
        _ace2_env_fail=1
    fi

    export CC=/usr/bin/gcc-14
    export CXX=/usr/bin/g++-14
    export TORCH_CUDA_ARCH_LIST=8.0          # A100 = sm80
    export MPICH_GPU_SUPPORT_ENABLED=1
    # The proxy is load-bearing for wandb: compute nodes have NO direct route and
    # reach api.wandb.ai only through it (polaris_env.sh, job 7253810).
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export http_proxy=http://proxy.alcf.anl.gov:3128

    _ace2_cuda_home=/soft/compilers/cudatoolkit/cuda-12.9.1
    export CUDA_HOME="${_ace2_cuda_home}"
    export CUDA_PATH="${_ace2_cuda_home}"
    export CUDA_TOOLKIT_BASE="${_ace2_cuda_home}"
    export PATH="${_ace2_cuda_home}/bin:${PATH}"
    export LD_LIBRARY_PATH="${_ace2_cuda_home}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${_ace2_cuda_home}/lib64:${LD_LIBRARY_PATH}"

    # Initialize conda from the install directory. The INSTALL is intact -- only
    # the modulefile that points at it is broken. The venv's interpreter is a
    # symlink into that same tree (pyvenv.cfg home=), so this is load-bearing.
    _ace2_conda_dir=/soft/applications/conda/2025-09-25/mconda3
    if [ ! -r "${_ace2_conda_dir}/etc/profile.d/conda.sh" ]; then
        echo "ERROR ACE2_ENV_CONDA_GONE: ${_ace2_conda_dir} no longer holds conda.sh."
        echo "  The base conda install itself has moved -- no longer a modulefile problem."
        _ace2_env_fail=1
    else
        # shellcheck disable=SC1091
        source "${_ace2_conda_dir}/etc/profile.d/conda.sh"
        conda activate base
    fi
fi

export ACE2_ENV_SOURCE

# ---- the fabric stack -------------------------------------------------------
# Self-built aws-ofi-nccl v1.21.1 + cray libfabric 2.3.1 + OFI_NCCL_PROGRESS_MODEL
# =AUTO.  Every /soft plugin build >= v1.9 fails fi_domain with ENOSYS against
# libfabric 2.3.1 -- including ALCF's own 2025-09 rebuilds -- because the CXI
# provider requires auto progress and newer plugin defaults do not request it
# (jobs 7563894 on NCCL 2.28.3 and 7568618 on 2.27.5: same answer, i.e. the pin is
# provider-side and NOT NCCL-version specific, which is why it is inherited here
# rather than re-measured).
#
# ⚠ A NONEXISTENT PATH ON LD_LIBRARY_PATH IS IGNORED, NOT HONOURED.  Job 7553811
# ran against a pin whose libfabric directory no longer existed, silently used a
# different transport, and died SIGSEGV with the blame landing elsewhere.  Hence
# the hard exit.
#
# Set even for 1-NODE runs.  Intra-node NCCL never leaves NVLink so the plugin is
# not exercised there, but leaving it unset on the 1-node arm and set on the rest
# would make the baseline a different configuration from the ladder it anchors.
ACE2_OFI_PLUGIN="${OFI_PLUGIN:-${MEMBER_ROOT:-/nonexistent}/sw/aws-ofi-nccl-1.21.1/lib}"
ACE2_OFI_LIBFABRIC="${OFI_LIBFABRIC:-/opt/cray/libfabric/2.3.1/lib64}"
for _d in "${ACE2_OFI_PLUGIN}" "${ACE2_OFI_LIBFABRIC}" /soft/libraries/hwloc/lib; do
    if [ ! -d "${_d}" ]; then
        echo "ERROR ACE2_OFI_PATH_MISSING: ${_d}"
        echo "  A fabric pin that does not exist is ignored, not honoured -- the job"
        echo "  would fall back to another transport and the row would measure"
        echo "  something else. Rebuild the plugin or pass -v OFI_PLUGIN=<dir>."
        _ace2_env_fail=1
    fi
done
if [ "${_ace2_env_fail}" -eq 0 ]; then
    # libfabric first so it outranks any system copy; hwloc because libnccl-net.so
    # needs libhwloc.so.0 and nothing on the default path provides it.
    export LD_LIBRARY_PATH="${ACE2_OFI_LIBFABRIC}:${ACE2_OFI_PLUGIN}:/soft/libraries/hwloc/lib:${LD_LIBRARY_PATH}"
    PROGRESS_MODEL="${PROGRESS_MODEL-AUTO}"
    [ -n "${PROGRESS_MODEL}" ] && export OFI_NCCL_PROGRESS_MODEL="${PROGRESS_MODEL}"
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
    export NCCL_CROSS_NIC=1
    export NCCL_NET="AWS Libfabric"
fi
export ACE2_OFI_PLUGIN ACE2_OFI_LIBFABRIC

ace2_env_report() {
    echo "ACE2_ENV_SOURCE   = ${ACE2_ENV_SOURCE}"
    echo "python            = $(command -v python || echo '<none>')"
    echo "CUDA_HOME         = ${CUDA_HOME:-<unset>}"
    echo "ACE2_VENV         = ${ACE2_VENV:-<unset>}"
    echo "fabric plugin     = ${ACE2_OFI_PLUGIN}"
    echo "fabric libfabric  = ${ACE2_OFI_LIBFABRIC}"
    echo "OFI_NCCL_PROGRESS_MODEL = ${OFI_NCCL_PROGRESS_MODEL:-<unset>}"
}

return ${_ace2_env_fail} 2>/dev/null || true
