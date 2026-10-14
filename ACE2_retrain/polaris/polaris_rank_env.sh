#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# PALS -> torchrun-style rank env shim for ACE2 (`fme`) on ALCF Polaris.
#
# fme recognises exactly two launchers
# (ace_exp/fme/core/distributed/torch_distributed.py:29-70):
#   * torchrun-style  — RANK in the environment, then init_method="env://"
#                       (also needs WORLD_SIZE / LOCAL_RANK / MASTER_ADDR /
#                       MASTER_PORT)
#   * srun            — ONLY when FME_USE_SRUN=1, via SLURM_PROCID +
#                       SRUN_DIST_FILE_PATH. Not applicable on PBS, and
#                       CLAUDE.md forbids srun on Polaris anyway.
#
# Polaris uses PALS mpiexec, which exports PMI_RANK / PMI_SIZE / PMI_LOCAL_RANK —
# none of which fme looks at. ⚠ Without this shim there is NO ERROR: `RANK` is
# absent, `TorchDistributed.is_available()` returns False, and
# `Distributed.__init__` silently constructs `NonDistributed`. You get N
# independent world_size=1 trainers, each training the FULL global batch on one
# GPU, each finishing, each writing a plausible step time. The scaling parser's
# WORLD_SIZE_MISMATCH guard exists because of exactly this.
#
# Wrap the python command with this script so each rank translates its own PMI_*
# into the generic names before the interpreter starts:
#
#   mpiexec -n 16 --ppn 4 --hostfile "$PBS_NODEFILE" --cpu-bind depth -d 8 \
#       bash ACE2_retrain/polaris/polaris_rank_env.sh \
#       "$PY" ACE2_retrain/ace2_telemetry.py config.yaml --override ...
#
# ⚠ `bash <shim> "${PY}" ...`, never `"${PY}"` directly under mpiexec: the venv's
# bin/python is a symlink into the base conda, PALS execs the resolved target, so
# pyvenv.cfg is never found and base-conda packages get imported instead. An
# intermediate shell preserves the venv (measured, job 7569872).
#
# MASTER_ADDR / MASTER_PORT must already be exported by the job script — they are
# identical across ranks, so they do not belong here.
#
# This is a deliberate COPY of physicsnemo_ai_rossby/polaris/polaris_rank_env.sh
# rather than a reference to it: that file lives inside a `git subtree` and its
# header documents physicsnemo's DistributedManager, not fme's. The mechanism is
# the same; the failure mode it prevents is different in each tree.

set -uo pipefail

: "${PMI_RANK:?not running under PALS mpiexec — PMI_RANK unset}"
: "${PMI_SIZE:?not running under PALS mpiexec — PMI_SIZE unset}"
: "${MASTER_ADDR:?export MASTER_ADDR in the job script before mpiexec}"
: "${MASTER_PORT:?export MASTER_PORT in the job script before mpiexec}"

export RANK="${PMI_RANK}"
export WORLD_SIZE="${PMI_SIZE}"
# PALS sets PMI_LOCAL_RANK; fall back to rank % gpus-per-node if it's absent.
export LOCAL_RANK="${PMI_LOCAL_RANK:-$(( PMI_RANK % ${GPUS_PER_NODE:-4} ))}"

exec "$@"
