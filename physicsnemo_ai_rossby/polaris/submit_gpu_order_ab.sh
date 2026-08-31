#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Queue the GPU_ORDER=reverse arm of the 2-node A/B as soon as a debug-scaling
# slot frees.
#
# WHY THIS ARM. ALCF's own GPU-MPI example pins each rank to the GPU on its NUMA
# node, in REVERSE device order:
#     gpu=$((num_gpus - 1 - PMI_LOCAL_RANK % num_gpus))   # "due to topology"
# (docs.alcf.anl.gov/running-jobs/example-job-scripts -> GPU MPI examples), and
# polaris_pbs_notes.md §1 measured the same map independently (dev0->NUMA3 ...
# dev3->NUMA0, job 7531456). Our launcher carries the knob but defaults to
# `forward`, and a CSV scan on 2026-08-31 found ZERO gpu_order=reverse rows: the
# whole ladder, the knob matrix, both production runs, the LR sweep and the HP
# sweep were all measured with every rank bound to the NUMA domain FURTHEST from
# its own GPU.
#
# The prediction being tested: the weak-scaling cliff is at 1->2 nodes (698 ->
# 2083 ms, 3x) and then nearly flat out to 48 nodes (2083 -> 3391 over a 24x
# node increase). A fixed cost that switches on the moment the network path
# engages, rather than one that grows with scale, is what a misbound NCCL
# host-side proxy thread looks like -- at 1 node NCCL is pure NVLink and runs no
# proxy threads at all.
#
# Numerically inert: this changes which physical GPU executes which rank, not
# what is computed. No science sign-off needed (unlike the clip/betas arms).
#
# Screen first, replicate later. The 2-node forward rows span 2024-2294 ms
# (~15%, CHANGELOG §4.4c), so ONE run only resolves a large effect -- which is
# exactly what this hypothesis predicts. A null result here needs reps before it
# means anything, and this script does not provide them.
#
# qsub only, never qstat in a loop (CLAUDE.md, login-node process cap): a
# rejected submit IS the "slot busy" signal.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

REVERSE_ARGS="TARGET_NODES=2,GPU_ORDER=reverse,STEPS=60,GRAD_NORM_LOG=0"
DEADLINE=$(( $(date +%s) + 10800 ))   # give up after 3 h

while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    out=$(qsub -q debug-scaling -l select=3:system=polaris -l walltime=00:50:00 \
              -v "${REVERSE_ARGS}" \
              polaris/polaris_ai_rossby_multinode_scaling.pbs 2>&1)
    rc=$?
    if [ ${rc} -eq 0 ]; then
        echo "GPU_ORDER_AB_SUBMITTED ${out}"
        exit 0
    fi
    # Expected while 7576946 holds the one-job-per-user debug-scaling slot.
    echo "$(date -u +%H:%M:%S) slot busy (rc=${rc}): ${out}"
    sleep 300
done

echo "ERROR GPU_ORDER_AB_NOT_SUBMITTED: no debug-scaling slot freed within 3 h"
exit 1
