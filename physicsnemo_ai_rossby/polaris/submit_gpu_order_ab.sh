#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Submit a 2-node arm sequence, one job at a time, as debug-scaling slots free.
#
# ARM 1 IS A CONTROL, AND IT COMES FIRST ON PURPOSE.
# Three consecutive 2-node jobs failed on 2026-08-31 (7576778 hung at the DDP
# init broadcast; 7576872 died at init_process_group with ncclUnhandledCudaError
# on known-sick x3111c0s37b1n0; 7576946 got past the trainer banner -- world_size
# =8, all ranks -- then wedged in the first training collective on one node's
# ranks only). The same config was green on the ladder and ran 3 h at 48 nodes.
#
# Two candidate explanations, and they must not be confounded:
#   (a) the machine/fabric is unstable today;
#   (b) the gradient-clipping-order fix + grad-norm diagnostic landed today.
#
# (b) looks unlikely for 7576946 -- the wedge is in a NCCL collective, and
# clip_and_measure_grads issues none; it runs only AFTER backward() returns, so a
# hang in DDP's backward all-reduce means it never executed. That is an argument,
# not a measurement. ARM 1 turns it into one: default knobs, GRAD_NORM_LOG=0, so
# _grad_stats is None and clip_and_measure_grads returns at its first line --
# behaviourally identical to the last known-green ladder config.
#
#   arm 1 green  -> machine is fine, today's code is implicated, stop and bisect
#   arm 1 hangs  -> machine, and no conclusion about the diagnostic is available
#
# ARM 2 is the never-measured NUMA pairing. ALCF's own GPU-MPI example binds each
# rank to the GPU on its NUMA node in REVERSE device order --
# gpu = num_gpus-1 - PMI_LOCAL_RANK % num_gpus, commented "need to assign GPUs in
# reverse order due to topology" (docs.alcf.anl.gov/running-jobs/
# example-job-scripts) -- and polaris_pbs_notes.md §1 measured the same map
# independently (dev0->NUMA3 ... dev3->NUMA0, job 7531456). The launcher carries
# the knob but defaults to `forward`, and a CSV scan on 2026-08-31 found ZERO
# gpu_order=reverse rows: the whole ladder, the knob matrix, both 48-node
# production runs, the LR sweep and the HP sweep were all measured with every
# rank bound to the NUMA domain FURTHEST from its own GPU.
#
# Worth testing because the penalty has the wrong shape for a comms-scaling
# problem: the cliff is at 1->2 nodes (698 -> 2083 ms, 3x) and then nearly flat
# to 48 (2083 -> 3391 across a 24x node increase). At 1 node NCCL is pure NVLink
# and runs no host-side proxy threads at all; at 2+ those threads drive the NICs
# from the rank's bound cores. The launcher already records that starving that
# same progress engine costs most of the bandwidth (4.08 GB/s without
# --cpu-bind). Numerically inert -- it changes which physical GPU executes which
# rank, not what is computed, so unlike the clip/betas arms it needs no science
# sign-off.
#
# Screening only: the 2-node forward rows already span 2024-2294 ms (~15%,
# CHANGELOG §4.4c), so one run per arm resolves a LARGE effect and nothing
# subtler. A null result needs reps before it means anything.
#
# qsub only, never qstat in a loop (CLAUDE.md, login-node process cap): a
# rejected submit IS the "slot busy" signal, so the retry doubles as the
# one-job-per-user sequencer.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# name:knobs -- order matters, arm 1 gates the interpretation of arm 2
ARMS=(
    "control-forward:TARGET_NODES=2,GPU_ORDER=forward,STEPS=60,GRAD_NORM_LOG=0"
    "reverse:TARGET_NODES=2,GPU_ORDER=reverse,STEPS=60,GRAD_NORM_LOG=0"
)
DEADLINE=$(( $(date +%s) + 14400 ))   # give up after 4 h

for arm in "${ARMS[@]}"; do
    name="${arm%%:*}"
    knobs="${arm#*:}"
    submitted=""
    while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
        out=$(qsub -q debug-scaling -l select=3:system=polaris -l walltime=00:50:00 \
                  -v "${knobs}" \
                  polaris/polaris_ai_rossby_multinode_scaling.pbs 2>&1)
        if [ $? -eq 0 ]; then
            submitted="${out}"
            echo "$(date -u +%H:%M:%S) GPU_ORDER_AB_SUBMITTED arm=${name} job=${out}"
            break
        fi
        echo "$(date -u +%H:%M:%S) arm=${name} slot busy: ${out}"
        sleep 300
    done
    if [ -z "${submitted}" ]; then
        echo "ERROR GPU_ORDER_AB_TIMEOUT arm=${name}: no slot within the deadline"
        exit 1
    fi
    # The next loop's first qsub is rejected until this job leaves the queue,
    # which is exactly the barrier we want -- no qstat polling required.
    sleep 60
done

echo "GPU_ORDER_AB_ALL_SUBMITTED"
