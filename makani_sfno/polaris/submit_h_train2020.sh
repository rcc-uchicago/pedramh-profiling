#!/bin/bash
# Port H production run (operator 2026-09-30): port G's channel set (83-in / 77-out,
# e3sm_alldata_ace2vars.yaml -- the same 24 channels removed) with the TRAIN YEARS
# changed to 2020-2044 (the view built by polaris_pack_alldata_trainview.pbs, its own
# stats; valid 2045-2047 and test 2048-2049 unchanged). G vs H isolates the train years.
#   bash polaris/submit_h_train2020.sh <QUEUE> <NODES 1|2|4> <EPOCHS> <WALLTIME hh:mm:ss> [scratch|warm]
# Same recipe, knobs (DEPEND, SPARE, DRY_RUN, G_WARM_CKPT) and checks as
# submit_g_ace2vars.sh, which it calls with SPLIT=2020. Runs are h_ace2vars_train2020_*.
# 36,500 train samples -> 1140 updates/epoch at batch 32 (G and A: 1368).
# warm: A sliced to 83/77 saw the excluded 2015-2019 years and learned under the
# production stats -- scratch is the clean test of the train years.
set -euo pipefail
if [ -n "${SPLIT:-}" ] && [ "${SPLIT}" != "2020" ]; then
    echo "ERROR SPLIT=${SPLIT} set in the environment; port H is SPLIT=2020 by definition"; exit 2
fi
SPLIT=2020 exec bash "$(dirname "${BASH_SOURCE[0]}")/submit_g_ace2vars.sh" "$@"
