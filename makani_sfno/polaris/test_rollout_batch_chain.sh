#!/bin/bash
# Login-node test of polaris_rollout_batch.pbs's chaining (DRY_RUN=1: no GPUs, no mpiexec, no env bootstrap).
# Batch a (1 node) finishes first and must wait; batch b (2 nodes: ARMS + ARMS2) must list both run dirs,
# score once and run the analysis; a late duplicate must not re-score.  PASS = TEST_ROLLBATCH_CHAIN_OK
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "${T}"' EXIT
mkdir -p "${T}/protocol_scores"
python3 - "${HERE}" "${T}/protocol_scores" <<'PY'
import random, sys
sys.path.insert(0, sys.argv[1])
import test_fp32_rerank as t
rng = random.Random(0)
t.write_tag(sys.argv[2], "g1dry_c", {"G1D39": dict(A=1.5, drift=14)}, rng)
t.write_tag(sys.argv[2], "f32a", {"FSEF30": dict(A=1.0, drift=9), "BF22": dict(A=1.7, dps=-7.0)}, rng)
PY
run() {  # <batch> <jobid> [arms2]
    PBS_O_WORKDIR="${HERE}/.." PBS_JOBID="$2.dry" BATCH="$1" EXPECT="a b" TAG=f32a XDIR="${T}/x" \
        ANALYSIS=fp32_rerank.py ARMS="X1 X2" ARMS2="${3:-}" DRY_RUN=1 EVAL_ROOT="${T}" \
        bash "${HERE}/polaris_rollout_batch.pbs" > "${T}/$2.out" 2>&1
}
ok=1
run a 111;        grep -q "^ROLLBATCH_WAITING" "${T}/111.out" || { echo "FAIL first batch did not wait"; cat "${T}/111.out"; ok=0; }
run b 222 "Y1 Y2"; grep -q "^FP32_RERANK_OK winner=FSEF30" "${T}/222.out" || { echo "FAIL last batch did not score/analyse"; cat "${T}/222.out"; ok=0; }
grep -q "^222 222_n2$" "${T}/x/done_b" || { echo "FAIL done_b does not list both run dirs"; ok=0; }
grep -q "DIRS='111 222 222_n2 '" "${T}/222.out" || { echo "FAIL scoring DIRS wrong"; ok=0; }
run b 333 "Y1";   grep -q "^ROLLBATCH_WAITING" "${T}/333.out" || { echo "FAIL late job scored twice"; ok=0; }
[ "${ok}" = 1 ] && echo "TEST_ROLLBATCH_CHAIN_OK" || echo "ERROR TEST_ROLLBATCH_CHAIN_FAILED"
