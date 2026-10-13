#!/bin/bash
# Login-node test of polaris_precision_cross.pbs's chaining (DRY_RUN=1: no GPUs, no env bootstrap).
# Simulates: fp32roll finishes first (must wait), bf16roll second (must score + analyse once), a third
# late job (must not score again). Synthetic score files come from test_precision_cross_analysis.py.
# PASS = TEST_PXC_CHAIN_OK
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "${T}"' EXIT
mkdir -p "${T}/protocol_scores"
python3 - "${HERE}" "${T}/protocol_scores" <<'PY'
import os, shutil, sys, tempfile
sys.path.insert(0, sys.argv[1])
import test_precision_cross_analysis as t
with tempfile.TemporaryDirectory() as tmp:
    ex, nw, _gate, (bx, bn) = t.build(tmp, r_eff=-1.5, t_eff=0.0)
    for src, dst in ((ex, "g1fs_channels.csv"), (nw, "pxc_channels.csv"), (bx, "g1fs_budget.txt"), (bn, "pxc_budget.txt")):
        shutil.copy(src, os.path.join(sys.argv[2], dst))
PY
run() {  # <cell> <jobid>
    PBS_O_WORKDIR="${HERE}/.." PBS_JOBID="$2.dry" CELL="$1" DRY_RUN=1 EVAL_ROOT="${T}" PXC_DIR="${T}/pxc" \
        bash "${HERE}/polaris_precision_cross.pbs" > "${T}/$2.out" 2>&1
}
ok=1
run fp32roll 111; grep -q "^PXC_WAITING" "${T}/111.out" || { echo "FAIL first finisher did not wait"; ok=0; }
run bf16roll 222; grep -q "^PRECISION_CROSS_RESULT outcome=1" "${T}/222.out" || { echo "FAIL second finisher did not score"; cat "${T}/222.out"; ok=0; }
run bf16roll 333; grep -q "^PXC_WAITING" "${T}/333.out" || { echo "FAIL late job scored twice"; ok=0; }
[ -f "${T}/pxc/results.txt" ] || { echo "FAIL results.txt missing"; ok=0; }
[ "${ok}" = 1 ] && echo "TEST_PXC_CHAIN_OK" || echo "ERROR TEST_PXC_CHAIN_FAILED"
