#!/bin/bash
# Submit the two new cells of the 2 x 2 precision cross (docs/2026-10-13_precision_cross_prereg.md):
#   bf16roll (debug):         FSEQ15 (gate: F-scratch EMA e15, --amp bf16) + G1EB15-19 (G1 EMA, --amp bf16)
#   fp32roll (debug-scaling): FSEF15-19 (F-scratch EMA, --amp off)
# Version N holds epoch N+1 (EMA_EXTRACT map, polaris_eval_gates.pbs 7731785).
#   bash polaris/submit_precision_cross.sh            # from makani_sfno/
#   DRY=1 bash polaris/submit_precision_cross.sh      # print the qsub lines, submit nothing
set -euo pipefail
cd "$(dirname "$0")/.."
E=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling
X=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/precision_cross_2026-10-13
FS=f_nosoil_2n_b32_e23_scratch          # bf16-trained
G1=f_nosoil_2n_b32_e23_scratch_fp32     # fp32-trained
if compgen -G "${X}/done_*" > /dev/null || [ -d "${X}/score.lock" ]; then
    echo "ERROR PXC_STALE_MARKERS in ${X} (a previous attempt); move them aside first"; exit 2
fi
arm_vars() {   # <arm> <run> <epoch> <flags>
    local ck="${E}/$2/training_checkpoints/ckpt_mp0_v$(($3 - 1)).tar"
    [ -f "${ck}" ] || { echo "ERROR CKPT_MISSING ${ck}" >&2; exit 3; }
    printf ',RUNDIR_%s=%s,CKPT_%s=%s,WEIGHTS_%s=ema,FLAGS_%s=%s' "$1" "$2" "$1" "${ck}" "$1" "$1" "$4"
}
B_ARMS="FSEQ15"; B_V="$(arm_vars FSEQ15 "${FS}" 15 '--amp bf16')"
F_ARMS=""; F_V=""
for e in 15 16 17 18 19; do
    B_ARMS="${B_ARMS} G1EB${e}"; B_V="${B_V}$(arm_vars "G1EB${e}" "${G1}" "${e}" '--amp bf16')"
    F_ARMS="${F_ARMS:+${F_ARMS} }FSEF${e}"; F_V="${F_V}$(arm_vars "FSEF${e}" "${FS}" "${e}" '--amp off')"
done
SHA="$(git rev-parse --short=12 HEAD)"
for cell in bf16roll fp32roll; do
    if [ "${cell}" = bf16roll ]; then q=debug; v="CELL=${cell},ARMS=${B_ARMS}${B_V}"
    else q=debug-scaling; v="CELL=${cell},ARMS=${F_ARMS}${F_V}"; fi
    if [ "${DRY:-0}" = "1" ]; then
        echo "qsub -q ${q} -v \"${v}\" polaris/polaris_precision_cross.pbs"
    else
        id=$(qsub -q "${q}" -v "${v}" polaris/polaris_precision_cross.pbs)
        echo "PXC_QUEUED cell=${cell} queue=${q} job=${id%%.*} sha=${SHA}"
    fi
done
