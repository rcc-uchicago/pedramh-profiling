#!/bin/bash
# Re-roll the bf16-trained candidates in fp32 (docs/2026-10-13_fp32_reroll_prereg.md, docs/makani-knowledge):
#   batch a  debug, 1 node:          FSEF23-28
#   batch b  debug-scaling, 2 nodes: FSEF29-32 FDEF28 FDEF32  |  node 2: BF22 BF24 BF22fix BF24fix DRYAIRF24
# Every arm rolls with --amp off; checkpoints and flags are the original jobs' (prereg table).
#   bash polaris/submit_fp32_reroll.sh          # from makani_sfno/
#   DRY=1 bash polaris/submit_fp32_reroll.sh    # print the qsub lines only
set -euo pipefail
cd "$(dirname "$0")/.."
E=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling
X=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/fp32_reroll_2026-10-13
FS=f_nosoil_2n_b32_e23_scratch; FD=fsF_anneal_nf4_b16_rtmax100_scratch; B=nf4_prod_b16_r1; DA=fs_anneal_dryair_nf4_b16_rb01
if compgen -G "${X}/done_*" > /dev/null || [ -d "${X}/score.lock" ]; then
    echo "ERROR ROLLBATCH_STALE_MARKERS in ${X}; move them aside first"; exit 2
fi
v() {   # <arm> <run> <ckpt file> <weights raw|ema> <flags>
    local ck="${E}/$2/training_checkpoints/$3"
    [ -f "${ck}" ] || { echo "ERROR CKPT_MISSING ${ck}" >&2; exit 3; }
    printf ',RUNDIR_%s=%s,CKPT_%s=%s,WEIGHTS_%s=%s,FLAGS_%s=%s' "$1" "$2" "$1" "${ck}" "$1" "$4" "$1" "$5"
}
A_ARMS=""; A_V=""; B_ARMS=""; B2_ARMS=""; B_V=""
for e in 23 24 25 26 27 28; do A_ARMS="${A_ARMS:+${A_ARMS} }FSEF${e}"; A_V="${A_V}$(v "FSEF${e}" "${FS}" "ckpt_mp0_v$((e - 1)).tar" ema '--amp off')"; done
for e in 29 30 31 32; do B_ARMS="${B_ARMS:+${B_ARMS} }FSEF${e}"; B_V="${B_V}$(v "FSEF${e}" "${FS}" "ckpt_mp0_v$((e - 1)).tar" ema '--amp off')"; done
for e in 28 32; do B_ARMS="${B_ARMS} FDEF${e}"; B_V="${B_V}$(v "FDEF${e}" "${FD}" "ckpt_mp0_v$((e - 1)).tar" ema '--amp off')"; done
for e in 22 24; do
    B2_ARMS="${B2_ARMS:+${B2_ARMS} }BF${e} BF${e}fix"
    B_V="${B_V}$(v "BF${e}" "${B}" "ckpt_mp0_e${e}_stable.tar" raw '--amp off')"
    B_V="${B_V}$(v "BF${e}fix" "${B}" "ckpt_mp0_e${e}_stable.tar" raw '--amp off --dry-air-fix on')"
done
B2_ARMS="${B2_ARMS} DRYAIRF24"; B_V="${B_V}$(v DRYAIRF24 "${DA}" ckpt_mp0_v23.tar raw '--amp off')"
COMMON="EXPECT=a b,XDIR=${X},TAG=f32a,ANALYSIS=fp32_rerank.py"
SHA="$(git rev-parse --short=12 HEAD)"
sub() {   # <batch> <queue> <nodes> <vars>
    if [ "${DRY:-0}" = "1" ]; then
        echo "qsub -q $2 -l select=$3:system=polaris -v \"BATCH=$1,${COMMON},$4\" polaris/polaris_rollout_batch.pbs"
    else
        local id; id=$(qsub -q "$2" -l "select=$3:system=polaris" -v "BATCH=$1,${COMMON},$4" polaris/polaris_rollout_batch.pbs)
        echo "ROLLBATCH_QUEUED batch=$1 queue=$2 nodes=$3 job=${id%%.*} sha=${SHA}"
    fi
}
sub a debug 1 "ARMS=${A_ARMS}${A_V}"
sub b debug-scaling 2 "ARMS=${B_ARMS},ARMS2=${B2_ARMS}${B_V}"
