#!/bin/bash
# Stage-by-stage driver for the G4D-after-63-epochs scoring (prereg.md here). debug and debug-scaling each take ONE
# queued job per user, so the work runs as two sequential lanes, each submitting its next stage on the previous
# stage's PASS token. File-only: no qstat polling.
#   lane D (debug):          J1 ckpt prep -> G4D56 sweep -> score -> G4D63 sweep -> score
#   lane S (debug-scaling):  5-yr run (4 arms) -> protocol score g4d63 -> [J1] G4D36 sweep -> score -> G4D43 sweep -> score
# After each K=56 score passes, that sweep's inference/nwp/*.nc are deleted (operator 2026-11-02) and logged in
# freed_2026-11-02.txt; scores/ stays. A sweep is not submitted with < 80 GB under the project's soft quota.
# Then test_compare_g4d_e63.sh + compare_g4d_e63.py -> results.txt.
# Usage: bash chain_driver.sh >> driver.log 2>&1
set -uo pipefail
D=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/g4d_e63_2026-11-02
E=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval
RUN=f_nosoil_4n_b32_e23_scratch_fp32_dryair_nf4
CK=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling/${RUN}/training_checkpoints
TGM=${E}/window_truth/truth_annual_gm_2015_2047_stride1.npz
J=${D}/jobs.txt
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/eval-protocol/makani_sfno \
    || exit 2

say() { echo "$(date -u +%FT%TZ) $*"; }
has() { [ -f "$1" ] && grep -q "^$2" "$1"; }
# wait_for <log> <token> <minutes>: 0 on token, 1 on a line-start ERROR token / PBS kill / timeout.
wait_for() {
    local log=$1 tok=$2 lim=$(( $3 * 60 )) t=0
    while [ ${t} -lt ${lim} ]; do
        has "${log}" "${tok}" && return 0
        if [ -f "${log}" ] && grep -q '^ERROR \|=>> PBS: job killed' "${log}"; then
            say "ERROR STAGE_FAILED ${log}: $(grep -m3 '^ERROR \|=>> PBS' "${log}" | tr '\n' ' ')"; return 1
        fi
        sleep 60; t=$(( t + 60 ))
    done
    say "ERROR STAGE_TIMEOUT ${log} (no ${tok} in $3 min)"; return 1
}
# sub <qsub args>: prints the job number. Retries for up to 30 min if the per-user queue limit refuses it.
sub() {
    local id n=0 err=${D}/.qsub_err.${BASHPID}
    until id=$(qsub "$@" 2>"${err}"); do
        n=$(( n + 1 ))
        [ ${n} -ge 30 ] && { say "ERROR QSUB_REFUSED $(tr '\n' ' ' < "${err}") :: $*" >&2; return 1; }
        sleep 60
    done
    rm -f "${err}"
    echo "${id%%.*}"
}
# disk_ok: >= 80 GB under the project's 50 TB soft quota (a sweep writes ~58 GB); waits up to 2 h for it.
disk_ok() {
    local t=0 free
    while [ ${t} -lt 7200 ]; do
        free=$(lfs quota -p 11644 /lus/eagle 2>/dev/null | awk '$1=="/lus/eagle" {gsub(/\*/,"",$2); print int(($3-$2)/1048576)}')
        [ -n "${free}" ] && [ "${free}" -ge 80 ] && return 0
        say "disk: ${free:-?} GiB under the soft quota; waiting"
        sleep 600; t=$(( t + 600 ))
    done
    say "ERROR DISK_LOW (${free:-?} GiB under the soft quota after 2 h)"; return 1
}
free_nc() {    # delete one scored sweep's NetCDFs, logging each file
    local arm=$1 s="${E}/$1/scores/k56_summary.csv" f n=0 b=0 sz
    [ -s "${s}" ] || { say "ERROR NO_SCORES ${s}: NetCDFs kept"; return 1; }
    for f in "${E}/${arm}"/inference/nwp/*.nc; do
        [ -f "${f}" ] || continue
        sz=$(stat -c %s "${f}"); echo "${f} ${sz} $(date -u -r "${f}" +%FT%TZ)" >> ${D}/freed_2026-11-02.txt
        rm -f "${f}" && n=$(( n + 1 )) && b=$(( b + sz ))
    done
    say "freed ${arm}: ${n} NetCDFs, $(( b / 1000000000 )) GB"
}
sweep_and_score() {    # <queue> <epoch>
    local q=$1 e=$2 js jk
    disk_ok || return 1
    js=$(sub -q "${q}" -o ${D}/s${e}_sweep.log \
         -v RUN=${RUN},CKPT=${D}/ema_ckpt_G4D${e}.tar,OUT_ROOT=${E}/G4D${e}_K56 polaris/polaris_eval_inference.pbs) || return 1
    echo "s${e}=${js} (${q}, G4D${e} K=56 sweep)" >> ${J}; say "queued s${e}=${js}"
    wait_for ${D}/s${e}_sweep.log EVAL_INFERENCE_OK 720 || return 1
    jk=$(sub -q "${q}" -o ${D}/k${e}_score.log -v RUN=G4D${e}_K56 polaris/polaris_score_k56.pbs) || return 1
    echo "k${e}=${jk} (${q}, G4D${e} K=56 score)" >> ${J}; say "queued k${e}=${jk}"
    wait_for ${D}/k${e}_score.log K56_SCORE_OK 720 || return 1
    free_nc G4D${e}_K56
}

lane_d() {
    local j1
    j1=$(sub -q debug -o ${D}/j1_ckpt.log ${D}/polaris_g4d63_ckpt_prep.pbs) || return 1
    echo "j1=${j1} (debug, EMA pins + extraction)" >> ${J}; say "queued j1=${j1}"
    wait_for ${D}/j1_ckpt.log G4D63_CKPT_OK 720 || return 1
    sweep_and_score debug 56 || return 1
    sweep_and_score debug 63 || return 1
    say "LANE_D_DONE"
}
lane_s() {
    local jc jp arms="G4D36 G4D43 G4D56 G4D63" v="" a
    for a in ${arms}; do
        v="${v},RUNDIR_${a}=${RUN},CKPT_${a}=${CK}/ckpt_mp0_v$(( ${a#G4D} - 1 )).tar,WEIGHTS_${a}=ema"
    done
    jc=$(sub -q debug-scaling -o ${D}/jc_climate.log -v "ARMS=${arms}${v}" polaris/polaris_climate_run.pbs) || return 1
    echo "jc=${jc} (debug-scaling, 5-yr protocol ${arms})" >> ${J}; say "queued jc=${jc}"
    wait_for ${D}/jc_climate.log CLIMATE_RUN_OK 720 || return 1
    jp=$(sub -q debug-scaling -o ${D}/jp_score.log -v DIRS=${jc},TAG=g4d63,TGM=${TGM} \
         polaris/polaris_protocol_score.pbs) || return 1
    echo "jp=${jp} (debug-scaling, protocol score g4d63)" >> ${J}; say "queued jp=${jp}"
    wait_for ${D}/jp_score.log PROTOCOL_SCORE_OK 720 || return 1
    wait_for ${D}/j1_ckpt.log G4D63_CKPT_OK 720 || return 1
    sweep_and_score debug-scaling 36 || return 1
    sweep_and_score debug-scaling 43 || return 1
    say "LANE_S_DONE"
}

echo "prereg sha256 $(sha256sum ${D}/prereg.md | cut -c1-16) at $(date -u +%FT%TZ), before any submission" > ${J}
lane_d & pd=$!
lane_s & ps=$!
rd=0; rs=0
wait ${pd} || rd=$?
wait ${ps} || rs=$?
say "lanes done: D rc=${rd} S rc=${rs}"
for f in j1_ckpt jc_climate jp_score s56_sweep k56_score s63_sweep k63_score s36_sweep k36_score s43_sweep k43_score; do
    [ -f ${D}/${f}.log ] && grep -h '^gate\|_EMA_OK\|^G4D63_CKPT_OK\|^CLIMATE_RUN_OK\|^PROTOCOL_SCORE_OK\|^EVAL_INFERENCE_OK\|^K56_SCORE_OK' \
        ${D}/${f}.log
done
bash ${D}/test_compare_g4d_e63.sh | tail -1
python3 ${D}/compare_g4d_e63.py
[ ${rd} -eq 0 ] && [ ${rs} -eq 0 ] && say "CHAIN_DONE" || { say "ERROR CHAIN_INCOMPLETE"; exit 4; }
