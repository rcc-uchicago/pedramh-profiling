#!/bin/bash
# v3 of the G4D-e63 chain (v2 stopped by hand ~19:30Z 2026-11-02). Operator: "since some of the eval jobs are short
# can we just submit them to preemptable". The remaining stages are mostly independent, so v3 runs them as a
# dependency graph on `preemptable` (project cap 10 concurrent, shared: this driver keeps <= MAXQ of its own in
# flight) instead of two serial lanes on debug / debug-scaling (1 queued job per user each).
#   adopted, already done:  j1, jc (5-yr runs), jp (5-yr scores, g4d63), s56 (G4D56 sweep)
#   adopted, in flight:     s36 = 7808129 (debug-scaling); maps smoke = 7808081 (debug, outside the graph)
#   new, all -r y on preemptable: k56 s63 k63 sA243 k36 s43 k43 s16 c16 -> xkd xks xc (need maps/READY) -> plot
# Pre-emption: jobs are rerunnable, so PBS requeues them. A '=>> PBS: job killed' line triggers ONE `qstat -x -f` for
# that job: requeued (Q/R/H) keeps waiting, anything else fails the stage. No routine qstat polling.
# Usage: bash chain_driver_v3.sh >> driver.log 2>&1
set -uo pipefail
D=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/g4d_e63_2026-11-02
M=${D}/maps
E=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval
RUN=f_nosoil_4n_b32_e23_scratch_fp32_dryair_nf4
X=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling
CK=${X}/${RUN}/training_checkpoints
G16EMA=${E}/g4d16_k56_lagged_2026-10-19/ema_ckpt_G4D16.tar
J=${D}/jobs.txt
Q=preemptable
MAXQ=6
STAGE_MIN=720            # per-stage limit, minutes from submission
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/eval-protocol/makani_sfno \
    || exit 2

say() { echo "$(date -u +%FT%TZ) $*"; }
declare -A LOG TOK DEPS STATE JOB T0 KILLSEEN
ORDER=()
stage() { ORDER+=("$1"); LOG[$1]=$2; TOK[$1]=$3; DEPS[$1]=$4; STATE[$1]=${5:-wait}; JOB[$1]=${6:-}; T0[$1]=$(date +%s); }

#      name   log                          token              deps            [state] [job]
stage  s56    ${D}/s56_sweep.log           EVAL_INFERENCE_OK  ""              ok      7807874
stage  jp     ${D}/jp_score.log            PROTOCOL_SCORE_OK  ""              ok      7807956
stage  s36    ${D}/s_G4D36_sweep.log       EVAL_INFERENCE_OK  ""              sub     7808129
stage  k56    ${D}/k56_score.log           K56_SCORE_OK       "s56"
stage  s63    ${D}/s_G4D63_sweep.log       EVAL_INFERENCE_OK  ""
stage  sA     ${D}/s_A243rr_sweep.log      EVAL_INFERENCE_OK  ""
stage  s43    ${D}/s_G4D43_sweep.log       EVAL_INFERENCE_OK  ""
stage  s16    ${D}/s_G4D16rr_sweep.log     EVAL_INFERENCE_OK  ""
stage  c16    ${D}/jc16_climate.log        CLIMATE_RUN_OK     ""
stage  k63    ${D}/k63_score.log           K56_SCORE_OK       "s63"
stage  k36    ${D}/k36_score.log           K56_SCORE_OK       "s36"
stage  k43    ${D}/k43_score.log           K56_SCORE_OK       "s43"
stage  xkd    ${M}/xk_d.log                MAPS_K56_OK        "k56 k63 sA READY"
stage  xks    ${M}/xk_s.log                MAPS_K56_OK        "k36 k43 s16 READY"
stage  xc     ${M}/xc.log                  MAPS_CLIM_OK       "jp c16 READY"
stage  plot   ${M}/plot.log                MAPS_PLOT_OK       "xkd xks xc"

qsub_args() {    # the stage's qsub arguments (after -q/-r/-o)
    case $1 in
        k56|k63|k36|k43) echo "-v RUN=G4D${1#k}_K56 polaris/polaris_score_k56.pbs" ;;
        s63) echo "-v RUN=${RUN},CKPT=${D}/ema_ckpt_G4D63.tar,OUT_ROOT=${E}/G4D63_K56 polaris/polaris_eval_inference.pbs" ;;
        s43) echo "-v RUN=${RUN},CKPT=${D}/ema_ckpt_G4D43.tar,OUT_ROOT=${E}/G4D43_K56 polaris/polaris_eval_inference.pbs" ;;
        s16) echo "-v RUN=${RUN},CKPT=${G16EMA},OUT_ROOT=${E}/G4D16rr_K56 polaris/polaris_eval_inference.pbs" ;;
        sA)  echo "-v RUN=prod1n_b32_sgdr,CKPT=${X}/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar,OUT_ROOT=${E}/A243rr_K56 polaris/polaris_eval_inference.pbs" ;;
        c16) echo "-v ARMS=G4D16,RUNDIR_G4D16=${RUN},CKPT_G4D16=${CK}/ckpt_mp0_v15.tar,WEIGHTS_G4D16=ema polaris/polaris_climate_run.pbs" ;;
        xc)  echo "-v CDIRS=7807790+${JOB[c16]} ${M}/polaris_maps_clim.pbs" ;;
        plot) echo "${M}/polaris_maps_plot.pbs" ;;
    esac
}
xk_arms() {
    case $1 in
        xkd) echo "G4D56:G4D56_K56:G4D56_K56 G4D63:G4D63_K56:G4D63_K56 A243:A243rr_K56:prod1n_b32_sgdr_K56" ;;
        xks) echo "G4D36:G4D36_K56:G4D36_K56 G4D43:G4D43_K56:G4D43_K56 G4D16:G4D16rr_K56:G4D16_K56" ;;
    esac
}
submit() {
    local s=$1 id err=${D}/.qsub_err.v3
    if [ "${s}" = xkd ] || [ "${s}" = xks ]; then
        id=$(qsub -q ${Q} -r y -o ${LOG[$s]} -v "ARMS=$(xk_arms ${s})" ${M}/polaris_maps_k56.pbs 2>${err})
    else
        # shellcheck disable=SC2046
        id=$(qsub -q ${Q} -r y -o ${LOG[$s]} $(qsub_args ${s}) 2>${err})
    fi || { say "qsub refused for ${s}: $(tr '\n' ' ' < ${err}) (will retry)"; return 1; }
    JOB[$s]=${id%%.*}; STATE[$s]=sub; T0[$s]=$(date +%s)
    echo "${s}=${JOB[$s]} (${Q} -r y)" >> ${J}; say "queued ${s}=${JOB[$s]}"
}
check() {        # sets STATE to ok / fail for a submitted stage
    local s=$1 log=${LOG[$1]} st
    if [ -f "${log}" ] && grep -q "^${TOK[$s]}" "${log}"; then STATE[$s]=ok; say "${s} OK (${TOK[$s]})"; return; fi
    if [ -f "${log}" ] && grep -q '^ERROR ' "${log}"; then
        STATE[$s]=fail; say "ERROR STAGE_FAILED ${s}: $(grep -m3 '^ERROR ' "${log}" | tr '\n' ' ')"; return
    fi
    if [ -f "${log}" ] && grep -q '=>> PBS: job killed' "${log}"; then
        local n; n=$(grep -c '=>> PBS: job killed' "${log}")
        if [ "${KILLSEEN[$s]:-0}" != "${n}" ]; then
            KILLSEEN[$s]=${n}
            st=$(qstat -x -f "${JOB[$s]}" 2>/dev/null | sed -n 's/^ *job_state = //p')
            case "${st}" in
                Q|R|H|W) say "${s} (${JOB[$s]}) was killed and is ${st} again (rerunnable): waiting" ;;
                *) STATE[$s]=fail; say "ERROR STAGE_KILLED ${s} (${JOB[$s]}) job_state='${st}'" ;;
            esac
        fi
    fi
    if [ "${STATE[$s]}" = sub ] && [ $(( $(date +%s) - ${T0[$s]} )) -gt $(( STAGE_MIN * 60 )) ]; then
        STATE[$s]=fail; say "ERROR STAGE_TIMEOUT ${s} (${JOB[$s]}) after ${STAGE_MIN} min"
    fi
}
deps_state() {   # ok | wait | blocked
    local d r=ok
    for d in ${DEPS[$1]}; do
        if [ "${d}" = READY ]; then [ -f ${M}/READY ] || r=wait; continue; fi
        case "${STATE[$d]}" in ok) ;; fail|blocked) echo blocked; return ;; *) r=wait ;; esac
    done
    echo ${r}
}

echo "v3 driver started $(date -u +%FT%TZ): preemptable -r y DAG, MAXQ=${MAXQ}; adopts s56/jp done, s36=7808129" >> ${J}
say "--- v2 stopped by hand (operator: use preemptable); v3 takes over ---"
while :; do
    inflight=0; active=0
    for s in "${ORDER[@]}"; do
        [ "${STATE[$s]}" = sub ] && check ${s}
        [ "${STATE[$s]}" = sub ] && [ "${s}" != s36 ] && inflight=$(( inflight + 1 ))
    done
    for s in "${ORDER[@]}"; do
        [ "${STATE[$s]}" = wait ] || continue
        case "$(deps_state ${s})" in
            blocked) STATE[$s]=blocked; say "${s} BLOCKED (a dependency failed)" ;;
            ok) [ ${inflight} -lt ${MAXQ} ] && submit ${s} && inflight=$(( inflight + 1 )) ;;
        esac
    done
    for s in "${ORDER[@]}"; do case "${STATE[$s]}" in wait|sub) active=1 ;; esac; done
    [ ${active} -eq 0 ] && break
    sleep 60
done
for s in "${ORDER[@]}"; do printf '%s=%s ' "${s}" "${STATE[$s]}"; done; echo
bash ${D}/test_compare_g4d_e63.sh | tail -1
python3 ${D}/compare_g4d_e63.py
bad=0; for s in "${ORDER[@]}"; do [ "${STATE[$s]}" = ok ] || bad=1; done
[ ${bad} -eq 0 ] && say "CHAIN_DONE" || { say "ERROR CHAIN_INCOMPLETE"; exit 4; }
