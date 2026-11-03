#!/bin/bash
# v2 of chain_driver.sh (stopped 2026-11-02 ~18:20Z after it had queued j1, jc, s56): same scoring stages, plus the
# map figures the operator asked for (RMSE maps at 6/30/60/270 h, 5-yr bias maps, with confidence intervals).
# Sweeps are no longer deleted right after scoring: the map extraction (maps/) reads them first.
#   lane D (debug):          [s56 queued by v1] -> k56 -> s63 -> k63 -> A243 re-roll sweep -> XK(G4D56,G4D63,A243rr)
#   lane S (debug-scaling):  [jc running, v1]  -> jp -> s36 -> k36 -> s43 -> k43 -> G4D16 re-roll sweep
#                            -> G4D16 5-yr re-roll -> XK(G4D36,G4D43,G4D16rr) -> XC(jc + G4D16 5-yr)
#   then (debug) plots -> compare_g4d_e63.py -> results.txt
# A243/G4D16 are re-rolled because their K=56 and 5-yr NetCDFs were deleted on 11-02; XK/XC compare the re-rolls
# with the original scores (reproducibility check). XK/XC/plot wait for maps/READY (written once the map code passes
# its tests on real files). Each X job deletes its inputs' NetCDFs only after its own PASS token.
# Usage: bash chain_driver_v2.sh >> driver.log 2>&1
set -uo pipefail
D=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/g4d_e63_2026-11-02
M=${D}/maps
E=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval
RUN=f_nosoil_4n_b32_e23_scratch_fp32_dryair_nf4
CK=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling/${RUN}/training_checkpoints
G16EMA=${E}/g4d16_k56_lagged_2026-10-19/ema_ckpt_G4D16.tar
TGM=${E}/window_truth/truth_annual_gm_2015_2047_stride1.npz
J=${D}/jobs.txt
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/eval-protocol/makani_sfno \
    || exit 2

say() { echo "$(date -u +%FT%TZ) $*"; }
has() { [ -f "$1" ] && grep -q "^$2" "$1"; }
wait_for() {    # <log> <token> <minutes>: 0 on token, 1 on a line-start ERROR / PBS kill / timeout
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
wait_file() {   # <file> <minutes>
    local t=0; while [ ${t} -lt $(( $2 * 60 )) ]; do [ -f "$1" ] && return 0; sleep 60; t=$(( t + 60 )); done
    say "ERROR WAIT_FILE_TIMEOUT $1"; return 1
}
sub() {         # prints the job number; retries 30 min on a per-user queue limit
    local id n=0 err=${D}/.qsub_err.${BASHPID}
    until id=$(qsub "$@" 2>"${err}"); do
        n=$(( n + 1 ))
        [ ${n} -ge 30 ] && { say "ERROR QSUB_REFUSED $(tr '\n' ' ' < "${err}") :: $*" >&2; return 1; }
        sleep 60
    done
    rm -f "${err}"; echo "${id%%.*}"
}
rec() { echo "$1" >> ${J}; say "queued $1"; }
sweep() {       # <queue> <tag> <ckpt> <out_root> [run]: K=56 sweep, waits for its PASS
    local q=$1 tag=$2 ck=$3 out=$4 run=${5:-${RUN}} js
    js=$(sub -q "${q}" -o ${D}/s_${tag}_sweep.log -v RUN=${run},CKPT=${ck},OUT_ROOT=${out} \
         polaris/polaris_eval_inference.pbs) || return 1
    rec "s_${tag}=${js} (${q}, ${tag} K=56 sweep)"
    wait_for ${D}/s_${tag}_sweep.log EVAL_INFERENCE_OK 720
}
score() {       # <queue> <e>
    local jk
    jk=$(sub -q "$1" -o ${D}/k$2_score.log -v RUN=G4D$2_K56 polaris/polaris_score_k56.pbs) || return 1
    rec "k$2=${jk} ($1, G4D$2 K=56 score)"
    wait_for ${D}/k$2_score.log K56_SCORE_OK 720
}
xk() {          # <queue> <name> <ARM:SWEEP:REFSCORES ...>
    local q=$1 name=$2 jx; shift 2
    wait_file ${M}/READY 1440 || return 1
    jx=$(sub -q "${q}" -o ${M}/xk_${name}.log -v "ARMS=$*" ${M}/polaris_maps_k56.pbs) || return 1
    rec "xk_${name}=${jx} (${q}, K=56 map extraction: $*)"
    wait_for ${M}/xk_${name}.log MAPS_K56_OK 720
}

lane_d() {
    wait_for ${D}/s56_sweep.log EVAL_INFERENCE_OK 720 || return 1      # s56 = 7807874, queued by v1
    score debug 56 || return 1
    sweep debug G4D63 ${D}/ema_ckpt_G4D63.tar ${E}/G4D63_K56 || return 1
    score debug 63 || return 1
    sweep debug A243rr ${CK%/${RUN}/training_checkpoints}/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar \
          ${E}/A243rr_K56 prod1n_b32_sgdr || return 1
    xk debug d "G4D56:G4D56_K56:G4D56_K56" "G4D63:G4D63_K56:G4D63_K56" "A243:A243rr_K56:prod1n_b32_sgdr_K56" || return 1
    say "LANE_D_DONE"
}
lane_s() {
    local jp jc2 jx
    wait_for ${D}/jc_climate.log CLIMATE_RUN_OK 720 || return 1      # jc = 7807790, queued by v1
    jp=$(sub -q debug-scaling -o ${D}/jp_score.log -v DIRS=7807790,TAG=g4d63,TGM=${TGM} \
         polaris/polaris_protocol_score.pbs) || return 1
    rec "jp=${jp} (debug-scaling, protocol score g4d63)"
    wait_for ${D}/jp_score.log PROTOCOL_SCORE_OK 720 || return 1
    sweep debug-scaling G4D36 ${D}/ema_ckpt_G4D36.tar ${E}/G4D36_K56 || return 1
    score debug-scaling 36 || return 1
    sweep debug-scaling G4D43 ${D}/ema_ckpt_G4D43.tar ${E}/G4D43_K56 || return 1
    score debug-scaling 43 || return 1
    sweep debug-scaling G4D16rr ${G16EMA} ${E}/G4D16rr_K56 || return 1
    jc2=$(sub -q debug-scaling -o ${D}/jc16_climate.log \
          -v "ARMS=G4D16,RUNDIR_G4D16=${RUN},CKPT_G4D16=${CK}/ckpt_mp0_v15.tar,WEIGHTS_G4D16=ema" \
          polaris/polaris_climate_run.pbs) || return 1
    rec "jc16=${jc2} (debug-scaling, G4D16 5-yr re-roll for the bias maps)"
    wait_for ${D}/jc16_climate.log CLIMATE_RUN_OK 720 || return 1
    xk debug-scaling s "G4D36:G4D36_K56:G4D36_K56" "G4D43:G4D43_K56:G4D43_K56" "G4D16:G4D16rr_K56:G4D16_K56" \
        || return 1
    jx=$(sub -q debug-scaling -o ${M}/xc.log -v "CDIRS=7807790 ${jc2}" ${M}/polaris_maps_clim.pbs) || return 1
    rec "xc=${jx} (debug-scaling, 5-yr bias-map extraction: 7807790 ${jc2})"
    wait_for ${M}/xc.log MAPS_CLIM_OK 720 || return 1
    say "LANE_S_DONE"
}

echo "v2 driver started $(date -u +%FT%TZ); adopts j1=7807789 (done), jc=7807790, s56=7807874" >> ${J}
mkdir -p ${M}
lane_d & pd=$!
lane_s & ps=$!
rd=0; rs=0
wait ${pd} || rd=$?
wait ${ps} || rs=$?
say "lanes done: D rc=${rd} S rc=${rs}"
if [ ${rd} -eq 0 ] && [ ${rs} -eq 0 ]; then
    jpl=$(sub -q debug -o ${M}/plot.log ${M}/polaris_maps_plot.pbs) && rec "plot=${jpl} (debug, figures)" \
        && wait_for ${M}/plot.log MAPS_PLOT_OK 360
fi
bash ${D}/test_compare_g4d_e63.sh | tail -1
python3 ${D}/compare_g4d_e63.py
[ ${rd} -eq 0 ] && [ ${rs} -eq 0 ] && has ${M}/plot.log MAPS_PLOT_OK && say "CHAIN_DONE" \
    || { say "ERROR CHAIN_INCOMPLETE"; exit 4; }
