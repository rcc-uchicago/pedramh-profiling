#!/bin/bash
# One-off: submit score → report → figures chains for the two beta1 amd-rtx
# inference runs (3136630 + 3136631). Inference outputs already exist in
# $WORK2/SFNO_Climate_Emulator/results/sfno_eval/<RUN_TAG>/inference/, so this script
# skips the inference stage and chains the 3 downstream stages.
#
# RUN THIS FROM A LOGIN NODE (login1/login2/login3). TACC blocks sbatch
# from compute nodes (incl. idev sessions like the current c454-073 shell).

set -euo pipefail
cd "$(dirname "$0")/.."

submit_chain() {
    local tag="$1"
    local family="$2"
    local exports="ALL,RUN_TAG=${tag},EVAL_SHA7=867fead,DATA_SHA7=8b395eb,TRAIN_SHA7=867fead,RUN_DIR=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/runs/${family}/plasim_${family#sfno_}/0,CKPT=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/runs/${family}/plasim_${family#sfno_}/0/training_checkpoints/best_ckpt_ema_mp0.tar,MODE=nwp,TEST_HOLDOUT=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/test_holdout,TRAIN_DIR=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/train,PACKAGER_TEST_SRC=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11/test,TRACK=own"

    local sco rep fig
    sco=$(sbatch --parsable --export="$exports" scripts/submit_eval_score.slurm)
    echo "score:   $sco   ($family)"
    rep=$(sbatch --parsable --dependency=afterok:$sco --export="$exports" scripts/submit_eval_report.slurm)
    echo "report:  $rep   afterok:$sco"
    fig=$(sbatch --parsable --dependency=afterok:$rep --export="$exports" scripts/submit_eval_figures.slurm)
    echo "figures: $fig   afterok:$rep"
}

echo "=== beta1_0p95 chain ==="
submit_chain \
    "20260520_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p95_ckpt-best_ckpt_ema_mp0" \
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p95"

echo
echo "=== beta1_0p97 chain ==="
submit_chain \
    "20260520_eval-867fead_data-8b395eb_family-sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p97_ckpt-best_ckpt_ema_mp0" \
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p97"

echo
echo "=== queue ==="
squeue -u "$USER" -o "%.10i %.9P %.20j %.2t %.10M %.6D %R"
