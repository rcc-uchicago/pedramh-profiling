#!/bin/bash -l
# One node of polaris_ace2_rollout_5yr.pbs: 4 independent fme inference runs, one per GPU.
# run r = node*4 + gpu;  arm = ARMS[r // 8];  start = r % 8   (ace2_rollout.py ARMS order)
# Env from the launcher: ROOT, PY, ACE2_DIR, REPO.

NODE="${PMI_RANK:-${PALS_RANKID:-0}}"
# shellcheck disable=SC1091
source "${REPO}/polaris_env.sh" > /dev/null || exit 2
# shellcheck disable=SC1091
source "${ACE2_DIR}/polaris/polaris_ace2_env.sh" > /dev/null || exit 2
# shellcheck disable=SC1091
source "$(dirname "${PY}")/activate"
export PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE WANDB_MODE=offline
unset FME_USE_SRUN
# shellcheck disable=SC1091
source "${ROOT}/ckpts.env"

ARMS=(era5_ai2 eamv3_ai2)
pids=()
for gpu in 0 1 2 3; do
    r=$((NODE * 4 + gpu)); arm="${ARMS[$((r / 8))]}"; start=$((r % 8))
    label="run_${arm}_s${start}"
    ckpt_var="CKPT_${arm}"; ckpt="${!ckpt_var}"
    steps="$("${PY}" -c "import json;print(json.load(open('${ROOT}/probe.json'))['steps']['${arm}'])")"
    compat="$(cat "${ROOT}/probe_${arm}.compat" 2>/dev/null || echo strict)"
    extra=(); [ "${compat}" = allow ] && extra=(--allow-incompatible)
    "${PY}" "${ACE2_DIR}/polaris/ace2_rollout.py" config --arm "${arm}" --start "${start}" \
        --steps "${steps}" --ckpt "${ckpt}" --inputs "${ROOT}/inputs_asc" \
        --out "${ROOT}/${label}" --yaml "${ROOT}/${label}.yaml" "${extra[@]}" > /dev/null || continue
    echo "node ${NODE} gpu ${gpu}: ${label} steps=${steps} compat=${compat} host=$(hostname)"
    CUDA_VISIBLE_DEVICES="${gpu}" timeout -k 30 3300 "${PY}" -m fme.ace.inference \
        "${ROOT}/${label}.yaml" > "${ROOT}/${label}.out" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}"; echo "node ${NODE}: pid ${p} rc=$?"; done
