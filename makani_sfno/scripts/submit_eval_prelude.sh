#!/bin/bash
# submit_eval_prelude.sh — sourceable env-resolution + collision-guard for
# the eval pipeline. Used by both standalone scripts/submit_eval.sh and the
# bundled-training-eval helper src/sfno_training/bundled_eval.sh
# (docs/2026-05-20_bundled_training_eval_plan.md §4.2).
#
# This file defines exactly one function: submit_eval_compute_env.
# It NEVER calls `exit` — only `return` — so sourcing it from inside a
# training SLURM job cannot terminate that job mid-way.
#
# On success the function exports:
#   RUN_DIR CKPT MODE TEST_HOLDOUT TRAIN_DIR PACKAGER_TEST_SRC TRACK
#   EVAL_SHA7 DATA_SHA7 TRAIN_SHA7 TRAIN_FAMILY RUN_TAG OUT_ROOT
#   BENCHMARK_5410_OUT_ROOT
# and writes "$OUT_ROOT/provenance.txt".
#
# Return codes:
#   0  success
#   2  required CLI tool missing (git)
#   3  collision guard tripped (OUT_ROOT exists and ALLOW_RERUN!=1)

submit_eval_compute_env() {
    local REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"

    # ---- defaults ----------------------------------------------------------
    : "${RUN_DIR:=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/0}"
    # Prefer EMA-best (canonical for EMA-enabled runs); fall back to raw-best
    # when EMA isn't available (legacy / EMA-disabled runs).
    if [ -z "${CKPT:-}" ]; then
        if [ -s "$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar" ]; then
            CKPT="$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar"
        else
            CKPT="$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar"
        fi
    fi
    : "${MODE:=nwp}"
    : "${TRACK:=own}"

    # ---- dataset family auto-resolution -----------------------------------
    # The eval-time TEST_HOLDOUT / TRAIN_DIR / PACKAGER_TEST_SRC MUST belong
    # to the same dataset family the model was TRAINED on, or scores are
    # meaningless (e.g. v10 vs v11 differ in `pl` convention). We derive
    # the family from RUN_DIR/config.json:train_data_path and either
    # auto-set the three data vars, or — if the caller passed values that
    # disagree with the trained family — abort.
    local CFG_JSON="$RUN_DIR/config.json"
    if [ ! -f "$CFG_JSON" ]; then
        echo "[submit_eval_prelude] ERROR: $CFG_JSON not found — cannot resolve dataset family" >&2
        return 2
    fi
    local TRAIN_FROM_CFG
    TRAIN_FROM_CFG="$("$REPO_ROOT/.venv/bin/python" -c "
import json, sys
c = json.load(open(sys.argv[1]))
p = c.get('train_data_path', '')
if not p:
    sys.exit('config.json has no train_data_path')
print(p)
" "$CFG_JSON")" || { echo "[submit_eval_prelude] ERROR: could not read train_data_path from $CFG_JSON" >&2; return 2; }

    local DATASET_ROOT
    DATASET_ROOT="$(dirname "$TRAIN_FROM_CFG")"   # e.g. .../sim52_zgplev_full_v11
    local DATASET_FAMILY
    DATASET_FAMILY="$(basename "$DATASET_ROOT")"
    # Map dataset family → packager-source family. The packager-source name
    # is `sim52_astro_64x128_zgplev` with the same suffix as the dataset
    # family ("", "_v11", ...).
    local DATASET_SUFFIX="${DATASET_FAMILY#sim52_zgplev_full}"   # "" or "_v11" or ...
    local PACKAGER_ROOT="$(dirname "$DATASET_ROOT")/sim52_astro_64x128_zgplev${DATASET_SUFFIX}"

    local EXPECTED_TEST_HOLDOUT="$DATASET_ROOT/test_holdout"
    local EXPECTED_TRAIN_DIR="$DATASET_ROOT/train"
    local EXPECTED_PACKAGER_TEST_SRC="$PACKAGER_ROOT/test"

    # If caller passed a value, it must match the trained family. Otherwise auto-set.
    local var
    for var in TEST_HOLDOUT TRAIN_DIR PACKAGER_TEST_SRC; do
        local expected_var="EXPECTED_$var"
        local expected="${!expected_var}"
        local actual="${!var:-}"
        if [ -z "$actual" ]; then
            printf -v "$var" '%s' "$expected"
        elif [ "$actual" != "$expected" ]; then
            echo "[submit_eval_prelude] FATAL: $var=$actual does not match the dataset family the model was trained on ($DATASET_FAMILY)." >&2
            echo "                       Expected: $expected" >&2
            echo "                       This is the v10/v11 confound — refusing to score a model on the wrong dataset family." >&2
            echo "                       Either unset $var (auto-resolves) or pass a path under $DATASET_ROOT (or $PACKAGER_ROOT for PACKAGER_TEST_SRC)." >&2
            return 2
        fi
    done
    echo "[submit_eval_prelude] dataset family auto-resolved from $CFG_JSON: $DATASET_FAMILY" >&2

    # ---- EVAL_SHA7 ---------------------------------------------------------
    if ! command -v git >/dev/null 2>&1; then
        echo "[submit_eval_prelude] ERROR: git is required" >&2
        return 2
    fi
    EVAL_SHA7="$(cd "$REPO_ROOT" && git rev-parse --short=7 HEAD)"

    # ---- DATA_SHA7 ---------------------------------------------------------
    local TEST_FILE_FOR_SHA="$PACKAGER_TEST_SRC/MOST.0121.h5"
    if [ ! -f "$TEST_FILE_FOR_SHA" ]; then
        echo "[submit_eval_prelude] WARNING: $TEST_FILE_FOR_SHA not found; setting DATA_SHA7=unknown" >&2
        DATA_SHA7="unknown"
    else
        DATA_SHA7="$("$REPO_ROOT/.venv/bin/python" -c "
import h5py, sys
with h5py.File(sys.argv[1], 'r') as f:
    sha = f.attrs.get('packager_git_sha', b'unknown')
    if isinstance(sha, bytes):
        sha = sha.decode('utf-8')
    print(sha[:7])
" "$TEST_FILE_FOR_SHA")"
    fi

    # ---- TRAIN_SHA7 --------------------------------------------------------
    if [ -f "$RUN_DIR/train_code_sha.txt" ]; then
        TRAIN_SHA7="$(head -c 7 "$RUN_DIR/train_code_sha.txt")"
    elif [ -f "$RUN_DIR/out.log" ]; then
        TRAIN_SHA7="$(grep -m1 "git hash:" "$RUN_DIR/out.log" | sed -E "s/.*git hash: b?'?([0-9a-f]{7}).*/\1/")"
        [ -z "$TRAIN_SHA7" ] && TRAIN_SHA7="unknown"
    else
        TRAIN_SHA7="unknown"
    fi

    # ---- RUN_TAG -----------------------------------------------------------
    TRAIN_FAMILY="$(basename "$(dirname "$(dirname "$RUN_DIR")")")"
    local DATE_STR
    DATE_STR="$(date +%Y%m%d)"
    local CKPT_BASENAME
    CKPT_BASENAME="$(basename "$CKPT" .tar)"
    if [ "${FULL_RUN_TAG:-0}" = "1" ]; then
        : "${RUN_TAG:=${DATE_STR}_eval-${EVAL_SHA7}_data-${DATA_SHA7}_train-${TRAIN_SHA7}_family-${TRAIN_FAMILY}_ckpt-${CKPT_BASENAME}}"
    else
        local _RT="${DATE_STR}_eval-${EVAL_SHA7}_data-${DATA_SHA7}"
        if [ "$TRAIN_SHA7" != "$EVAL_SHA7" ]; then
            _RT="${_RT}_train-${TRAIN_SHA7}"
        fi
        _RT="${_RT}_family-${TRAIN_FAMILY}"
        if [ "$CKPT_BASENAME" != "best_ckpt_mp0" ]; then
            _RT="${_RT}_ckpt-${CKPT_BASENAME}"
        fi
        : "${RUN_TAG:=${_RT}}"
    fi

    # ---- OUT_ROOT ----------------------------------------------------------
    : "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval/$RUN_TAG}"

    # ---- Collision guard (tightened — see plan §4.7) -----------------------
    # Any existing $OUT_ROOT requires ALLOW_RERUN=1, regardless of whether
    # the recorded CKPT matches. This is strictly stronger than the historic
    # CKPT-path-only check and closes the same-CKPT-resumed-run hole.
    if [ -e "$OUT_ROOT" ]; then
        if [ "${ALLOW_RERUN:-0}" != "1" ]; then
            cat >&2 <<EOF
[submit_eval_prelude] FATAL: $OUT_ROOT already exists.
  RUN_TAG  = $RUN_TAG
  CKPT     = $CKPT
Set ALLOW_RERUN=1 to overwrite into the existing dir, OR
pass RUN_TAG=<unique-name>, OR
move/remove the existing $OUT_ROOT.

This guard was tightened on 2026-05-20 to refuse any existing OUT_ROOT
(not just CKPT-path mismatch). See docs/2026-05-20_bundled_training_eval_plan.md §4.7.
EOF
            return 3
        else
            echo "[submit_eval_prelude] ALLOW_RERUN=1 — proceeding into existing $OUT_ROOT" >&2
            # Defensive logging: if the existing provenance recorded a
            # different CKPT, warn loudly (does not abort under ALLOW_RERUN=1).
            if [ -f "$OUT_ROOT/provenance.txt" ]; then
                local existing_ckpt
                existing_ckpt="$(grep -m1 '^CKPT=' "$OUT_ROOT/provenance.txt" | cut -d= -f2-)"
                if [ -n "$existing_ckpt" ] && [ "$existing_ckpt" != "$CKPT" ]; then
                    echo "[submit_eval_prelude] WARNING: existing provenance records CKPT=$existing_ckpt but this run uses CKPT=$CKPT" >&2
                fi
            fi
        fi
    fi

    mkdir -p "$OUT_ROOT" "$REPO_ROOT/logs"

    # ---- Provenance sidecar (always full) ----------------------------------
    cat > "$OUT_ROOT/provenance.txt" <<EOF
RUN_TAG=$RUN_TAG
EVAL_SHA7=$EVAL_SHA7
DATA_SHA7=$DATA_SHA7
TRAIN_SHA7=$TRAIN_SHA7
TRAIN_FAMILY=$TRAIN_FAMILY
CKPT=$CKPT
CKPT_BASENAME=$CKPT_BASENAME
RUN_DIR=$RUN_DIR
MODE=$MODE
TEST_HOLDOUT=$TEST_HOLDOUT
TRAIN_DIR=$TRAIN_DIR
PACKAGER_TEST_SRC=$PACKAGER_TEST_SRC
TRACK=$TRACK
DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

    # ---- 5410 benchmark overlay default -----------------------------------
    : "${BENCHMARK_5410_OUT_ROOT:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid}"

    # ---- Echo + export ----------------------------------------------------
    cat <<EOF
[submit_eval_prelude]
  RUN_DIR           = $RUN_DIR
  CKPT              = $CKPT
  MODE              = $MODE
  TEST_HOLDOUT      = $TEST_HOLDOUT
  TRAIN_DIR         = $TRAIN_DIR
  PACKAGER_TEST_SRC = $PACKAGER_TEST_SRC
  EVAL_SHA7         = $EVAL_SHA7
  DATA_SHA7         = $DATA_SHA7
  TRAIN_SHA7        = $TRAIN_SHA7
  TRAIN_FAMILY      = $TRAIN_FAMILY
  RUN_TAG           = $RUN_TAG
  OUT_ROOT          = $OUT_ROOT
  TRACK             = $TRACK
EOF

    export RUN_DIR CKPT MODE TEST_HOLDOUT TRAIN_DIR PACKAGER_TEST_SRC TRACK
    export EVAL_SHA7 DATA_SHA7 TRAIN_SHA7 TRAIN_FAMILY RUN_TAG OUT_ROOT
    export BENCHMARK_5410_OUT_ROOT
    return 0
}
