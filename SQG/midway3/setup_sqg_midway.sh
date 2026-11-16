#!/bin/bash
# setup_sqg_midway.sh -- the login-node part of the SQG Midway3 bring-up (MIDWAY3_INROW_PROMPT.md §4, §5).
#   bash SQG/midway3/setup_sqg_midway.sh            # from the repo root, Midway3 LOGIN node; a few minutes
# Installs and files only: it never imports jax/torch/warp (CLAUDE.md #3). The import checks are Job 1,
# SQG/midway3/sqg_env_probe_midway.sh (SQG_MIDWAY_ENV_OK).
#   1. Option A: Weidong's wdenv, read only. Its `pip freeze` is compared with SQG/polaris/requirements-wdenv.txt
#      (the pins Polaris' sqg-env copies); any difference in a key package -> ERROR (then use Option B,
#      SQG/midway3/setup_sqg_env_midway.sh, not written until needed).
#   2. Warp 1.17.0 and nvmath-python 1.0.0 (+ numba-cuda 0.30.4) into pip --target dirs under $SQG_ROOT/pylibs,
#      --no-deps for all three: wdenv already has numpy, and the harnesses import numpy before they put these dirs
#      on sys.path, so a second numpy in the target dir would be dead weight at best.
#   3. The training root: $SQG_ROOT/AllTrainData/trainingdata_m0X<LABEL_TAIL>/checkpoint/ with the 16 o1.2
#      checkpoints (t = 100, 200, 300, 400 of members 1-4) symlinked from Weidong's tree (never written).
# Prints SQG_MIDWAY_SETUP_OK on success.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SQG_ROOT="${SQG_ROOT:-/project/rcc/mehta5/sqg_midway3}"
PY="${SQG_PY:-/project/pedramh/weidong/miniconda3/envs/wdenv/bin/python}"
WD_ALLTRAIN=/project/pedramh/weidong/SQG/SQG_deterForcing/AllTrainData
LABEL_TAIL="_frozen_ring_NU_B_v2_nu0.0002_B1_mu1_kf3.5_dkf0.5_seed1_N4096_dt5e-06_Tstart0_Tspin20_Tprod1100_save0.0025_pair0.02"
case "$SQG_ROOT" in /project/pedramh/weidong/*) echo "ERROR SQG_ROOT_IN_WEIDONG_TREE $SQG_ROOT"; exit 2 ;; esac
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SQG_ROOT}/pip_cache}"     # NOT $HOME (24.7 of 30 GB used, 2026-11-16)
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
mkdir -p "$SQG_ROOT"/{pylibs,tmp,logs,AllTrainData,val_store} "$PIP_CACHE_DIR"
echo "repo=$REPO sqg_root=$SQG_ROOT py=$PY pip_cache=$PIP_CACHE_DIR"

# ---- 1. Option A: wdenv, read only ----------------------------------------------------------------------------------
[ -x "$PY" ] || { echo "ERROR SQG_WDENV_MISSING $PY (use Option B)"; exit 3; }
FREEZE="$SQG_ROOT/wdenv-freeze.txt"
"$PY" -m pip --disable-pip-version-check freeze > "$FREEZE"
python3 - "$FREEZE" "$REPO/SQG/polaris/requirements-wdenv.txt" <<'EOF'
import sys
def pins(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if "==" in line and not line.startswith("#"):
            k, _, v = line.partition("==")
            out[k.lower().replace("_", "-")] = v.split(";")[0].strip()
    return out
have, want = pins(sys.argv[1]), pins(sys.argv[2])
key = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "pyqg-jax", "torch", "timm", "numpy",
       "nvidia-cufft-cu12", "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-nvcc-cu12",
       "nvidia-cuda-runtime-cu12")
bad = [k for k in key if k in want and have.get(k) != want[k]]
other = [k for k in want if k not in key and have.get(k) != want[k]]
for k in key:
    if k in want:
        print(f"  {k}=={have.get(k)} (Polaris sqg-env: {want[k]})")
print(f"  {len(want)} Polaris pins, {len(other)} non-key differences: {', '.join(other[:12])}")
if bad:
    print("ERROR SQG_WDENV_DIFFERS " + " ".join(f"{k}={have.get(k)}!={want[k]}" for k in bad))
    sys.exit(1)
print("SQG_WDENV_MATCHES key pins")
EOF

# ---- 2. Warp + nvmath in --target dirs ------------------------------------------------------------------------------
PIP=("$PY" -m pip --disable-pip-version-check install --progress-bar off --no-deps --no-compile)
W="$SQG_ROOT/pylibs/warp-1.17.0"
NV="$SQG_ROOT/pylibs/nvmath-1.0.0"
[ -d "$W/warp" ] || "${PIP[@]}" --target "$W" warp-lang==1.17.0
[ -d "$NV/nvmath" ] || "${PIP[@]}" --target "$NV" nvmath-python==1.0.0 numba-cuda==0.30.4
ls -d "$W"/warp_lang-*.dist-info "$NV"/nvmath_python-*.dist-info "$NV"/numba_cuda-*.dist-info

# ---- 3. training root: 16 symlinked checkpoints ---------------------------------------------------------------------
n=0
for X in 1 2 3 4; do
    d="$SQG_ROOT/AllTrainData/trainingdata_m0${X}${LABEL_TAIL}/checkpoint"
    mkdir -p "$d"
    for t in 0100 0200 0300 0400; do
        src="$WD_ALLTRAIN/trainingdata_m0${X}${LABEL_TAIL}/checkpoint/ckpt_t${t}.00.npz"
        [ -r "$src" ] || { echo "ERROR SQG_CKPT_MISSING $src (ask the user to Globus the staged set)"; exit 4; }
        ln -sfn "$src" "$d/ckpt_t${t}.00.npz"
        n=$((n + 1))
    done
done
echo "checkpoints linked: $n under $SQG_ROOT/AllTrainData"
[ -s "$SQG_ROOT/val_store/theta.npy" ] && echo "val store: present" \
    || echo "val store: NOT YET COPIED (Jobs 3A-3C need $SQG_ROOT/val_store/{theta.npy,attrs.json} from Polaris)"
echo "SQG_MIDWAY_SETUP_OK root=$SQG_ROOT"
