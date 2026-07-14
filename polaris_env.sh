# polaris_env.sh — shared, USER-AGNOSTIC environment for every polaris_*.pbs here.
#
# Source it AFTER `module use /soft/modulefiles; module load conda; conda activate base`.
# Any member of the lighthouse-uchicago project can run the scripts unchanged: all
# WRITES go to your own member dir, all heavy read-only inputs are shared.
#
# It resolves three roots:
#   MEMBER_ROOT : YOUR writable dir  -> caches, TMPDIR, logs, run/exp dirs, bench CSVs
#   SHARED_ROOT : group-readable artifacts you may REUSE (converted datasets + SFNO venv)
#   E3SM_ROOT   : the shared read-only E3SM source archive
#
# Overrides (all optional):
#   POLARIS_MEMBER=<dir under members/>   # if auto-detect picks wrong
#   POLARIS_SHARED=<dir>                  # where to reuse converted data/venv from
#   POLARIS_SFNO_VENV=<dir>               # force a specific SFNO venv
#   POLARIS_E3SM_ROOT=<dir>               # alternate E3SM archive

MEMBERS=/eagle/projects/lighthouse-uchicago/members

# --- MEMBER_ROOT ------------------------------------------------------------
# NOTE: the member dir name is NOT always $USER (e.g. user rmehta1987 -> members/mehta5),
# so: explicit override, then $USER, then the first writable dir.
if [ -n "${POLARIS_MEMBER:-}" ]; then
    MEMBER_ROOT="${MEMBERS}/${POLARIS_MEMBER}"
elif [ -d "${MEMBERS}/${USER}" ] && [ -w "${MEMBERS}/${USER}" ]; then
    MEMBER_ROOT="${MEMBERS}/${USER}"
else
    # Pick the dir you OWN (-O), not merely the first writable one — the members/ dirs
    # are group-readable, so a plain -w scan could latch onto someone else's folder.
    MEMBER_ROOT=""
    for _d in "${MEMBERS}"/*/; do
        if [ -O "${_d}" ] && [ -w "${_d}" ]; then MEMBER_ROOT="${_d%/}"; break; fi
    done
fi
if [ -z "${MEMBER_ROOT}" ] || [ ! -w "${MEMBER_ROOT}" ]; then
    echo "ERROR POLARIS_MEMBER_UNRESOLVED: no writable dir found under ${MEMBERS}"
    echo "  Fix: export POLARIS_MEMBER=<your folder name under members/>   (e.g. jesswan)"
    return 2 2>/dev/null || exit 2
fi
export MEMBER_ROOT

# --- SHARED_ROOT: reuse instead of rebuilding -------------------------------
# mehta5's converted datasets + SFNO venv are group-readable (drwxr-sr-x), so a second
# user can skip ~75 GB of conversion and the torch_harmonics source build entirely.
export SHARED_ROOT="${POLARIS_SHARED:-${MEMBERS}/mehta5}"

# --- E3SM source archive (read-only, owned by jesswan) -----------------------
export E3SM_ROOT="${POLARIS_E3SM_ROOT:-/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101}"

# --- per-user writable caches (persistent on eagle; NOT node-local /local/scratch,
#     which is wiped at job end so every job would recompile) -----------------
export TMPDIR="${MEMBER_ROOT}/tmp"
export TORCHINDUCTOR_CACHE_DIR="${MEMBER_ROOT}/torchinductor_cache"
export TRITON_CACHE_DIR="${MEMBER_ROOT}/triton_cache"
export POLARIS_LOG_DIR="${MEMBER_ROOT}/polaris_logs"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${POLARIS_LOG_DIR}" 2>/dev/null

# --- knobs every model needs on this cluster --------------------------------
export WANDB_MODE=offline
export HDF5_USE_FILE_LOCKING=FALSE          # Lustre: h5py/netCDF locking must be off
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- SFNO venv (makani / physicsnemo only): prefer your own, else the shared one ---
if [ -n "${POLARIS_SFNO_VENV:-}" ]; then
    SFNO_VENV="${POLARIS_SFNO_VENV}"
elif [ -x "${MEMBER_ROOT}/conda-envs/sfno-venv/bin/python" ]; then
    SFNO_VENV="${MEMBER_ROOT}/conda-envs/sfno-venv"
else
    SFNO_VENV="${SHARED_ROOT}/conda-envs/sfno-venv"
fi
export SFNO_VENV

# --- per-user derived data locations (converted stages) ----------------------
# If you have your own copy use it; otherwise fall back to the shared one (read-only).
_pick() {  # _pick <var> <relative-path>
    local _mine="${MEMBER_ROOT}/$2" _shared="${SHARED_ROOT}/$2"
    if [ -e "${_mine}" ]; then echo "${_mine}"; elif [ -e "${_shared}" ]; then echo "${_shared}"; else echo "${_mine}"; fi
}

# --- base-conda top-ups (netCDF4 / zarr / torch_harmonics 0.7.4) -------------
# The base conda lacks these. They MUST NOT live in a `pip install --user` dir: ALCF homes
# are mode 0700, so ~/.local packages are readable by one person only and every other
# member's Pangu/SI job dies on `ModuleNotFoundError: torch_harmonics` / `netCDF4`.
# Build with polaris_setup_base_topups.sh.
#
# ⚠ Exported as a PATH ONLY — deliberately NOT prepended to PYTHONPATH here. PYTHONPATH
#   outranks a venv's site-packages, so this dir's torch_harmonics 0.7.4 would shadow the
#   SFNO venv's 0.9.x and re-break makani. Pangu/SI/S2S opt in themselves; the SFNO
#   scripts must never add it.
export POLARIS_TOPUPS="$(_pick POLARIS_TOPUPS conda-envs/polaris-topups)"

# Call this from any job that runs on the BASE conda (Pangu / SI / S2S / probe) — NOT from an
# SFNO job. It turns the two silent failure modes into loud ones:
#   * top-ups missing  -> the installer's job silently falls back to their private ~/.local
#     and passes, while everyone else gets ModuleNotFoundError. That is the original bug,
#     and it is invisible precisely to the person who would have to fix it.
#   * a package still resolving out of /home/<someone>/.local -> the result is not
#     reproducible by anyone else, so it is not a result.
polaris_require_topups() {
    if [ ! -d "${POLARIS_TOPUPS}" ]; then
        echo "ERROR TOPUPS_MISSING: ${POLARIS_TOPUPS}"
        echo "  Build it once on a LOGIN node:  bash <repo>/polaris_setup_base_topups.sh"
        echo "  (netCDF4 / torch_harmonics / tensorly / natsort / cartopy live there — the base"
        echo "   conda has none of them.)"
        return 3
    fi
    python - <<'_PY' || return 3
import sys
bad = []
for m in ("netCDF4", "torch_harmonics", "tensorly", "natsort"):
    try:
        mod = __import__(m)
    except ImportError:
        continue                      # a genuinely absent module fails later, with a clear name
    if "/.local/" in (mod.__file__ or ""):
        bad.append("%s <- %s" % (m, mod.__file__))
if bad:
    print("ERROR PRIVATE_DEPS_ON_PATH")
    print("  These resolved from somebody's PRIVATE home (mode 0700 on ALCF), so this run is")
    print("  reproducible by exactly one person:")
    for b in bad:
        print("    " + b)
    print("  Fix: bash <repo>/polaris_setup_base_topups.sh, and make sure $POLARIS_TOPUPS is")
    print("  on PYTHONPATH ahead of the user site.")
    sys.exit(3)
_PY
}
export SI_STAGE="$(_pick SI_STAGE si_e3sm_stage)"
export MAKANI_DATA="$(_pick MAKANI_DATA data/e3sm_makani)"
export SEQZARR_DATA="$(_pick SEQZARR_DATA e3sm_seqzarr)"
# PanguWeather aux (~17 GB: the Z->Z_2 stats + the CDF-5->NETCDF4 climatology).
# Lives OUTSIDE the repo so it can be shared read-only like the datasets above —
# otherwise every user re-encodes a 16 GB climatology into their own clone.
export PANGU_AUX="$(_pick PANGU_AUX pangu_polaris_data)"

polaris_env_report() {
    echo "--- polaris_env ---"
    echo "  MEMBER_ROOT   = ${MEMBER_ROOT}   (writes go here)"
    echo "  SHARED_ROOT   = ${SHARED_ROOT}"
    echo "  E3SM_ROOT     = ${E3SM_ROOT}"
    echo "  SI_STAGE      = ${SI_STAGE}"
    echo "  MAKANI_DATA   = ${MAKANI_DATA}"
    echo "  SEQZARR_DATA  = ${SEQZARR_DATA}"
    echo "  PANGU_AUX     = ${PANGU_AUX}"
    echo "  SFNO_VENV     = ${SFNO_VENV}"
    echo "  logs          = ${POLARIS_LOG_DIR}"
}
