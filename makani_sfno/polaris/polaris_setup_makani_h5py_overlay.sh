#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Build the minimal h5py overlay that Makani needs on Polaris after the Cray PE
# roll deleted hdf5 1.14.3.5.  Run ONCE, on a LOGIN node (it needs the ALCF
# proxy for PyPI and compiles nothing):
#
#     bash makani_sfno/polaris/polaris_setup_makani_h5py_overlay.sh
#
# PASS = MAKANI_H5PY_OVERLAY_OK
#
# WHY
#   makani/utils/metric.py:19 is a bare `import h5py`, so h5py sits on the
#   import path of every makani entrypoint -- including --enable_synthetic_data
#   runs, which therefore do NOT dodge it.  The base conda's h5py is linked
#   against libhdf5_parallel_gnu_123.so.200 (hdf5 1.14.3.5); the PE now ships
#   only 1.14.3.9, whose soname is ...gnu.so.310.  200 -> 310 is a real
#   soversion bump, so unlike the libmpi_gnu_123 case that one CANNOT be fixed
#   with a symlink -- see polaris_makani_env.sh's comments.  A PyPI manylinux
#   wheel vendors its own libhdf5 (h5py.libs/libhdf5-*.so.320) and needs no
#   Cray hdf5 at all.
#
# WHY AN OVERLAY AND NOT `pip install` INTO THE VENV
#   $SFNO_VENV is shared (the 2026-07 top-ups made these envs reproducible by a
#   second user), and mutating a shared env to fix one consumer is the failure
#   mode CLAUDE.md rule #5 exists for.  An overlay is additive and removable
#   with `rm -rf`, and it is the pattern $POLARIS_TOPUPS already established.
#
# WHY IT INSTALLS EXACTLY ONE PACKAGE
#   The overlay goes on PYTHONPATH, which outranks site-packages.  An overlay
#   that also carried numpy/torch/torch_harmonics would shadow the venv's own --
#   the failure the Makani PBS scripts' TORCH_HARMONICS_SHADOWED gate exists to
#   catch.  `--no-deps` is therefore load-bearing, not tidiness.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
source "${REPO}/polaris_env.sh" || exit 2
# shellcheck disable=SC1091
source "${REPO}/makani_sfno/polaris/polaris_makani_env.sh" || exit 2

if [ ! -x "${SFNO_VENV}/bin/python" ]; then
    echo "ERROR SFNO_VENV_MISSING: ${SFNO_VENV}"
    echo "  Build it first:  bash ${REPO}/polaris_setup_sfno_venv.sh"
    exit 3
fi

OVERLAY="${MAKANI_H5PY_OVERLAY:-${MEMBER_ROOT}/conda-envs/makani-h5py-overlay}"
echo "=== makani h5py overlay ==="
echo "target   : ${OVERLAY}"
echo "python   : ${SFNO_VENV}/bin/python"
echo "env from : ${MAKANI_ENV_SOURCE}"

mkdir -p "${OVERLAY}" || exit 2

# --target + --no-deps: a flat directory holding h5py and nothing else.
# --upgrade so a re-run replaces rather than silently keeping a stale copy.
PYTHONNOUSERSITE=1 "${SFNO_VENV}/bin/python" -m pip install \
    --no-deps --upgrade --only-binary :all: \
    --target "${OVERLAY}" h5py || {
        echo "ERROR H5PY_INSTALL_FAILED"
        echo "  Login nodes reach PyPI through http_proxy=proxy.alcf.anl.gov:3128, which"
        echo "  polaris_makani_env.sh exports. If this failed with a network error, confirm"
        echo "  that proxy is still the right one before assuming PyPI is down."
        exit 3
    }

# ---- gate 1: the overlay holds ONLY h5py ------------------------------------
# Anything else here is a package that would shadow the venv's copy.
_extra="$(find "${OVERLAY}" -maxdepth 1 -mindepth 1 \
            ! -name 'h5py' ! -name 'h5py.libs' ! -name 'h5py-*' \
            ! -name '__pycache__' -printf '%f\n' 2>/dev/null)"
if [ -n "${_extra}" ]; then
    echo "ERROR OVERLAY_NOT_MINIMAL: unexpected entries in ${OVERLAY}:"
    echo "${_extra}" | sed 's/^/    /'
    echo "  These go on PYTHONPATH ahead of the venv and can shadow torch/torch_harmonics."
    echo "  Wipe the dir and re-run; do not hand-prune it."
    exit 3
fi

# ---- gate 2: it imports, from the overlay, with a vendored libhdf5 ----------
PYTHONNOUSERSITE=1 PYTHONPATH="${OVERLAY}" "${SFNO_VENV}/bin/python" - <<'PY' || exit 3
import os, sys
try:
    import h5py
except Exception as exc:                     # noqa: BLE001 - want the reason verbatim
    print("ERROR H5PY_IMPORT_FAILED: %s" % exc)
    sys.exit(3)
overlay = os.path.realpath(os.environ["PYTHONPATH"])
got = os.path.realpath(h5py.__file__)
if not got.startswith(overlay + os.sep):
    print("ERROR H5PY_NOT_FROM_OVERLAY")
    print("  imported from : %s" % got)
    print("  expected under: %s" % overlay)
    sys.exit(3)
libs = os.path.join(os.path.dirname(got), os.pardir, "h5py.libs")
vendored = sorted(f for f in os.listdir(libs) if f.startswith("libhdf5")) if os.path.isdir(libs) else []
if not vendored:
    print("ERROR H5PY_NO_VENDORED_HDF5")
    print("  This wheel expects a system libhdf5, which is the thing that is gone.")
    print("  Do not 'fix' it by putting cray-hdf5-parallel/1.14.3.9 on LD_LIBRARY_PATH:")
    print("  soversion 310 != the 200 this was built against.")
    sys.exit(3)
print("h5py %s (hdf5 %s) from the overlay" % (h5py.__version__, h5py.version.hdf5_version))
print("vendored: %s" % ", ".join(vendored))
PY

echo "MAKANI_H5PY_OVERLAY_OK ${OVERLAY}"
echo
echo "Nothing else needs doing: polaris_makani_env.sh puts this dir on PYTHONPATH"
echo "automatically whenever it exists. To undo:  rm -rf ${OVERLAY}"
