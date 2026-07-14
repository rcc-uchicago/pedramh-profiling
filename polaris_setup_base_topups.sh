#!/bin/bash -l
# ============================================================================
# One-time setup: the BASE-CONDA TOP-UPS, installed to a SHARED, WORLD-READABLE
# dir on eagle.
#
#     bash polaris_setup_base_topups.sh          # PASS = "TOPUPS_OK"
#
# WHY THIS EXISTS (it is not cosmetic — it was a real, silent blocker):
# The ALCF base conda lacks netCDF4 / zarr / torch_harmonics. These were originally
# added with `pip install --user`, which lands them in $PYTHONUSERBASE under
# /home/<installer>/.local/... . ALCF home dirs are mode 0700, so those packages are
# readable by EXACTLY ONE PERSON. Every "green" Pangu/SI run was therefore green only
# for the installer; a second member's identical job dies on:
#     ModuleNotFoundError: No module named 'torch_harmonics'   (networks/modulus_sfno/sfnonet.py)
#     ModuleNotFoundError: No module named 'netCDF4'           (utils/data_loader_multifiles.py)
# Installing to a shared eagle dir makes the smokes reproducible by the whole project,
# which is the point of the deliverable.
#
# Consumed via $POLARIS_TOPUPS (resolved in polaris_env.sh). Pangu / SI / S2S prepend it
# to PYTHONPATH themselves.
#
# ⚠ DO NOT prepend $POLARIS_TOPUPS to PYTHONPATH globally in polaris_env.sh. PYTHONPATH
#   outranks a venv's site-packages, so the torch_harmonics 0.7.4 in here would shadow the
#   SFNO venv's source-built 0.9.x and re-break makani ("cannot import name
#   precompute_latitudes") — the exact bug PYTHONNOUSERSITE=1 was added to kill, since
#   PYTHONNOUSERSITE does NOT block PYTHONPATH. The SFNO scripts assert torch_harmonics
#   resolves inside their venv, so a leak fails loudly instead of silently.
#
# Run on a LOGIN node: compute nodes have no outbound network.
# ============================================================================
set -uo pipefail

if [[ "$(hostname)" != *login* ]]; then
    echo "ERROR WRONG_NODE: run this on a Polaris LOGIN node (compute nodes have no network)."
    exit 2
fi

module use /soft/modulefiles
module load conda
conda activate base

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/polaris_env.sh" || exit 2

TARGET="${POLARIS_TOPUPS}"
echo "=== base top-ups -> ${TARGET}"
echo "    python: $(command -v python)  ($(python -V 2>&1))"
mkdir -p "${TARGET}" || { echo "ERROR MKDIR_FAILED: ${TARGET}"; exit 2; }

# --- why --no-deps is MANDATORY here ---------------------------------------
# `pip install --target` cannot see the base conda's site-packages, so it treats every
# dependency as missing and RE-RESOLVES THE WHOLE STACK. Without --no-deps this pulled
# torch 2.13.0 + a full CUDA-13 stack + numpy 2.5.1 into the target (4.1 GB). Because
# consumers put $POLARIS_TOPUPS on PYTHONPATH — which outranks site-packages — that torch
# would SHADOW the base conda's torch 2.8.0/cu12.9 and silently move every smoke onto an
# untested toolchain. So: install --no-deps, and add ONLY the deps the base genuinely
# lacks. Base already provides numpy, certifi, packaging, h5py, torch (verified 2026-07-14).
#
# Pins match what the GREEN smokes actually ran on (polaris_pbs_notes.md §2):
#   torch_harmonics 0.7.4 — NOT 0.9.x. 0.9.1's attention/_C.so ABI-breaks torch 2.8
#   ("undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv") and it ships no sdist.
#   The SFNO frameworks need 0.9.x and get it from their own isolated venv instead.
#   zarr <3 — the PhysicsNeMo SeqZarr path targets the v2 API.
rm -rf "${TARGET}" && mkdir -p "${TARGET}"      # idempotent: never layer a stale resolve
# HOW THIS LIST WAS DERIVED (do not guess — re-derive it):
#   1. AST-scan the base-conda trees (PanguWeather/v2.0, si, s2s/v2.0, s2s-lightning) for
#      third-party top-level imports.
#   2. Import each with PYTHONNOUSERSITE=1 (a second member's view) + this dir on PYTHONPATH.
#   3. Anything missing that the installer HAS is a private-home leak and belongs here.
#   4. CAREFUL — "the installer lacks it too" does NOT prove it is unused. That only holds for
#      code paths that have actually RUN GREEN. The S2S/port smokes have never run on Polaris
#      (blocked on the ERA5 stage), so for them "missing for everyone" means BROKEN FOR
#      EVERYONE, not unused. An earlier version of this list made exactly that error and
#      dropped cf_xarray, which s2s-lightning/modules/train_module.py:52 imports bare — both
#      port entrypoints would have died at import the moment ERA5 landed, after a multi-TB
#      Globus stage, with the docs insisting the env was "proven by the probe".
#      dask / h5pickle / muon / transformer_engine are absent for everyone AND are not on any
#      entrypoint's import chain (verified by importing the entrypoints); cf_xarray was.
#
#   tensorly / tltorch / natsort / nvtx / cartopy — PanguWeather's SFNO stack imports both
#   (networks/modulus_sfno/{factorizations,layers}.py). They were ALSO only ever present as
#   private --user installs, so Pangu died with "No module named 'tensorly'" for anyone else
#   (caught by job 7253539, run with PYTHONNOUSERSITE=1 to impersonate a second member).
python -m pip install --no-cache-dir --target "${TARGET}" --no-deps --upgrade \
    "netCDF4==1.7.4" "zarr==2.18.7" "torch_harmonics==0.7.4" \
    "cftime" "numcodecs<0.16" "asciitree" "fasteners" \
    "tensorly==0.9.0" "tensorly-torch==0.5.0" \
    "natsort==8.4.0" "nvtx==0.2.15" "cf_xarray" \
    "cartopy==0.25.0" "shapely==2.1.2" "pyproj==3.7.2" "pyshp==3.1.4" \
    || { echo "ERROR PIP_FAILED"; exit 2; }

# --- guard: nothing in here may shadow ANY base-conda package ----------------
# A top-up dir on PYTHONPATH must ADD to the base env, never override it. An earlier version
# of this guard hardcoded a blocklist (torch/numpy/nvidia/triton) and therefore MISSED that
# `h5netcdf` is already in the base conda (1.6.4) — this dir was silently upgrading it to
# 1.8.1 for every job. Enumerate the target and ask the base env about each name instead:
# a blocklist only catches the shadowing you already thought of.
python - "${TARGET}" <<'_PY' || exit 2
import os, subprocess, sys
target = sys.argv[1]
names = set()
for e in os.listdir(target):
    if e.endswith(".dist-info") or e.endswith(".libs") or e.startswith("_") or e == "bin":
        continue
    n = e[:-3] if e.endswith(".py") else e
    if os.path.isdir(os.path.join(target, e)) or e.endswith(".py"):
        names.add(n)
# Ask a CLEAN base interpreter (no top-ups, no user site) what it can already import.
probe = "import importlib.util as u;print([n for n in %r if u.find_spec(n) is not None])" % sorted(names)
env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
env["PYTHONNOUSERSITE"] = "1"
out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
shadowed = eval(out.stdout.strip() or "[]")
if shadowed:
    print("ERROR TOPUPS_SHADOWS_BASE: these already exist in the base conda: %s" % ", ".join(shadowed))
    print("  $POLARIS_TOPUPS goes on PYTHONPATH, which OUTRANKS site-packages, so this dir")
    print("  would silently REPLACE the base's copy for every job — including the toolchain")
    print("  every GREEN smoke was validated against. Drop them from the pin list (the base")
    print("  already provides them) or justify the override explicitly.")
    sys.exit(2)
print("  no base-conda package is shadowed by the top-ups")
_PY

# World-readable: the whole point is that other members can import these.
chmod -R a+rX "${TARGET}" 2>/dev/null || true

echo "--- verifying the top-ups import with the USER site DISABLED (i.e. as another member) ---"
# PYTHONNOUSERSITE=1 reproduces a second member's view: they cannot see the installer's
# ~/.local. If this passes, the packages are genuinely coming from the shared dir.
PYTHONNOUSERSITE=1 PYTHONPATH="${TARGET}" python - "${TARGET}" <<'PY' || { echo "ERROR VERIFY_FAILED"; exit 2; }
import sys
target = sys.argv[1]
bad = [p for p in sys.path if "/.local/" in p]
assert not bad, "user-site leaked into sys.path: %s" % bad
for m in ("netCDF4", "zarr", "torch_harmonics", "cftime", "numcodecs",
          "tensorly", "tltorch", "natsort", "nvtx", "cartopy", "cf_xarray"):
    mod = __import__(m)
    assert "/.local/" not in mod.__file__, "%s resolved to a private home: %s" % (m, mod.__file__)
    print("  OK  %-16s %-10s %s" % (m, getattr(mod, "__version__", "?"), mod.__file__))
import torch_harmonics
assert torch_harmonics.__version__.startswith("0.7"), \
    "expected torch_harmonics 0.7.x for the base env, got %s" % torch_harmonics.__version__

# The base stack must survive the PYTHONPATH prepend untouched: these are what the GREEN
# smokes ran on, so if the top-ups displace them the whole baseline is void.
import torch, numpy
for mod, want in ((torch, "2.8"), (numpy, "2.2")):
    assert not mod.__file__.startswith(target), \
        "%s is being served BY THE TOP-UPS (%s) — it must come from the base conda" % (
            mod.__name__, mod.__file__)
    assert mod.__version__.startswith(want), \
        "%s is %s, expected the base conda's %s.x — the top-ups displaced it" % (
            mod.__name__, mod.__version__, want)
    print("  OK  %-16s %-10s (base conda, not shadowed)" % (mod.__name__, mod.__version__))
PY

echo "TOPUPS_OK"
