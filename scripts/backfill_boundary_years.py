#!/usr/bin/env python3
"""backfill_boundary_years.py — emit the 7 early-year MOST.{YYYY}.h5 boundary
files the EXP15_heatwave_AIRES full config needs but that the v11 build never
produced (its split was train=[12,111], so years < 12 were never packed).

Required output years: 0002, 0004, 0005, 0006, 0008, 0009, 0010
(IC years 0007/0009/0010/0011/0013/0014/0015 → MOST.{ic_year-5}.h5; see
 AI-RES-clean/docs/2026-05-27_sfno_boundary_backfill_task.md).

This is a thin wrapper over `plasim_makani_packager.packager.process_one` — it
reuses the EXACT packing code that produced the existing 100 train files, so the
HDF5 layout/attrs are byte-for-byte consistent (only the source year differs).
Inputs and parameters are pinned to the v11 dataset provenance recorded in
  .../sim52_astro_64x128_zgplev_v11/metadata/data.json :
    source_postproc_root = .../data/postproc/sim52
    source_boundary_root = .../data/boundary_astro_v11/sim52
    sst_mode=surface, sst_land_fill_K=271.35, rsdt=astronomical.

Year 0002 is a packager WARMUP year (packager.WARMUP_YEARS = {1,2}), normally
skipped. We bypass that gate here because:
  * the warmup designation guards *training* spin-up and the per-split
    /timestamp offset — neither applies to inference-time boundary backfill;
  * the SFNO bridge (AI-RES-clean/forecast_modules/PanguPlasimFS/
    sfno_bridge_dataset.py:112-118) reads ONLY /forcing + the plasim_time_units
    attr from these files — never /fields_state or /timestamp. Year-2 forcing
    (lsm/sg/z0/sst/rsdt/sic — static + boundary/astronomical fields) is
    physically appropriate as cyclic boundary forcing.

Run:
    cd /home1/11114/zhixingliu/projects/SFNO_Climate_Emulator
    source .venv/bin/activate
    PYTHONPATH=src python3 scripts/backfill_boundary_years.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from plasim_makani_packager import packager

# v11 provenance (metadata/data.json). Roots are the *parents* of sim{NN}/.
POSTPROC_ROOT = Path("/scratch/11114/zhixingliu/AI-RES/data/postproc")
BOUNDARY_ROOT = Path("/scratch/11114/zhixingliu/AI-RES/data/boundary_astro_v11")
OUTPUT_ROOT = Path(
    "/scratch/11114/zhixingliu/AI-RES/data/makani/sim52_astro_64x128_zgplev_v11"
)
SIM = 52
SST_LAND_FILL_K = 271.35
# Postprocessor SHA recorded on the existing 100 files (packager_git_sha on the
# new files reflects current HEAD; the .nc inputs are unchanged).
POSTPROCESSOR_GIT_SHA = "8b395ebe51c495994aca9f28ce5450b196629ed2"

TARGET_YEARS = (2, 4, 5, 6, 8, 9, 10)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Bypass the warmup gate so year 2 is emitted (see module docstring).
    packager.WARMUP_YEARS = frozenset()

    # train_years lo=2 so every target year classifies as "train"; valid/test
    # ranges mirror the v11 build. (Only affects split routing + the unread
    # /timestamp offset; existing files are never touched — overwrite=False.)
    opts = argparse.Namespace(
        postproc_root=POSTPROC_ROOT,
        boundary_root=BOUNDARY_ROOT,
        output_root=OUTPUT_ROOT,
        train_years=(2, 111),
        valid_years=(11, 11),
        test_years=(121, 128),
        sst_land_fill_k=SST_LAND_FILL_K,
        postprocessor_git_sha=POSTPROCESSOR_GIT_SHA,
        overwrite=False,
        dry_run=False,
        verbose=True,
    )

    for year in TARGET_YEARS:
        packager.process_one(SIM, year, opts)


if __name__ == "__main__":
    main()
