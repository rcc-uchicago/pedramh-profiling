#!/bin/bash
# package_sim52_astro.sh — orchestrator for the sim52 astronomical-rsdt
# packaging pipeline. Runs the four local stages (stats, metadata, validate)
# after the packager SLURM array has finished. Packager + adaptor are
# dispatched separately via src/emulator_adaptor/submit.slurm and
# src/plasim_makani_packager/submit.slurm.
#
# This is a user-facing checklist; edit and source as needed. Not a
# turn-key driver.

set -euo pipefail

POSTPROC_ROOT="${POSTPROC_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/postproc}"
BOUNDARY_ROOT="${BOUNDARY_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/boundary_astro}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sim52_astro_64x128}"
SIMS="${SIMS:-52}"
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "=== [1/4] Phase 0: boundary adaptor (run via src/emulator_adaptor/submit.slurm) ==="
echo "        sbatch --array=0-\$((N-1)) src/emulator_adaptor/submit.slurm"
echo "        (set SIMS / YEAR_START / YEAR_END / INPUT_ROOT / OUTPUT_ROOT / RSDT_METHOD=astronomical)"
echo

echo "=== [2/4] Phase 1: packager (run via src/plasim_makani_packager/submit.slurm) ==="
N=$(python3 -m plasim_makani_packager.packager --sims $SIMS --count-tasks)
echo "        task count = $N"
echo "        sbatch --array=0-$((N-1)) src/plasim_makani_packager/submit.slurm"
echo

echo "=== [3/4] Phase 2: stats (training split only) ==="
python3 -m plasim_makani_packager.stats --output-root "$OUTPUT_ROOT" -v

echo "=== [3.5/4] Phase 3: metadata + config render ==="
python3 -m plasim_makani_packager.metadata \
    --output-root "$OUTPUT_ROOT" \
    --exp-dir "$EXP_DIR" \
    --rsdt-method astronomical \
    --sst-land-fill-k 271.35 \
    -v

echo "=== [4/4] Phase 4a: structural validation ==="
python3 -m plasim_makani_packager.validate \
    --output-root "$OUTPUT_ROOT" --mode structural -v

echo
echo "Packager pipeline complete."
echo "Next: Phase 4b Makani smoke test (requires makani + physicsnemo env):"
echo "    python3 -m plasim_makani_packager.validate --output-root $OUTPUT_ROOT --mode makani_smoke"
