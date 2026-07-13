"""Validate that the most recent row in bench_test_results.csv looks sane.

Called by validate_bench.sh after bench_gpu_test.sh completes.
Exits 0 if all checks pass, 1 otherwise (which will block the downstream
SLURM dependency and prevent a wasted full-node job).

Checks
──────
- CSV file exists and has at least one data row
- All expected columns are present
- step_med  > 0 and < 60  (seconds; a step taking >1 min is clearly broken)
- samples_per_s > 0
- peak_mem_gb_max_rank > 0
- n_steps_counted == SNFO_BENCH_STEPS (default 20 for the smoke test)
"""

import csv
import os
import sys

CSV_PATH   = os.environ.get("SNFO_BENCH_CSV", "bench_test_results.csv")
EXPECTED_N = int(os.environ.get("SNFO_BENCH_STEPS", "20"))

REQUIRED_COLUMNS = [
    "timestamp", "git_sha", "run_num", "n_gpus", "batch_per_gpu",
    "precision", "step_med", "step_p90", "step_mean", "step_std",
    "samples_per_s", "peak_mem_gb_max_rank", "n_steps_counted",
]

errors = []

def fail(msg):
    errors.append(msg)
    print(f"  FAIL  {msg}")

def ok(msg):
    print(f"  OK    {msg}")

print(f"\n=== Validating {CSV_PATH} ===\n")

if not os.path.exists(CSV_PATH):
    fail(f"CSV file not found: {CSV_PATH}")
    sys.exit(1)

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    fail("CSV has no data rows")
    sys.exit(1)

row = rows[-1]  # most recent run
ok(f"CSV exists with {len(rows)} row(s); checking last row (run_num={row.get('run_num', '?')})")

# Column presence
missing = [c for c in REQUIRED_COLUMNS if c not in row]
if missing:
    fail(f"Missing columns: {missing}")
else:
    ok("All required columns present")

# Numeric sanity
try:
    step_med = float(row["step_med"])
    if step_med <= 0:
        fail(f"step_med={step_med:.4f} — must be > 0")
    elif step_med > 60:
        fail(f"step_med={step_med:.1f}s — unreasonably large (> 60s)")
    else:
        ok(f"step_med={step_med:.4f}s")
except (KeyError, ValueError) as e:
    fail(f"step_med parse error: {e}")

try:
    sps = float(row["samples_per_s"])
    if sps <= 0:
        fail(f"samples_per_s={sps:.3f} — must be > 0")
    else:
        ok(f"samples_per_s={sps:.3f}")
except (KeyError, ValueError) as e:
    fail(f"samples_per_s parse error: {e}")

try:
    mem = float(row["peak_mem_gb_max_rank"])
    if mem <= 0:
        fail(f"peak_mem_gb_max_rank={mem:.3f} — must be > 0")
    else:
        ok(f"peak_mem_gb_max_rank={mem:.3f} GB")
except (KeyError, ValueError) as e:
    fail(f"peak_mem_gb_max_rank parse error: {e}")

try:
    n = int(row["n_steps_counted"])
    if n != EXPECTED_N:
        fail(f"n_steps_counted={n}, expected {EXPECTED_N}")
    else:
        ok(f"n_steps_counted={n} (matches SNFO_BENCH_STEPS)")
except (KeyError, ValueError) as e:
    fail(f"n_steps_counted parse error: {e}")

print()
if errors:
    print(f"VALIDATION FAILED — {len(errors)} error(s). Full bench job will not run.")
    sys.exit(1)
else:
    print("Validation passed. Safe to proceed to full bench.")
    sys.exit(0)
