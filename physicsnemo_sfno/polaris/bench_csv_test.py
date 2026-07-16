#!/usr/bin/env python3
"""Self-running test for the PHYSICSNEMO_BENCH_CSV tee (bench_csv.py + its train.py wiring).

    python physicsnemo_sfno/polaris/bench_csv_test.py        # PASS = "BENCH_CSV_OK (N tests)"

Stdlib only — no torch/GPU/cluster needed, so it runs anywhere (a login node included;
it writes a few KB to a tempdir). What it CANNOT prove: that a real training run reaches
the hooks — that is the smoke's job (polaris_sfno_allyears_smoke.pbs exports the env var
and greps a real minibatch row, PASS token PHYSICSNEMO_CSV_OK). What it DOES prove that
the smoke cannot: the no-op-when-unset contract, the frozen schema, append-vs-resume, the
CUDA-graph aliasing defense, and — the subtree-pull tripwire — that train.py still
carries all five vendor-divergence blocks.
"""
import csv
import os
import sys
import tempfile
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1] / "examples" / "weather" / "unified_recipe"
sys.path.insert(0, str(RECIPE))
from bench_csv import ENV_VAR, FLUSH_EVERY, SCHEMA, BenchCSVTee  # noqa: E402

# The schema, RESTATED as a literal on purpose (same pattern as the allyears smoke's
# 103+5 pin): bench_csv.py deriving-from-itself can never catch its own drift. If this
# assert fires, someone changed the frozen schema — that breaks every existing consumer
# and needs a new file/column policy decision, not a test edit.
FROZEN = ["timestamp", "epoch", "step", "loss", "lr",
          "gb_per_s", "valid_error", "n_gpus", "git_sha", "run_name"]

PASSED = 0


def ok(name, cond, detail=""):
    # detail is failure context — printing it beside an "ok" would read as a problem.
    global PASSED
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail and not cond else ''}")
    if not cond:
        print(f"ERROR BENCH_CSV_TEST_FAILED: {name}")
        sys.exit(1)
    PASSED += 1


class FakeStaticScalar:
    """A CUDA-graph-style static buffer: one object whose value mutates in place.
    If the tee stores references instead of clones, every flushed row reads the LAST
    value — exactly the aliasing bug the clone exists to prevent."""

    def __init__(self, v):
        self.v = v

    def clone(self):
        return FakeStaticScalar(self.v)

    def __float__(self):
        return float(self.v)


def rows_of(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def main():
    tmp = tempfile.mkdtemp(prefix="bench_csv_test_")

    # -- 1. frozen schema ---------------------------------------------------------------
    ok("schema/frozen_literal", SCHEMA == FROZEN, f"SCHEMA={SCHEMA}")

    # -- 2. no-op contract: unset env / non-zero rank -----------------------------------
    os.environ.pop(ENV_VAR, None)
    ok("noop/env_unset", BenchCSVTee.from_env(rank=0, n_gpus=4, run_name="x") is None)
    os.environ[ENV_VAR] = os.path.join(tmp, "rank.csv")
    ok("noop/nonzero_rank", BenchCSVTee.from_env(rank=1, n_gpus=4, run_name="x") is None)
    ok("noop/nonzero_rank_wrote_nothing", not os.path.exists(os.environ[ENV_VAR]))

    # -- 3. the three row kinds land, parse, and carry the right cells ------------------
    path = os.path.join(tmp, "metrics.csv")
    os.environ[ENV_VAR] = path
    tee = BenchCSVTee.from_env(rank=0, n_gpus=4, run_name="Unified-Training")
    ok("from_env/rank0_returns_tee", tee is not None)
    for step, loss in enumerate([0.5, 0.25, 0.125]):
        tee.minibatch(epoch=0, step=step, loss=loss)
    tee.epoch(0, lr=1e-3, gb_per_s=2.5)
    tee.validation(0, valid_error=0.75)
    r = rows_of(path)
    ok("rows/header", r[0] == FROZEN)
    ok("rows/count", len(r) == 1 + 3 + 1 + 1, f"{len(r)} lines")
    body = r[1:]
    ok("rows/minibatch_losses",
       [row[3] for row in body[:3]] == ["0.5", "0.25", "0.125"]
       and [row[2] for row in body[:3]] == ["0", "1", "2"])
    ok("rows/minibatch_empties", all(row[4] == row[5] == row[6] == "" for row in body[:3]))
    ok("rows/epoch_row", body[3][4] == "0.001" and body[3][5] == "2.5"
       and body[3][2] == "" and body[3][3] == "" and body[3][6] == "")
    ok("rows/validation_row", body[4][6] == "0.75" and body[4][3] == "" and body[4][4] == "")
    ok("rows/constants_every_row",
       all(row[7] == "4" and row[9] == "Unified-Training" for row in body))

    # -- 4. CUDA-graph aliasing defense: a mutating buffer must be cloned ---------------
    path2 = os.path.join(tmp, "alias.csv")
    os.environ[ENV_VAR] = path2
    tee2 = BenchCSVTee.from_env(rank=0, n_gpus=1, run_name="alias")
    buf = FakeStaticScalar(1.0)
    tee2.minibatch(0, 0, buf)
    buf.v = 2.0  # the "next step" overwrites the static buffer in place
    tee2.minibatch(0, 1, buf)
    buf.v = 3.0
    tee2.epoch(0, lr=0.0, gb_per_s=0.0)  # forces the flush
    vals = [row[3] for row in rows_of(path2)[1:3]]
    ok("alias/clone_preserves_per_step_values", vals == ["1.0", "2.0"],
       f"got {vals} (['3.0', '3.0'] means the clone was dropped)")

    # -- 5. buffering: FLUSH_EVERY forces rows out mid-epoch, none are lost -------------
    path3 = os.path.join(tmp, "buffer.csv")
    os.environ[ENV_VAR] = path3
    tee3 = BenchCSVTee.from_env(rank=0, n_gpus=1, run_name="buf")
    for step in range(FLUSH_EVERY + 50):
        tee3.minibatch(0, step, float(step))
    mid = len(rows_of(path3)) - 1
    ok("buffer/flushes_at_threshold", mid == FLUSH_EVERY,
       f"{mid} rows on disk mid-epoch (want exactly {FLUSH_EVERY}; 0 means a kill "
       f"loses the whole epoch, {FLUSH_EVERY + 50} means per-step writes)")
    tee3.epoch(0, lr=0.0, gb_per_s=0.0)
    ok("buffer/epoch_drains_the_rest",
       len(rows_of(path3)) == 1 + FLUSH_EVERY + 50 + 1)

    # -- 6. resume appends without a second header ---------------------------------------
    tee4 = BenchCSVTee.from_env(rank=0, n_gpus=1, run_name="buf")
    tee4.validation(1, valid_error=0.5)
    r = rows_of(path3)
    ok("resume/no_duplicate_header",
       sum(1 for row in r if row == FROZEN) == 1 and r[-1][6] == "0.5")

    # -- 7. schema mismatch refuses loudly (never silently corrupts an old file) --------
    path4 = os.path.join(tmp, "wrong.csv")
    with open(path4, "w") as f:
        f.write("some,other,header\n1,2,3\n")
    os.environ[ENV_VAR] = path4
    try:
        BenchCSVTee.from_env(rank=0, n_gpus=1, run_name="x")
        ok("schema/mismatch_raises", False)
    except RuntimeError as e:
        ok("schema/mismatch_raises", "schema mismatch" in str(e))

    # -- 8. the subtree-pull tripwire: train.py still carries all five blocks -----------
    src = (RECIPE / "train.py").read_text()
    for needle, label in [
        ("from bench_csv import BenchCSVTee", "import"),
        ("BenchCSVTee.from_env", "factory"),
        ("bench_csv.minibatch(", "minibatch hook"),
        ("bench_csv.epoch(", "epoch hook"),
        ("bench_csv.validation(", "validation hook"),
    ]:
        ok(f"wiring/{label}", needle in src,
           f"train.py lost '{needle}' — a subtree pull dropped the vendor divergence")
    ok("wiring/rank_gate_still_upstream",
       "if dist.rank == 0:" in src,
       "the rank-0 validation gate the tee's validation hook relies on is gone")

    print(f"\nBENCH_CSV_OK ({PASSED} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
