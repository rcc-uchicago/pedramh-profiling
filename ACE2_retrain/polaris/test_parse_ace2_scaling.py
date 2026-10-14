#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Tests for parse_ace2_scaling.py -- no allocation, no GPU, no torch.

    python3.11 ACE2_retrain/polaris/test_parse_ace2_scaling.py
    pytest -q ACE2_retrain/polaris/test_parse_ace2_scaling.py

PASS = ACE2_SCALING_PARSE_OK.

The point is not the arithmetic; it is the silent-failure guards. An ACE2 row
that came from N independent world_size=1 trainers (fme falls back to
NonDistributed the moment the rank shim does not apply -- no error, no warning),
from a TCP fallback instead of the Slingshot plugin, from 12 surviving ranks out
of 16, or from a walltime-truncated arm is a plausible number that means nothing
-- and every one of those costs real allocation time to discover any other way.

The cross-project column contract is tested too: this table has to concatenate
with ai-rossby's and makani's, so a column added here alone is a defect.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_ace2_scaling as P  # noqa: E402

# Stand-in for what the harness actually prints, kept verbatim in shape so a
# reworded ace2_telemetry.py breaks a test rather than a campaign.
#
# ⚠ The prefix is COPIED FROM A REAL LOG (job 7586496), not assumed. PALS
# `--label` emits "<hostname> <rank>: ", not the bare "<rank>: " the ai-rossby
# parser this was adapted from expects. Getting that wrong does not fail loudly:
# ranks_reporting falls back to counting banner lines, and the two world-size
# guards -- which exist to be independent -- silently become one.
PALS = "x3005c0s13b0n0.hsn.cm.polaris.alcf.anl.gov {r}: "
BANNER = (PALS + "ACE2_BANNER steps_per_epoch={spe} world_size={w} rank={r} "
          "local_batch=1 global_batch={w} device=cuda:{d} torch=2.10.0+cu129")


def good_log(ranks=16, world=None, spe=60):
    world = ranks if world is None else world
    lines = [PALS.format(r=0) + "x3005c0s13b0n0:1203158:1203212 [0] NCCL INFO "
             "Using network AWS Libfabric",
             PALS.format(r=1) + "x3005c0s13b0n0:1203159:1203213 [1] NCCL INFO "
             "Using network AWS Libfabric"]
    lines += [BANNER.format(r=r, w=world, d=r % 4, spe=spe) for r in range(ranks)]
    lines.append("0: EPOCH_TELEMETRY epoch=1 n=60 step_med=512.7ms gpu_busy=94.1% "
                 "peak=28.44GB ema=1 -> /x/tel.csv")
    return "\n".join(lines) + "\n"


TEL_COLUMNS = ["timestamp", "harness", "git_sha", "run_name", "host", "epoch",
               "n_gpus", "batch_per_gpu", "amp_dtype", "n_loaders", "n_steps",
               "step_med_ms", "step_p90_ms", "step_mean_ms", "step_std_ms",
               "samples_per_s", "epoch_wall_s", "gpu_busy_frac", "peak_mem_gb",
               "ema_active", "lr"]

RUN_NAME = "ace2_mn4n_b16_rep1_7600001"


def tel_row(run_name=RUN_NAME, n_gpus=16, n_steps=60, step_med=512.7,
            gpu_busy=0.941):
    return {
        "timestamp": "2026-09-02T10:05:00", "harness": "ace2",
        "git_sha": "abc123def456", "run_name": run_name, "host": "x3001c0s1b0n0",
        "epoch": 1, "n_gpus": n_gpus, "batch_per_gpu": 1, "amp_dtype": "bf16",
        "n_loaders": 4, "n_steps": n_steps, "step_med_ms": step_med,
        "step_p90_ms": 540.2, "step_mean_ms": 515.9, "step_std_ms": 12.3,
        "samples_per_s": 29.8, "epoch_wall_s": 32.2, "gpu_busy_frac": gpu_busy,
        "peak_mem_gb": 28.44, "ema_active": 1, "lr": 1e-04,
    }


BASE_ARGV = [
    "--nodes", "4",
    "--ranks", "16",
    "--local-batch", "1",
    "--global-batch", "16",
    "--steps", "60",
    "--data", "nc",
    "--run-name", RUN_NAME,
    "--jobid", "7600001",
    "--env-source", "manual-reconstruction",
    "--omp-threads", "2",
]


def run(log_text, tel_rows, extra_argv=(), csv_seed=None):
    """Run main() on a temp log + telemetry CSV. Returns (rc, row, stdout)."""
    tmp = tempfile.mkdtemp()
    log = os.path.join(tmp, "arm.log")
    tel = os.path.join(tmp, "tel.csv")
    out = os.path.join(tmp, "scaling.csv")
    with open(log, "w") as fh:
        fh.write(log_text)
    with open(tel, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TEL_COLUMNS)
        w.writeheader()
        for r in tel_rows:
            w.writerow(r)
    if csv_seed is not None:
        with open(out, "w", newline="") as fh:
            fh.write(csv_seed)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = P.main(list(BASE_ARGV) + ["--log", log, "--csv", out,
                                       "--telemetry-csv", tel] + list(extra_argv))
    rows = []
    if os.path.exists(out):
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
    return rc, (rows[-1] if rows else None), buf.getvalue()


# --- the contract ---------------------------------------------------------


def test_fields_match_ai_rossby_verbatim():
    """The two tables must concatenate; a column added on one side alone breaks it.

    Read out of the sibling parser's SOURCE rather than imported: the two live in
    different trees with colliding top-level module names, and importing both
    into one interpreter is exactly the PYTHONPATH hazard CLAUDE.md warns about.
    """
    import ast

    sibling = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "physicsnemo_ai_rossby", "polaris", "parse_ai_rossby_scaling.py",
    )
    tree = ast.parse(open(sibling).read())
    fields = next(
        ast.literal_eval(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FIELDS" for t in n.targets)
    )
    assert P.FIELDS == fields, (
        "ACE2's scaling columns have drifted from ai-rossby's; the two tables no "
        "longer concatenate (CLAUDE.md #10)"
    )


def test_row_keys_match_fields():
    rc, row, _ = run(good_log(), [tel_row()])
    assert row is not None
    assert list(row) == P.FIELDS


# --- the happy path -------------------------------------------------------


def test_clean_run_passes_and_derives_throughput():
    rc, row, out = run(good_log(), [tel_row()])
    assert rc == 0, out
    assert row["transport"] == "AWS Libfabric"
    assert row["world_sizes_seen"] == "16"
    assert row["ranks_reporting"] == "16"
    assert row["n_steps"] == "60"
    # 1 sample / 0.5127 s = 1.9505 samples/s/rank, x16 ranks
    assert abs(float(row["samples_s_rank"]) - 1.9505) < 1e-3
    assert abs(float(row["samples_s_total"]) - 31.208) < 1e-2
    assert row["data"] == "nc"


def test_last_row_for_this_run_wins_not_the_last_row_in_the_file():
    """The telemetry CSV is append-only and shared across arms."""
    rows = [tel_row(run_name="some_other_arm", step_med=999.9),
            tel_row(step_med=100.0),
            tel_row(run_name="a_later_other_arm", step_med=888.8)]
    rc, row, out = run(good_log(), rows)
    assert rc == 0, out
    assert float(row["step_med_ms"]) == 100.0


# --- the five guards ------------------------------------------------------


def test_missing_rank_shim_is_caught():
    """fme silently builds NonDistributed when RANK is absent -- 16 lone trainers."""
    rc, row, out = run(good_log(ranks=16, world=1), [tel_row()])
    assert rc == 4
    assert "WORLD_SIZE_MISMATCH" in out
    assert row["world_sizes_seen"] == "1"


def test_dead_ranks_are_caught():
    rc, row, out = run(good_log(ranks=12, world=16), [tel_row()])
    assert rc == 4
    assert "RANKS_REPORTING_MISMATCH" in out


def test_absent_banner_is_an_error_not_a_pass():
    """No banner means the guards did not run -- not that they passed."""
    rc, row, out = run("0: NCCL INFO NET/OFI Using network AWS Libfabric\n",
                       [tel_row()])
    assert rc == 4
    assert "NO_TRAINER_BANNER" in out
    # and it names the likely cause rather than leaving it to be rediscovered
    assert "ace2_telemetry.py" in out


def test_truncated_arm_is_caught():
    rc, row, out = run(good_log(), [tel_row(n_steps=37)])
    assert rc == 4
    assert "STEP_COUNT_MISMATCH" in out


def test_epoch_longer_than_the_arm_is_caught():
    """The wrap guard: a loader serving more steps than the arm timed.

    That is what a dropped `sample_with_replacement` override looks like, and the
    resulting bias moves with the global batch -- i.e. along the axis being swept.
    """
    rc, row, out = run(good_log(spe=5475), [tel_row()])
    assert rc == 4
    assert "EPOCH_LENGTH_MISMATCH" in out


def test_telemetry_ngpus_disagreement_is_caught():
    """A second, independent view of the world size."""
    rc, row, out = run(good_log(), [tel_row(n_gpus=8)])
    assert rc == 4
    assert "TELEMETRY_NGPUS_MISMATCH" in out


def test_no_telemetry_row_is_fatal():
    rc, row, out = run(good_log(), [tel_row(run_name="a_different_arm")])
    assert rc == 4
    assert "NO_TELEMETRY_ROW" in out


def test_unknown_transport_warns_but_does_not_fail():
    """A timing is still a timing; it just cannot be tabled as fabric evidence."""
    log = "\n".join(BANNER.format(r=r, w=16, d=r % 4, spe=60) for r in range(16))
    rc, row, out = run(log + "\n", [tel_row()])
    assert rc == 0, out
    assert row["transport"] == "UNKNOWN"
    assert "NO_TRANSPORT_LINE" in out


def test_ranks_reporting_reads_pals_labels_not_banner_count():
    """The two world-size guards must stay independent.

    If the label regex misses, `ranks_reporting` falls back to counting banner
    lines -- which is the same quantity `world_sizes_seen` is derived from, so
    the second opinion quietly becomes an echo of the first. Distinct ranks
    emitting the SAME banner text is the case that separates them.
    """
    dup = "\n".join(BANNER.format(r=7, w=16, d=3, spe=60) for _ in range(16))
    parsed = P.parse_log(dup + "\n")
    assert parsed["_banner_lines"] == 16      # 16 lines...
    assert parsed["ranks_reporting"] == 1     # ...but one rank wrote all of them


def test_label_regex_does_not_read_a_timestamp_as_a_rank():
    line = ("x3005c0s13b0n0.hsn.cm.polaris.alcf.anl.gov 0: 2026-09-02 20:33:12,949 "
            "ACE2_BANNER steps_per_epoch=60 world_size=4 rank=0 local_batch=1")
    parsed = P.parse_log(line + "\n")
    assert parsed["ranks_reporting"] == 1


def test_two_word_transport_name_is_not_truncated():
    """`\\S+` would record "AWS Libfabric" as "AWS" -- a different transport."""
    parsed = P.parse_log("0: NCCL INFO NET/OFI Using network AWS Libfabric\n")
    assert parsed["transport"] == "AWS Libfabric"


# --- ACE2-specific --------------------------------------------------------


def test_low_gpu_busy_frac_warns_and_points_at_the_loader():
    """The first question for ACE2, not an afterthought.

    Every rank reads the same 2.4 TB NetCDF on one Lustre OST. If the epoch is
    mostly loader wait, a comms analysis of the result is meaningless -- so the
    parser has to say so, in the row's own output, at the moment it is produced.
    """
    rc, row, out = run(good_log(), [tel_row(gpu_busy=0.42)])
    assert rc == 0, out          # a valid measurement of a different thing
    assert "LOW_GPU_BUSY_FRAC" in out
    assert "getstripe" in out


def test_high_gpu_busy_frac_is_silent():
    rc, row, out = run(good_log(), [tel_row(gpu_busy=0.976)])
    assert rc == 0
    assert "LOW_GPU_BUSY_FRAC" not in out


def test_banner_regex_ignores_anything_the_launcher_echoed():
    """A launcher header saying world_size=16 must not satisfy the guard."""
    log = ("=== ace2 1node: nodes=1 ranks=4 world_size=16 ===\n"
           "steps_per_epoch=60 world_size=16 (echoed by the launcher)\n")
    parsed = P.parse_log(log)
    assert parsed["_banner_lines"] == 0
    assert parsed["world_sizes_seen"] == ""


def test_schema_drift_refuses_to_append():
    seed = "jobid,nodes,ranks\n1,1,4\n"
    try:
        run(good_log(), [tel_row()], csv_seed=seed)
    except SystemExit as exc:
        assert "SCALING_CSV_SCHEMA_DRIFT" in str(exc)
    else:
        raise AssertionError("appending onto a drifted header must abort")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print("ERROR %s: %s" % (t.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s raised %s: %s" % (t.__name__, type(exc).__name__, exc))
    if failed:
        print("ERROR ACE2_SCALING_PARSE_FAILED (%d/%d)" % (failed, len(tests)))
        sys.exit(1)
    print("ACE2_SCALING_PARSE_OK (%d tests)" % len(tests))
