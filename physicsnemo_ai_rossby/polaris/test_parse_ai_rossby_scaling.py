#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Tests for parse_ai_rossby_scaling.py -- no allocation, no GPU, no torch.

    python physicsnemo_ai_rossby/polaris/test_parse_ai_rossby_scaling.py
    pytest -q physicsnemo_ai_rossby/polaris/test_parse_ai_rossby_scaling.py

PASS = AI_ROSSBY_SCALING_PARSE_OK.

The point is not the arithmetic; it is the silent-failure guards.  A multi-node
scaling row that came from N independent world_size=1 trainers, from a TCP
fallback instead of the Slingshot plugin, from 12 surviving ranks out of 16, or
from a walltime-truncated 37-step arm is a plausible number that means nothing
-- and every one of those costs real allocation time to discover any other way.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_ai_rossby_scaling as P  # noqa: E402

# Stand-ins for what the harness actually prints, kept verbatim in shape so a
# reworded train.py breaks a test rather than a campaign.  The banner is
# train.py:887 (every rank); PALS `--label` supplies the "<rank>: " prefix.
BANNER = ("{r}: [2026-08-27 10:00:0{r}][pangu_plasim_train][INFO] - "
          "steps_per_epoch=2737, stages=1, total_epochs=1, world_size={w}, "
          "device=cuda:{d}")


def good_log(ranks=16, world=None):
    world = ranks if world is None else world
    lines = ["0: [rank0] NCCL INFO NET/OFI Using network AWS Libfabric",
             "1: [rank1] NCCL INFO NET/OFI Using network AWS Libfabric"]
    lines += [BANNER.format(r=r, w=world, d=r % 4) for r in range(ranks)]
    lines.append("0: EPOCH_TELEMETRY epoch=1 n=60 step_med=512.7ms gpu_busy=94.1% "
                 "peak=28.44GB ema=1 -> /x/tel.csv")
    lines.append("SFNO_E3SM_MULTINODE_OK")
    return "\n".join(lines) + "\n"


TEL_COLUMNS = ["timestamp", "harness", "git_sha", "run_name", "host", "epoch",
               "n_gpus", "batch_per_gpu", "amp_dtype", "n_loaders", "n_steps",
               "step_med_ms", "step_p90_ms", "step_mean_ms", "step_std_ms",
               "samples_per_s", "epoch_wall_s", "gpu_busy_frac", "peak_mem_gb",
               "ema_active", "lr"]

RUN_NAME = "ar_mn4n_b16_rep1_7600001"


def tel_row(run_name=RUN_NAME, n_gpus=16, n_steps=60, step_med=512.7):
    return {
        "timestamp": "2026-08-27T10:05:00", "harness": "ai_rossby",
        "git_sha": "abc123def456", "run_name": run_name, "host": "x3001c0s1b0n0",
        "epoch": 1, "n_gpus": n_gpus, "batch_per_gpu": 1, "amp_dtype": "bf16",
        "n_loaders": 1, "n_steps": n_steps, "step_med_ms": step_med,
        "step_p90_ms": 540.2, "step_mean_ms": 515.9, "step_std_ms": 12.3,
        "samples_per_s": 29.8, "epoch_wall_s": 32.2, "gpu_busy_frac": 0.941,
        "peak_mem_gb": 28.44, "ema_active": 1, "lr": 1.22e-04,
    }


BASE_ARGV = [
    "--nodes", "4",
    "--ranks", "16",
    "--local-batch", "1",
    "--global-batch", "16",
    "--steps", "60",
    "--run-name", RUN_NAME,
    "--jobid", "7600001",
    "--env-source", "manual-reconstruction",
    "--omp-threads", "1",
]


def _write_tel(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TEL_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _run(text, tmpdir, tel_rows=None, extra=None, csv_name="scaling.csv"):
    log = os.path.join(tmpdir, "run.log")
    with open(log, "w") as fh:
        fh.write(text)
    tel = os.path.join(tmpdir, "tel.csv")
    _write_tel(tel, tel_rows if tel_rows is not None else [tel_row()])
    csv_path = os.path.join(tmpdir, csv_name)
    argv = (["--log", log, "--csv", csv_path, "--telemetry-csv", tel]
            + BASE_ARGV + (extra or []))
    rc = P.main(argv)
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    return rc, rows, csv_path


FAILURES = []


def check(cond, name):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s" % name)
        FAILURES.append(name)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        # 1. Happy path: rc 0, exactly one row, schema intact.
        rc, rows, csv_path = _run(good_log(), td)
        check(rc == 0, "green run exits 0")
        check(len(rows) == 1, "one row written")
        check(list(rows[0]) == P.FIELDS, "csv header == FIELDS")

        # 2. Transport keeps BOTH words. "AWS" alone reads as a different stack.
        check(rows[0]["transport"] == "AWS Libfabric", "two-word transport kept")

        # 3. Both world-size views agree with the launch.
        check(rows[0]["world_sizes_seen"] == "16", "world_size read off the banner")
        check(rows[0]["ranks_reporting"] == "16", "16 distinct rank labels counted")

        # 4. Timings come from the telemetry CSV, at full precision.
        check(rows[0]["step_med_ms"] == "512.7", "step_med from telemetry csv")
        check(rows[0]["step_p90_ms"] == "540.2", "p90 survives (not in the print)")
        # 1 sample / 0.5127 s = 1.9505 per rank; x16 ranks.
        check(rows[0]["samples_s_rank"] == "1.9505", "per-rank samples/s")
        check(rows[0]["samples_s_total"] == "31.2073", "total samples/s")

        # 5. Appending a second row does not rewrite the header.
        rc2, rows2, _ = _run(good_log(), td)
        check(rc2 == 0 and len(rows2) == 2, "second row appends")

    with tempfile.TemporaryDirectory() as td:
        # 6. THE guard: N independent world_size=1 trainers.
        rc, rows, _ = _run(good_log(ranks=16, world=1), td)
        check(rc == 4, "world_size=1 for a 16-rank launch is rejected")
        check(len(rows) == 1, "the bad row is still written, for the record")

    with tempfile.TemporaryDirectory() as td:
        # 7. Ranks that never reached the banner. world_size stays 16, so this
        #    isolates the second guard: the first one sees nothing wrong.
        rc, rows, _ = _run(good_log(ranks=12, world=16), td)
        check(rc == 4, "12 reporting ranks out of 16 is rejected")
        check(rows[0]["world_sizes_seen"] == "16",
              "...even though world_size itself looked right")

    with tempfile.TemporaryDirectory() as td:
        # 8. A launcher echo must not be able to satisfy the world-size guard.
        text = ("ranks = 16   world_size=16\n"
                "0: [rank0] NCCL INFO Using network AWS Libfabric\n")
        rc, rows, _ = _run(text, td)
        check(rc == 4, "no trainer banner is an ERROR, not a silent pass")
        check(rows[0]["world_sizes_seen"] == "", "launcher echo not counted")

    with tempfile.TemporaryDirectory() as td:
        # 9. No transport line: a WARN, not a failure -- the timing is still real.
        text = "\n".join(BANNER.format(r=r, w=16, d=r % 4) for r in range(16)) + "\n"
        rc, rows, _ = _run(text, td)
        check(rc == 0, "missing transport line warns but does not fail")
        check(rows[0]["transport"] == "UNKNOWN", "transport recorded as UNKNOWN")

    with tempfile.TemporaryDirectory() as td:
        # 10. A truncated arm.
        rc, _, _ = _run(good_log(), td, tel_rows=[tel_row(n_steps=37)])
        check(rc == 4, "37 timed steps for a 60-step arm is rejected")

    with tempfile.TemporaryDirectory() as td:
        # 11. The trainer's own idea of the world size, independent of the log.
        rc, _, _ = _run(good_log(), td, tel_rows=[tel_row(n_gpus=4)])
        check(rc == 4, "telemetry n_gpus != launched ranks is rejected")

    with tempfile.TemporaryDirectory() as td:
        # 12. No telemetry row at all (epoch never ended / knob unset).
        rc, _, _ = _run(good_log(), td, tel_rows=[])
        check(rc == 4, "absent telemetry row is not a measurement")

    with tempfile.TemporaryDirectory() as td:
        # 13. The row for THIS run is selected, not the file's last row.
        rows_in = [tel_row(run_name="some_earlier_arm", step_med=111.1),
                   tel_row(step_med=222.2),
                   tel_row(run_name="a_later_arm", step_med=999.9)]
        rc, rows, _ = _run(good_log(), td, tel_rows=rows_in)
        check(rc == 0 and rows[0]["step_med_ms"] == "222.2",
              "telemetry row matched on run_name, not position")

    with tempfile.TemporaryDirectory() as td:
        # 14. Schema drift refuses to append rather than corrupting the file.
        csv_path = os.path.join(td, "old.csv")
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["jobid", "nodes", "step_ms"])
        drifted = False
        try:
            _run(good_log(), td, csv_name="old.csv")
        except SystemExit as exc:
            drifted = "SCALING_CSV_SCHEMA_DRIFT" in str(exc)
        check(drifted, "a drifted header refuses the append")

    if FAILURES:
        print("ERROR AI_ROSSBY_SCALING_PARSE_FAILED: %d checks failed: %s"
              % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("AI_ROSSBY_SCALING_PARSE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
