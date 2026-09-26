#!/usr/bin/env python3
"""Guard for a running fme (ACE2) training job. ONE process, no numpy, never calls qstat.

    python3 monitor/ace2/ace2_train_watch.py <exp_dir> <o_file> <runtime_s> [--deadline ISO] [--poll 300]
    python3 monitor/ace2/ace2_train_watch.py --selftest

Emits one line per EVENT and nothing otherwise:
  EPOCH    a new "Time taken for epoch" in out.log: wall, train loss, lr, best val, projection vs deadline
  CKPT     a file appeared/changed in training_checkpoints/ (name, size, mtime)
  STALL    out.log not written for --stall-min minutes while the deadline has not passed
  ERR      Traceback / nan loss / OutOfMemory / NCCL error lines newly appended to the .o file
  END      TRAIN_DONE / ACE2_POLARIS_TRAIN_OK / ACE2_POLARIS_TRAIN_FAILED in the .o file
  DEADLINE T-2h and T-0 of the walltime
Run it under a long-timeout background shell and re-arm when it expires (memory: pid cap 256, threads count).
Python 3.6-compatible on purpose: the login node's system python3 is 3.6 and the venv must not be used there.
"""
import argparse
import datetime as dt
import os
import re
import sys
import time

EPOCH_RE = re.compile(r"Time taken for epoch (\d+) is ([0-9.]+) sec")
TRAIN_RE = re.compile(r"Train loss: ([0-9.eE+-]+|nan)")
LR_RE = re.compile(r"\blr: ([0-9.eE+-]+)")
BEST_RE = re.compile(r"(\d+) complete epochs and 0 additional batches.*best_validation_loss ([0-9.]+)")
ERR_RE = re.compile(r"Traceback \(most recent call last\)|OutOfMemoryError|CUDA out of memory|NCCL (error|WARN)|Watchdog caught|Train loss: nan|ACE2_LOSS_NOT_FINITE|^ERROR |: ERROR ")
END_RE = re.compile(r"TRAIN_DONE|ACE2_POLARIS_TRAIN_OK|ACE2_POLARIS_TRAIN_FAILED|=== .*done")


def parse_epoch_block(lines):
    """From lines appended since the last tick, return the completed epochs as dicts."""
    out, cur = [], None
    for l in lines:
        m = EPOCH_RE.search(l)
        if m:
            cur = {"epoch": int(m.group(1)), "wall_s": float(m.group(2)), "t": l[:19]}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = TRAIN_RE.search(l)
        if m and "train" not in cur:
            cur["train"] = m.group(1)
        m = LR_RE.search(l)
        if m and "lr" not in cur:
            cur["lr"] = float(m.group(1))
        m = BEST_RE.search(l)
        if m and int(m.group(1)) == cur["epoch"]:
            cur["best_val"] = float(m.group(2))
    return out


def selftest() -> int:
    canned = [
        "2026-09-26 01:00:34,063 - root - INFO - Time taken for epoch 7 is 11172.85 sec",
        "2026-09-26 01:00:34,063 - root - INFO - Train loss: 0.1786961406469345",
        "2026-09-26 01:00:34,068 - root - INFO -     lr: 7.575000000000001e-05",
        "2026-09-26 01:00:40,000 - root - INFO - Saving latest checkpoint model trained for 7 complete epochs and 0 additional batches, or 85645 total batches, with best_validation_loss 0.18446573",
        "2026-09-26 01:07:00,000 - root - INFO - Saving latest checkpoint model trained for 7 complete epochs and 500 additional batches, or 86145 total batches, with best_validation_loss 0.18446573",
    ]
    ep = parse_epoch_block(canned)
    assert len(ep) == 1 and ep[0]["epoch"] == 7 and ep[0]["train"] == "0.1786961406469345"
    assert abs(ep[0]["lr"] - 7.575e-05) < 1e-12 and ep[0]["best_val"] == 0.18446573
    assert parse_epoch_block(canned[1:]) == []                      # no epoch line → nothing
    assert ERR_RE.search("x 0: Train loss: nan") and not ERR_RE.search("Train loss: 0.2")
    assert ERR_RE.search("[rank0]: Traceback (most recent call last):") and not ERR_RE.search("INFO - Traceback-free")
    assert ERR_RE.search("ERROR ACE2_LOSS_NOT_FINITE train=nan") and not ERR_RE.search("INFO - no ERRORS so far")
    assert END_RE.search("ACE2_POLARIS_TRAIN_OK nodes=1") and not END_RE.search("training continues")
    print("ACE2_WATCH_SELFTEST_OK")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dir", nargs="?")
    p.add_argument("o_file", nargs="?")
    p.add_argument("runtime_s", nargs="?", type=int, default=3300)
    p.add_argument("--deadline", default="2026-09-28T03:24:08", help="walltime end, UTC ISO")
    p.add_argument("--poll", type=int, default=300)
    p.add_argument("--stall-min", type=int, default=30)
    p.add_argument("--epochs", type=int, default=27)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.exp_dir or not args.o_file:
        p.error("exp_dir and o_file are required (or --selftest)")

    out_log = os.path.join(args.exp_dir, "out.log")
    ckdir = os.path.join(args.exp_dir, "training_checkpoints")
    deadline = dt.datetime.strptime(args.deadline, "%Y-%m-%dT%H:%M:%S")   # py3.6: no fromisoformat
    pos_out = os.path.getsize(out_log) if os.path.exists(out_log) else 0
    pos_o = os.path.getsize(args.o_file) if os.path.exists(args.o_file) else 0
    seen_ck = {f: os.path.getmtime(os.path.join(ckdir, f)) for f in os.listdir(ckdir)} if os.path.isdir(ckdir) else {}
    walls, done, stalled, warned2h = [], 0, False, False
    # seed the epoch history so projections use every epoch so far
    if os.path.exists(out_log):
        for r in parse_epoch_block(open(out_log, errors="replace").read().splitlines()):
            walls.append(r["wall_s"]); done = max(done, r["epoch"])
    now = dt.datetime.utcnow()
    print(f"watch start {now.isoformat(timespec='seconds')}Z epochs_done={done} deadline={deadline.isoformat()} "
          f"T-{(deadline-now).total_seconds()/3600:.1f}h", flush=True)
    t_end = time.time() + args.runtime_s
    while time.time() < t_end:
        time.sleep(args.poll)
        now = dt.datetime.utcnow(); stamp = now.strftime("%m-%d %H:%M")
        # out.log
        if os.path.exists(out_log):
            sz = os.path.getsize(out_log)
            if sz > pos_out:
                with open(out_log, "rb") as fh:
                    fh.seek(pos_out); chunk = fh.read(sz - pos_out).decode("utf8", "replace")
                pos_out = sz; stalled = False
                for r in parse_epoch_block(chunk.splitlines()):
                    walls.append(r["wall_s"]); done = max(done, r["epoch"])
                    mean = sum(walls) / len(walls)
                    left = (deadline - now).total_seconds() / mean
                    print(f"{stamp} EPOCH {r['epoch']} wall={r['wall_s']:.0f}s train={r.get('train','?')} "
                          f"lr={r.get('lr', float('nan')):.3e} best_val={r.get('best_val', float('nan')):.6f} "
                          f"| mean_epoch={mean/3600:.3f}h fits_before_deadline={left:.2f} remaining={args.epochs - r['epoch']}", flush=True)
                for l in chunk.splitlines():
                    if "Train loss: nan" in l or "Traceback" in l:
                        print(f"{stamp} ERR out.log: {l.strip()[:160]}", flush=True)
            else:
                age_min = (time.time() - os.path.getmtime(out_log)) / 60
                if age_min > args.stall_min and now < deadline and not stalled:
                    print(f"{stamp} STALL out.log unwritten for {age_min:.0f} min (job may be preempted/killed; qstat once)", flush=True)
                    stalled = True
        # checkpoints
        if os.path.isdir(ckdir):
            for f in sorted(os.listdir(ckdir)):
                pth = os.path.join(ckdir, f); m = os.path.getmtime(pth)
                if f not in seen_ck:
                    print(f"{stamp} CKPT new {f} size={os.path.getsize(pth)} mtime={time.strftime('%m-%d %H:%M', time.localtime(m))}", flush=True)
                elif f != "ckpt.tar" and m != seen_ck[f]:
                    print(f"{stamp} CKPT rewritten {f} mtime={time.strftime('%m-%d %H:%M', time.localtime(m))}", flush=True)
                seen_ck[f] = m
        # .o file
        if os.path.exists(args.o_file):
            sz = os.path.getsize(args.o_file)
            if sz > pos_o:
                with open(args.o_file, "rb") as fh:
                    fh.seek(pos_o); chunk = fh.read(sz - pos_o).decode("utf8", "replace")
                pos_o = sz
                for l in chunk.splitlines():
                    if END_RE.search(l):
                        print(f"{stamp} END {l.strip()[:200]}", flush=True)
                    elif ERR_RE.search(l):
                        print(f"{stamp} ERR .o: {l.strip()[:160]}", flush=True)
        # deadline
        left_h = (deadline - now).total_seconds() / 3600
        if left_h <= 2 and not warned2h:
            print(f"{stamp} DEADLINE T-{left_h:.1f}h: expect a mid-epoch kill; last ckpt.tar write is the resume point", flush=True)
            warned2h = True
        if left_h <= 0:
            print(f"{stamp} DEADLINE reached; check qstat -x once, then the resume recipe (brief §1)", flush=True)
            break
    print("watch exit", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
