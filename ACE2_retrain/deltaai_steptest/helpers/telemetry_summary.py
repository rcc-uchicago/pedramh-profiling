#!/usr/bin/env python3
"""Summarise GPU + host telemetry per fme epoch phase (train / val / inference).

    python telemetry_summary.py <experiment_dir>/out.log gpu.csv mem.tsv

gpu.csv  = nvidia-smi --query-gpu=timestamp,index,utilization.gpu,clocks.sm,clocks.max.sm,
           power.draw,enforced.power.limit,temperature.gpu,clocks_throttle_reasons.active,
           memory.used --format=csv -l 15
mem.tsv  = mem_sampler.sh output
Phase windows come from fme's own out.log markers:
  train k : "Starting training step ... <k-1> complete epochs" -> "Starting validation step ... <k> epochs"
  val   k : "Starting validation step ... <k> epochs"          -> "Validation complete"
  inf   k : "Starting inference step ... <k> epochs"           -> "Time taken for epoch <k>"
One TELEM line per (phase, epoch). Compare train e1 vs train e2: that is the question.
Pure stdlib; missing files are skipped.
"""
import csv
import datetime as dt
import os
import re
import statistics as st
import sys


def ts(s):
    s = s.strip().replace("/", "-").replace(",", ".")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s[:26], fmt)
        except ValueError:
            pass
    return None


def windows(log):
    marks = []
    pats = [
        ("train_start", re.compile(r"Starting training step for model trained for (\d+) complete epochs"), 1),
        ("val_start", re.compile(r"Starting validation step for model trained for (\d+) epochs"), 0),
        ("val_end", re.compile(r"Validation complete"), None),
        ("inf_start", re.compile(r"Starting inference step for model trained for (\d+) epochs"), 0),
        ("epoch_end", re.compile(r"Time taken for epoch (\d+) is"), 0),
    ]
    for line in open(log, errors="ignore"):
        t = ts(line[:23])
        if t is None:
            continue
        for name, p, off in pats:
            m = p.search(line)
            if m:
                marks.append((t, name, int(m.group(1)) + off if off is not None else None))
    out, cur_epoch, open_ = [], None, {}
    for t, name, ep in marks:
        if ep is not None:
            cur_epoch = ep
        if name in ("train_start", "val_start", "inf_start"):
            phase = name.split("_")[0]
            if phase == "val":  # training of this epoch ends where validation starts
                if ("train", cur_epoch) in open_:
                    out.append(("train", cur_epoch, open_.pop(("train", cur_epoch)), t))
            open_[(phase, cur_epoch)] = t
        elif name == "val_end" and ("val", cur_epoch) in open_:
            out.append(("val", cur_epoch, open_.pop(("val", cur_epoch)), t))
        elif name == "epoch_end" and ("inf", cur_epoch) in open_:
            out.append(("inf", cur_epoch, open_.pop(("inf", cur_epoch)), t))
    return out


def num(x):
    m = re.search(r"-?\d+(\.\d+)?", x or "")
    return float(m.group(0)) if m else None


def load_gpu(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path) as f:
        rd = csv.reader(f)
        hdr = [h.strip().split(" ")[0] for h in next(rd)]
        for r in rd:
            if len(r) != len(hdr):
                continue
            d = dict(zip(hdr, [c.strip() for c in r]))
            t = ts(d.get("timestamp", ""))
            if t:
                rows.append((t, d))
    return rows


def load_mem(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    hdr = None
    for line in open(path):
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if hdr is None:
            hdr = parts
            continue
        d = dict(zip(hdr, parts))
        t = ts(d.get("time", ""))
        if t:
            rows.append((t, d))
    return rows


def med(vals):
    vals = [v for v in vals if v is not None]
    return f"{st.median(vals):.1f}" if vals else "NA"


def main(log, gpu_path=None, mem_path=None):
    gpu, mem = load_gpu(gpu_path), load_mem(mem_path)
    for phase, ep, t0, t1 in windows(log):
        g = [d for t, d in gpu if t0 <= t <= t1]
        m = [d for t, d in mem if t0 <= t <= t1]
        thr = sorted({d.get("clocks_throttle_reasons.active", "") for d in g} - {""})

        def delta(key):
            v = [num(d.get(key)) for d in m]
            v = [x for x in v if x is not None]
            return f"{v[-1] - v[0]:.0f}" if len(v) >= 2 else "NA"

        print(
            f"TELEM {phase} e{ep} dur_s={(t1 - t0).total_seconds():.0f} n_gpu_samples={len(g)} "
            f"util%={med([num(d.get('utilization.gpu')) for d in g])} "
            f"sm_mhz={med([num(d.get('clocks.sm')) for d in g])}/{med([num(d.get('clocks.max.sm')) for d in g])} "
            f"power_w={med([num(d.get('power.draw')) for d in g])}/{med([num(d.get('enforced.power.limit')) for d in g])} "
            f"temp_c={med([num(d.get('temperature.gpu')) for d in g])} throttle={'|'.join(thr) or 'NA'} "
            f"| anon_g={med([num(d.get('cg_anon_gb')) for d in m])} file_g={med([num(d.get('cg_file_gb')) for d in m])} "
            f"cur/lim_g={med([num(d.get('cg_current_gb')) for d in m])}/{med([num(d.get('cg_limit_gb')) for d in m])} "
            f"refault_d={delta('cg_refault_file')} events_max_d={delta('cg_events_max')} "
            f"load1={med([num(d.get('loadavg_1m')) for d in m])} py_procs={med([num(d.get('n_python_procs')) for d in m])}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
