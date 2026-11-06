"""summarize_ncu.py -- per-kernel numbers from Nsight Compute `--page raw --csv` exports (profile_ab3_ncu_polaris.pbs).

  python summarize_ncu.py <ncu_<tag>_raw.csv> [more ...] --out <json>

Each input is `ncu --import <rep> --csv --page raw --print-units base`: a header row of column / metric names, usually
a units row, then one row per profiled kernel launch ("ID", "Kernel Name", launch columns, every collected metric).
Per kernel name, averaged over its instances: duration; DRAM read + write bytes, achieved GB/s and % of peak; memory
and SM throughput % of peak; theoretical vs achieved occupancy and which launch limit binds; registers / thread; block
and grid; local-memory (register spill) loads / stores; FP64 pipe utilisation; L1 / L2 hit rates; the top-5 warp-stall
reasons. Metric names move between ncu versions, so every field is a wishlist of column names tried in order
(WISHLIST): a field none of whose names is a column is null and listed under "missing"; nothing here raises on an
absent column. Quoted fields, thousands separators, "n/a" and non-base units (usecond, Mbyte, Gbyte/second ...) are
tolerated. Stdlib only: runs under the job's python or a login node's python3. Prints NCU_SUMMARY_OK.
"""
import argparse
import csv
import json
import os
import re
import sys

# field -> ncu metric (or launch column) names, most recent first. Comments: the ncu "details" page row they feed.
WISHLIST = [
    ("duration_ns", ["gpu__time_duration.sum"]),                                           # Duration
    ("dram_read_bytes", ["dram__bytes_read.sum"]),
    ("dram_write_bytes", ["dram__bytes_write.sum"]),
    ("dram_bytes", ["dram__bytes.sum"]),
    ("dram_bytes_per_second", ["dram__bytes.sum.per_second"]),
    ("dram_pct_of_peak", ["dram__throughput.avg.pct_of_peak_sustained_elapsed",           # DRAM Throughput
                          "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("memory_pct_of_peak", ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",   # Memory Throughput
                            "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("sm_pct_of_peak", ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]),             # Compute (SM) Throughput
    ("occupancy_theoretical_pct", ["sm__maximum_warps_per_active_cycle_pct"]),            # Theoretical Occupancy
    ("occupancy_achieved_pct", ["sm__warps_active.avg.pct_of_peak_sustained_active"]),    # Achieved Occupancy
    ("occupancy_limit_registers", ["launch__occupancy_limit_registers"]),                 # Block Limit Registers
    ("occupancy_limit_shared_mem", ["launch__occupancy_limit_shared_mem"]),
    ("occupancy_limit_warps", ["launch__occupancy_limit_warps"]),
    ("occupancy_limit_blocks", ["launch__occupancy_limit_blocks"]),
    ("registers_per_thread", ["launch__registers_per_thread"]),
    ("block_size", ["launch__block_size"]),                                               # also "Block Size" tuple
    ("grid_size", ["launch__grid_size"]),                                                 # also "Grid Size" tuple
    ("waves_per_sm", ["launch__waves_per_multiprocessor"]),
    ("shared_mem_static_bytes", ["launch__shared_mem_per_block_static"]),
    ("shared_mem_dynamic_bytes", ["launch__shared_mem_per_block_dynamic"]),
    ("local_load_inst", ["smsp__sass_inst_executed_op_local_ld.sum", "smsp__inst_executed_op_local_ld.sum"]),
    ("local_store_inst", ["smsp__sass_inst_executed_op_local_st.sum", "smsp__inst_executed_op_local_st.sum"]),
    ("local_load_sectors", ["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"]),
    ("local_store_sectors", ["l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"]),
    ("fp64_pipe_pct", ["sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",    # Pipe Utilization: FP64
                       "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
                       "smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active"]),
    ("fp64_inst_dfma", ["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"]),
    ("fp64_inst_dmul", ["smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"]),
    ("fp64_inst_dadd", ["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"]),
    ("l1_hit_pct", ["l1tex__t_sector_hit_rate.pct"]),                                     # L1/TEX Hit Rate
    ("l2_hit_pct", ["lts__t_sector_hit_rate.pct"]),                                       # L2 Hit Rate
    ("l1_bytes", ["l1tex__t_bytes.sum"]),
    ("l2_bytes", ["lts__t_bytes.sum"]),
    ("issue_active_pct", ["smsp__issue_active.avg.pct_of_peak_sustained_active"]),        # Issue Slot Utilization
    ("eligible_warps_per_cycle", ["smsp__warps_eligible.avg.per_cycle_active"]),
    ("active_warps_per_scheduler", ["smsp__warps_active.avg.per_cycle_active"]),
    ("inst_executed", ["smsp__inst_executed.sum"]),
    ("threads_per_inst", ["smsp__thread_inst_executed_per_inst_executed.ratio"]),         # divergence: 32 = none
    ("global_ld_sector_efficiency_pct", ["smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"]),
    ("global_st_sector_efficiency_pct", ["smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct"]),
    ("sm_frequency_hz", ["sm__cycles_elapsed.avg.per_second", "gpc__cycles_elapsed.avg.per_second"]),
    ("dram_frequency_hz", ["dram__cycles_elapsed.avg.per_second"]),
    ("elapsed_cycles", ["gpc__cycles_elapsed.max", "sm__cycles_elapsed.avg"]),
]
# warp-stall metric families, most recent first; the first one with any column present is used
STALL_FAMILIES = ["warp_latency_issue_stalled", "warps_issue_stalled", "warp_issue_stalled"]
STALL_REASON = re.compile(r"issue_stalled_(.+?)(?:_per_issue_active|_per_warp_active)?\.(ratio|pct)$")
UNIT = re.compile(r"^([numKMGT]?)(second|byte|byte/second|byte/s)$")
PREFIX = {"": 1.0, "n": 1e-9, "u": 1e-6, "m": 1e-3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
LIMITS = ["registers", "shared_mem", "warps", "blocks"]


def number(s):
    """'3,272,324.90' -> 3272324.9; '', 'n/a', text -> None."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if not s or s.lower() in ("n/a", "nan", "inf", "-inf"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def unit_scale(u):
    """Factor that takes a value in unit `u` to ncu's base units: nanoseconds for time (nsecond -> 1, usecond -> 1e3),
    bytes for data (Mbyte -> 1e6, Gbyte/second -> 1e9); 1 for anything else (%, ratios, inst, ...)."""
    m = UNIT.match((u or "").strip())
    if not m:
        return 1.0
    prefix, base = m.groups()
    return PREFIX[prefix] * (1e9 if base == "second" else 1.0)


def tuple_product(s):
    """'(256, 1, 1)' -> 256.0 (ncu's launch columns); None if it is not such a tuple."""
    nums = re.findall(r"\d+", re.sub(r",(?=\d)", "", s or ""))     # '4,096' is one number; ', ' separates the axes
    if not nums:
        return None
    p = 1
    for n in nums:
        p *= int(n)
    return float(p)


def read_raw_csv(path):
    """(header, units, rows): rows are {column: text}; units is {column: unit} (empty when there is no units row).

    Leading non-CSV lines (ncu's ==WARNING== / ==PROF== chatter) are skipped: the header is the first row that holds
    a "Kernel Name" column. The units row is recognised by a non-numeric "ID".
    """
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = None
        for row in reader:
            if "Kernel Name" in row:
                header = row
                break
        if header is None:
            raise ValueError("no header row with a 'Kernel Name' column")
        id_col = header.index("ID") if "ID" in header else 0
        units, rows = {}, []
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            row = row + [""] * (len(header) - len(row))
            if not rows and not units and number(row[id_col]) is None:
                units = dict(zip(header, row))
                continue
            rows.append(dict(zip(header, row)))
    return header, units, rows


def pick(rows, units, names):
    """(mean over rows, column used) for the first name that is a column with at least one numeric value."""
    for name in names:
        if name not in rows[0]:
            continue
        vals = [number(r.get(name)) for r in rows]
        vals = [v * unit_scale(units.get(name)) for v in vals if v is not None]
        if vals:
            return sum(vals) / len(vals), name
    return None, None


def stalls(rows, header):
    """(family, [(reason, mean value) ...] sorted descending) from the first STALL_FAMILIES family present."""
    for fam in STALL_FAMILIES:
        cols = [c for c in header if fam in c and STALL_REASON.search(c)]
        if not cols:
            continue
        out, seen = [], set()
        for c in sorted(cols, key=lambda c: (not c.endswith(".ratio"), c)):   # .ratio first; .pct duplicates skipped
            reason = STALL_REASON.search(c).group(1)
            if reason in seen:
                continue
            vals = [number(r.get(c)) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                seen.add(reason)
                out.append((reason, sum(vals) / len(vals)))
        return fam, sorted(out, key=lambda kv: -kv[1])
    return None, []


def summarize_kernel(rows, units, header, tag, source):
    k = {"tag": tag, "source": os.path.basename(source), "instances": len(rows), "metrics": {}, "metric_columns": {},
         "missing": []}
    for field, names in WISHLIST:
        v, col = pick(rows, units, names)
        k["metrics"][field] = v
        if col is None:
            k["missing"].append(field)
        else:
            k["metric_columns"][field] = col
    m = k["metrics"]
    # launch geometry: the metric if collected, else ncu's "(x, y, z)" launch columns
    for field, col in (("block_size", "Block Size"), ("grid_size", "Grid Size")):
        if m[field] is None and col in rows[0]:
            m[field] = tuple_product(rows[0][col])
            if m[field] is not None:
                k["missing"].remove(field)
                k["metric_columns"][field] = col
        if col in rows[0]:
            k[col.lower().replace(" ", "_") + "_xyz"] = rows[0][col].strip()

    d = {}
    durs = [number(r.get(k["metric_columns"].get("duration_ns", ""))) for r in rows]
    durs = [v * unit_scale(units.get(k["metric_columns"].get("duration_ns"))) for v in durs if v is not None]
    d["duration_us"] = round(m["duration_ns"] / 1e3, 2) if m["duration_ns"] is not None else None
    d["duration_us_min"] = round(min(durs) / 1e3, 2) if durs else None
    d["duration_us_max"] = round(max(durs) / 1e3, 2) if durs else None
    rd, wr = m["dram_read_bytes"], m["dram_write_bytes"]
    total = rd + wr if rd is not None and wr is not None else m["dram_bytes"]
    d["dram_bytes_total"] = total
    d["dram_read_fraction"] = round(rd / total, 3) if total and rd is not None else None
    if total is not None and m["duration_ns"]:
        d["achieved_GBps"] = round(total / (m["duration_ns"] * 1e-9) / 1e9, 1)
    elif m["dram_bytes_per_second"] is not None:
        d["achieved_GBps"] = round(m["dram_bytes_per_second"] / 1e9, 1)
    else:
        d["achieved_GBps"] = None
    pct = m["dram_pct_of_peak"]
    d["peak_GBps_implied"] = round(d["achieved_GBps"] / (pct / 100.0), 1) if d["achieved_GBps"] and pct else None
    lim = [(m["occupancy_limit_" + x], x) for x in LIMITS if m["occupancy_limit_" + x] is not None]
    d["occupancy_limiter"] = min(lim)[1] if lim else None
    d["occupancy_limit_blocks_per_sm"] = min(lim)[0] if lim else None
    if m["block_size"] and m["grid_size"]:
        d["threads_total"] = m["block_size"] * m["grid_size"]
    ld, st = m["local_load_inst"], m["local_store_inst"]
    d["local_memory_used"] = None if ld is None and st is None else bool((ld or 0) + (st or 0) > 0)
    fam, top = stalls(rows, header)
    k["stall_family"] = fam
    k["stalls_top5"] = [{"reason": r, "value": round(v, 3)} for r, v in top[:5]]
    k["derived"] = d
    return k


def summarize(paths):
    kernels, inputs, bad = {}, [], 0
    for path in paths:
        tag = re.search(r"ncu_(.+?)_raw\.csv$", os.path.basename(path))
        tag = tag.group(1) if tag else os.path.splitext(os.path.basename(path))[0]
        try:
            header, units, rows = read_raw_csv(path)
            if not rows:
                raise ValueError("no kernel rows")
        except (OSError, ValueError, UnicodeDecodeError) as e:
            print(f"ERROR NCU_BAD_CSV {path}: {e}")
            bad += 1
            continue
        inputs.append({"path": path, "tag": tag, "rows": len(rows), "columns": len(header), "units_row": bool(units)})
        by_name = {}
        for r in rows:
            by_name.setdefault(r.get("Kernel Name", "?").strip(), []).append(r)
        for name, krows in by_name.items():
            key = name if name not in kernels else f"{name} [{tag}]"
            kernels[key] = summarize_kernel(krows, units, header, tag, path)
    return {"inputs": inputs, "bad_inputs": bad, "kernels": kernels,
            "missing_any": sorted({f for k in kernels.values() for f in k["missing"]})}


def fmt(v, nd=1, width=6):
    return f"{v:{width}.{nd}f}" if isinstance(v, (int, float)) else f"{'-':>{width}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="ncu --page raw --csv exports")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = summarize(a.csv)
    if not r["kernels"]:
        print("ERROR NCU_NO_KERNELS: no usable kernel rows in " + " ".join(a.csv))
        sys.exit(1)
    with open(a.out, "w") as fh:
        json.dump(r, fh, indent=1)
    print(f"{'kernel':34s} {'tag':11s} n {'us':>8s} {'GB/s':>6s} {'DRAM%':>6s} {'mem%':>5s} {'SM%':>5s} "
          f"{'occ th/ach':>11s} regs {'spill':>5s} {'fp64%':>5s} {'L2%':>5s}  top stall")
    rows = sorted(r["kernels"].items(), key=lambda kv: -(kv[1]["metrics"]["duration_ns"] or 0))
    for name, k in rows[:8]:
        m, d = k["metrics"], k["derived"]
        top = k["stalls_top5"][0]["reason"] if k["stalls_top5"] else "-"
        print(f"{name[:34]:34s} {k['tag'][:11]:11s} {k['instances']} {fmt(d['duration_us'], 1, 8)} "
              f"{fmt(d['achieved_GBps'], 0)} {fmt(m['dram_pct_of_peak'])} {fmt(m['memory_pct_of_peak'], 0, 5)} "
              f"{fmt(m['sm_pct_of_peak'], 0, 5)} {fmt(m['occupancy_theoretical_pct'], 0, 7)}/{fmt(m['occupancy_achieved_pct'], 0, 3)} "
              f"{fmt(m['registers_per_thread'], 0, 4)} {fmt(m['local_load_inst'], 0, 5)} {fmt(m['fp64_pipe_pct'], 0, 5)} "
              f"{fmt(m['l2_hit_pct'], 0, 5)}  {top}")
    if r["missing_any"]:
        print(f"missing metrics (null in the JSON): {len(r['missing_any'])}: {', '.join(r['missing_any'][:8])}"
              + (" ..." if len(r["missing_any"]) > 8 else ""))
    print(f"NCU_SUMMARY_OK {a.out}")


if __name__ == "__main__":
    main()
