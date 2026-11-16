"""summarize_ncu_wu.py <raw.csv> [<raw.csv> ...] -- per-kernel memory behaviour from `ncu --import X --csv --page raw`.

For each kernel name (instances averaged): time (us), DRAM read / write (MB), global load / store sectors per request
and bytes used per sector (%), registers per thread, achieved occupancy (%). Standard library only. Prints one line
per kernel and WU_NCU_SUMMARY_OK; writes <csv>.summary.json beside each input.
A fully coalesced warp access of 4-byte words is 4 sectors per request (128 B); 8-byte (complex64) elements, 8.
"""
import csv
import json
import sys

M = {"time": "gpu__time_duration.sum", "dram_rd": "dram__bytes_read.sum", "dram_wr": "dram__bytes_write.sum",
     "ld_sec": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "ld_req": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
     "st_sec": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "st_req": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
     "ld_use": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
     "st_use": "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
     "regs": "launch__registers_per_thread", "occ": "sm__warps_active.avg.pct_of_peak_sustained_active"}
SCALE = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def summarize(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    units = rows[1] if len(rows) > 1 and not rows[1][0].strip().isdigit() else [""] * len(head)
    data = rows[2:] if units is rows[1] else rows[1:]
    col = {k: head.index(v) for k, v in M.items() if v in head}
    kcol = head.index("Kernel Name") if "Kernel Name" in head else None
    agg = {}
    for r in data:
        if kcol is None or len(r) <= kcol:
            continue
        name = r[kcol][:90]
        d = agg.setdefault(name, {"n": 0})
        d["n"] += 1
        for k, c in col.items():
            v = num(r[c])
            if v is None:
                continue
            u = units[c] if c < len(units) else ""
            v *= SCALE.get(u, 1.0)
            d[k] = d.get(k, 0.0) + v
    out = {}
    for name, d in agg.items():
        n = d.pop("n")
        e = {k: v / n for k, v in d.items()}
        if e.get("ld_req"):
            e["ld_sec_per_req"] = e.get("ld_sec", 0.0) / e["ld_req"]
        if e.get("st_req"):
            e["st_sec_per_req"] = e.get("st_sec", 0.0) / e["st_req"]
        e["instances"] = n
        out[name] = e
    with open(path + ".summary.json", "w") as f:
        json.dump(out, f, indent=1)
    for name, e in sorted(out.items(), key=lambda kv: -kv[1].get("time", 0.0)):
        print(f"{name[:60]:60s} n={e['instances']:3d} t={e.get('time', float('nan')):8.1f}us "
              f"rd={e.get('dram_rd', float('nan')):7.1f}MB wr={e.get('dram_wr', float('nan')):7.1f}MB "
              f"ld {e.get('ld_sec_per_req', float('nan')):5.2f} sec/req {e.get('ld_use', float('nan')):5.1f}% | "
              f"st {e.get('st_sec_per_req', float('nan')):5.2f} sec/req {e.get('st_use', float('nan')):5.1f}% | "
              f"regs {e.get('regs', float('nan')):4.0f} occ {e.get('occ', float('nan')):5.1f}%")
    return out


def main():
    for p in sys.argv[1:]:
        print(f"== {p}")
        summarize(p)
    print("WU_NCU_SUMMARY_OK")


if __name__ == "__main__":
    main()
