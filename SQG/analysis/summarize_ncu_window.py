"""summarize_ncu_window.py -- every kernel of a solver loop iteration, from one Nsight Compute window export.

  python summarize_ncu_window.py <ncu_<tag>_raw.csv> --out <json> [--steps-per-period 2]

The input is `ncu --import <rep> --csv --page raw --print-units base` of a run that profiled ALL kernels of a
contiguous window (no --kernel-name filter): the call's entry kernels, then the loop body repeating. Rows are taken in
launch order (ncu's "ID"). Each launch gets summarize_ncu.py's per-kernel numbers (duration, DRAM read / write bytes
and GB/s, % of peak, L1 / L2 hit, global load / store sector efficiency, registers, occupancy, top stall).

The loop period is found from the launch signatures (kernel name, grid, block): the smallest P >= 2 such that the
last 2P launches are one sequence repeated. The last full period is reported in order, with its total duration and
DRAM bytes, and per solver step (/ --steps-per-period: 2 for a scan unrolled by 2). If no period is found, the window
is reported grouped by signature only. Durations are ncu's: kernels serialised, caches flushed before each one (ncu's
default), production clocks if the job used --clock-control none. Stdlib only. Prints NCU_WINDOW_OK.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_ncu import number, read_raw_csv, summarize_kernel   # noqa: E402

FIELDS = ("duration_us", "dram_read_MB", "dram_write_MB", "achieved_GBps", "dram_pct_of_peak", "l1_hit_pct",
          "l2_hit_pct", "ld_sector_eff_pct", "st_sector_eff_pct", "registers", "occupancy_achieved_pct", "top_stall")


def launch_record(row, units, header, path):
    k = summarize_kernel([row], units, header, "window", path)
    m, d = k["metrics"], k["derived"]
    rd, wr = m["dram_read_bytes"], m["dram_write_bytes"]
    return {"id": int(number(row.get("ID")) or 0), "name": row.get("Kernel Name", "?").strip(),
            "grid": k.get("grid_size_xyz", ""), "block": k.get("block_size_xyz", ""),
            "duration_us": d["duration_us"], "dram_read_MB": None if rd is None else round(rd / 1e6, 2),
            "dram_write_MB": None if wr is None else round(wr / 1e6, 2), "achieved_GBps": d["achieved_GBps"],
            "dram_pct_of_peak": None if m["dram_pct_of_peak"] is None else round(m["dram_pct_of_peak"], 1),
            "l1_hit_pct": None if m["l1_hit_pct"] is None else round(m["l1_hit_pct"], 1),
            "l2_hit_pct": None if m["l2_hit_pct"] is None else round(m["l2_hit_pct"], 1),
            "ld_sector_eff_pct": None if m["global_ld_sector_efficiency_pct"] is None
            else round(m["global_ld_sector_efficiency_pct"], 1),
            "st_sector_eff_pct": None if m["global_st_sector_efficiency_pct"] is None
            else round(m["global_st_sector_efficiency_pct"], 1),
            "registers": m["registers_per_thread"], "threads": d.get("threads_total"),
            "occupancy_achieved_pct": None if m["occupancy_achieved_pct"] is None
            else round(m["occupancy_achieved_pct"], 1),
            "top_stall": k["stalls_top5"][0]["reason"] if k["stalls_top5"] else None,
            "missing": k["missing"]}


def find_period(sigs, min_period=2):
    """Smallest P >= min_period with sigs[-2P:-P] == sigs[-P:], else None."""
    n = len(sigs)
    for p in range(min_period, n // 2 + 1):
        if sigs[n - 2 * p:n - p] == sigs[n - p:]:
            return p
    return None


def summarize(path, steps_per_period):
    header, units, rows = read_raw_csv(path)
    if not rows:
        raise ValueError("no kernel rows")
    launches = sorted((launch_record(r, units, header, path) for r in rows), key=lambda x: x["id"])
    sigs = [(x["name"], x["grid"], x["block"]) for x in launches]
    p = find_period(sigs)
    out = {"input": path, "launches": len(launches), "period": p, "steps_per_period": steps_per_period}
    groups, order = {}, []
    for x in launches:
        key = f"{x['name']} {x['grid']}x{x['block']}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(x)

    def mean(xs, f):
        v = [x[f] for x in xs if isinstance(x[f], (int, float))]
        return round(sum(v) / len(v), 2) if v else None

    out["by_signature"] = [{"signature": k, "n": len(groups[k]), **{f: mean(groups[k], f) for f in FIELDS
                                                                    if f != "top_stall"},
                            "top_stall": groups[k][-1]["top_stall"]} for k in order]
    if p:
        per = launches[-p:]
        tot_us = sum(x["duration_us"] or 0 for x in per)
        tot_rd = sum(x["dram_read_MB"] or 0 for x in per)
        tot_wr = sum(x["dram_write_MB"] or 0 for x in per)
        out["last_period"] = per
        out["period_totals"] = {"duration_us": round(tot_us, 1), "dram_read_MB": round(tot_rd, 1),
                                "dram_write_MB": round(tot_wr, 1),
                                "per_step_us": round(tot_us / steps_per_period, 1),
                                "per_step_dram_MB": round((tot_rd + tot_wr) / steps_per_period, 1),
                                "GBps_over_period": round((tot_rd + tot_wr) * 1e6 / (tot_us * 1e-6) / 1e9, 1)
                                if tot_us else None}
    out["missing_any"] = sorted({f for x in launches for f in x["missing"]})
    for x in launches:
        del x["missing"]
    return out


def fmt(v, nd=1, w=7):
    return f"{v:{w}.{nd}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps-per-period", type=int, default=2)
    a = ap.parse_args()
    try:
        r = summarize(a.csv, a.steps_per_period)
    except (OSError, ValueError) as e:
        print(f"ERROR NCU_WINDOW_BAD_CSV {a.csv}: {e}")
        sys.exit(1)
    with open(a.out, "w") as fh:
        json.dump(r, fh, indent=1)
    rows = r.get("last_period") or []
    print(f"{os.path.basename(a.csv)}: {r['launches']} launches, period {r['period']}")
    print(f"  {'kernel':36s} {'grid x block':18s} {'us':>7s} {'rd MB':>7s} {'wr MB':>7s} {'GB/s':>7s} {'DRAM%':>6s} "
          f"{'L2%':>5s} {'ldEff':>5s} {'stEff':>5s} regs  stall")
    for x in rows:
        print(f"  {x['name'][:36]:36s} {(x['grid'] + 'x' + x['block'])[:18]:18s} {fmt(x['duration_us'])} "
              f"{fmt(x['dram_read_MB'])} {fmt(x['dram_write_MB'])} {fmt(x['achieved_GBps'], 0)} "
              f"{fmt(x['dram_pct_of_peak'], 1, 6)} {fmt(x['l2_hit_pct'], 0, 5)} {fmt(x['ld_sector_eff_pct'], 0, 5)} "
              f"{fmt(x['st_sector_eff_pct'], 0, 5)} {fmt(x['registers'], 0, 4)}  {x['top_stall'] or '-'}")
    if r.get("period_totals"):
        t = r["period_totals"]
        print(f"  period: {t['duration_us']} us, DRAM {t['dram_read_MB']} + {t['dram_write_MB']} MB; per step "
              f"{t['per_step_us']} us, {t['per_step_dram_MB']} MB, {t['GBps_over_period']} GB/s")
    else:
        print("  WARN NCU_WINDOW_NO_PERIOD: grouped by signature only")
    print(f"NCU_WINDOW_OK {a.out}")


if __name__ == "__main__":
    main()
