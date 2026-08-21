#!/usr/bin/env python3
"""Turn `ncu --csv` long-format output into the item-7 table.

    python3 ncu_summarize.py <ncu.csv>

ncu emits one ROW PER METRIC PER LAUNCH, so it has to be pivoted before it says
anything. The two numbers plan item 7 turns on:

* **`dram__throughput.avg.pct_of_peak_sustained_elapsed`** — real DRAM traffic as
  a fraction of peak. §0d's 17-27% is an estimate of *useful* bytes from launch
  geometry; this is what the memory system actually did.
* **sectors/request** — coalescing. An L1TEX sector is 32 B, so a fully-coalesced
  warp request is 128 B / 32 = **4 sectors for fp32** and **8 for complex64**
  (32 lanes x 8 B = 256 B). Materially above that is wasted traffic.

Together they separate the two readings §0d could not:
    ~25% of peak AND sectors/request at the ideal -> the copies are efficient,
        so the lever is FEWER bytes (do not materialise them).
    near peak AND sectors/request well above ideal -> the traffic is inflated by
        the access pattern, so the lever is CONTIGUITY (fix the layout).

No `statistics.fmean`: it is 3.8+ and the Polaris login node runs 3.6. That trap
has already been fixed twice in this repo.
"""
import collections
import csv
import re
import sys

IDEAL_SECTORS = {"float": 4, "complex64": 8, "bf16": 2, "?": 4}


def _dtype(name):
    if "complex<float>" in name or "complex64" in name:
        return "complex64"
    if "BFloat16" in name:
        return "bf16"
    if re.search(r"\bfloat\b", name):
        return "float"
    return "?"


def _short(name):
    for pat in ("direct_copy_kernel_cuda", "conj_kernel_cuda"):
        if pat in name:
            base = pat.replace("_kernel_cuda", "")
            tag = ("nocast" if "gpu_kernel_impl_nocast" in name else
                   "vec" if "vectorized_elementwise" in name else
                   "unrolled" if "unrolled_elementwise" in name else "")
            return f"{base}.{tag}<{_dtype(name)}>" if tag else f"{base}<{_dtype(name)}>"
    return name.split("(")[0][:44]


def load(path):
    """[(kernel_label, {metric: float})] — one entry per profiled launch."""
    with open(path, newline="") as fh:
        # ncu prefixes the CSV with ==PROF== banner lines; find the real header.
        lines = [ln for ln in fh if not ln.startswith("==")]
    rdr = csv.DictReader(lines)
    kcol = mcol = vcol = idcol = None
    for c in (rdr.fieldnames or []):
        lc = c.lower()
        if kcol is None and "kernel name" in lc:
            kcol = c
        elif mcol is None and "metric name" in lc:
            mcol = c
        elif vcol is None and "metric value" in lc:
            vcol = c
        elif idcol is None and lc.strip() == "id":
            idcol = c
    if not (kcol and mcol and vcol):
        sys.exit(f"ERROR NCU_CSV_UNRECOGNISED: need Kernel Name / Metric Name / "
                 f"Metric Value columns, got {rdr.fieldnames}")
    launches = collections.OrderedDict()
    for row in rdr:
        key = (row.get(idcol) or "0", row[kcol])
        try:
            val = float(str(row[vcol]).replace(",", ""))
        except (TypeError, ValueError):
            continue
        launches.setdefault(key, {})[row[mcol]] = val
    return [(_short(k[1]), m) for k, m in launches.items()]


def report(launches):
    if not launches:
        sys.exit("ERROR NCU_NO_LAUNCHES parsed")
    by = collections.defaultdict(list)
    for label, m in launches:
        by[label].append(m)

    def g(m, *names):
        for n in names:
            for k, v in m.items():
                if k.endswith(n) or k == n:
                    return v
        return None

    mean = lambda v: sum(v) / len(v) if v else float("nan")
    print(f"{'kernel':<34}{'n':>3}{'DRAM MB':>10}{'%peak':>8}"
          f"{'SM%':>7}{'sec/req ld':>12}{'sec/req st':>12}{'ideal':>7}")
    print("  " + "-" * 92)
    for label, ms in sorted(by.items(), key=lambda kv: -len(kv[1])):
        rd = [g(m, "dram__bytes_read.sum") or 0 for m in ms]
        wr = [g(m, "dram__bytes_write.sum") or 0 for m in ms]
        pk = [g(m, "dram__throughput.avg.pct_of_peak_sustained_elapsed") for m in ms]
        sm = [g(m, "sm__throughput.avg.pct_of_peak_sustained_elapsed") for m in ms]
        sl = [g(m, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum") for m in ms]
        rl = [g(m, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum") for m in ms]
        ss = [g(m, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum") for m in ms]
        rs = [g(m, "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum") for m in ms]
        ratio = lambda a, b: (mean([x for x in a if x is not None])
                              / mean([x for x in b if x is not None])
                              if any(x for x in b if x) else float("nan"))
        dt = label.split("<")[-1].rstrip(">").split(",")[-1]
        ideal = IDEAL_SECTORS.get(dt, 4)
        print(f"{label:<34}{len(ms):>3}"
              f"{(mean(rd) + mean(wr)) / 1e6:>10.1f}"
              f"{mean([x for x in pk if x is not None]):>7.1f}%"
              f"{mean([x for x in sm if x is not None]):>6.1f}%"
              f"{ratio(sl, rl):>12.2f}{ratio(ss, rs):>12.2f}{ideal:>7}")
    print("\n  sec/req materially above `ideal` = uncoalesced, i.e. the traffic is")
    print("  inflated by the access pattern -> the lever is CONTIGUITY.")
    print("  sec/req at ideal with a LOW %peak = efficient but unnecessary copies")
    print("  -> the lever is FEWER bytes. See plan item 7's decision rule.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: ncu_summarize.py <ncu.csv>")
    report(load(sys.argv[1]))
