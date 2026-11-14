"""Synthetic-input test of summarize_ncu.py (stdlib only, no ncu / GPU; runs on a login node).

  python3 SQG/analysis/test_summarize_ncu.py          # prints NCU_TEST_OK
Builds two `ncu --page raw --csv`-shaped files and checks: the units row is recognised and non-base units (usecond,
Mbyte) are scaled; quoted fields with commas (a cuFFT template name) and thousands separators parse; two kernels x two
instances are averaged per kernel name; a metric absent from the header is null and listed as missing, not an error;
block / grid fall back to ncu's "(x, y, z)" launch columns; achieved GB/s, implied peak, the occupancy limiter and
the top-5 stall reasons are derived; ncu's ==WARNING== chatter before the header is skipped; the tag comes from the
file name; an unreadable input gives a greppable ERROR line.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AB3 = "loop_add_multiply_fusion_1"
MUL = "wrapped_multiply_1"
FFT = ("void regular_fft_factor<(unsigned int)64, EPT<(unsigned int)8>, (unsigned int)32, (unsigned int)4, "
       "(padding_t)1, (twiddle_t)0, (loadstore_modifier_t)2, (layout_t)1, unsigned int, double>(kernel_arguments_t<T9>)")
LAUNCH = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size",
          "Device", "CC"]
STALL = "smsp__average_warp_latency_issue_stalled_{}.ratio"
REASONS = ["long_scoreboard", "wait", "lg_throttle", "math_pipe_throttle", "not_selected", "barrier"]


def write(path, header, units, rows, preamble=()):
    with open(path, "w", newline="") as fh:
        for line in preamble:
            fh.write(line + "\n")
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)                  # ncu quotes every field
        w.writerow(header)
        if units is not None:
            w.writerow(units)
        for r in rows:
            w.writerow(r)


def build_main(path):
    """Full-set export: two kernels x two instances, base units, one metric (L2 hit rate) absent."""
    metrics = ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
               "dram__throughput.avg.pct_of_peak_sustained_elapsed",
               "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
               "sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm__maximum_warps_per_active_cycle_pct",
               "sm__warps_active.avg.pct_of_peak_sustained_active", "launch__occupancy_limit_registers",
               "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps", "launch__occupancy_limit_blocks",
               "launch__registers_per_thread", "launch__block_size", "launch__grid_size",
               "smsp__sass_inst_executed_op_local_ld.sum", "smsp__sass_inst_executed_op_local_st.sum",
               "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active", "l1tex__t_sector_hit_rate.pct"]
    metrics += [STALL.format(r) for r in REASONS]
    metrics += [STALL.format("long_scoreboard").replace(".ratio", ".pct")]     # a .pct duplicate: must not double-count
    header = LAUNCH + metrics
    units = [""] * len(LAUNCH) + ["nsecond", "byte", "byte", "%", "%", "%", "%", "%", "block", "block", "block", "block",
                                  "register/thread", "block", "block", "inst", "inst", "%", "%"] + [""] * 6 + ["%"]
    assert len(units) == len(header)

    def row(i, name, dur, rd, wr, pct, occ_ach, regs, ld, st, fp64, stall):
        launch = [str(i), "12345", "python", "x3005c0s7b0n0", name, "1", "7", "(256, 1, 1)", "(32768, 1, 1)", "0", "8.0"]
        vals = [dur, rd, wr, pct, "26.1", "41.0", "100", occ_ach, "4", "32", "8", "32", regs, "256", "32,768",
                ld, st, fp64, "33.3"] + stall + ["n/a"]
        return launch + vals
    rows = [
        row(0, AB3, "3,272,000", "1,000,000,000", "270,000,000", "25.0", "61.0", "64", "0", "0", "12.5",
            ["30.0", "5.0", "12.0", "8.0", "6.0", "1.0"]),
        row(1, AB3, "3,268,000", "1,000,000,000", "270,000,000", "25.0", "60.0", "64", "0", "0", "12.5",
            ["32.0", "5.0", "12.0", "8.0", "6.0", "1.0"]),
        row(2, MUL, "308,000", "268,435,456", "134,217,728", "82.0", "95.0", "16", "100", "50", "2.0",
            ["40.0", "1.0", "2.0", "0.5", "3.0", "0.0"]),
        row(3, MUL, "308,000", "268,435,456", "134,217,728", "82.0", "95.0", "16", "100", "50", "2.0",
            ["40.0", "1.0", "2.0", "0.5", "3.0", "0.0"]),
    ]
    write(path, header, units, rows)


def build_fft(path):
    """A sparser export: ncu chatter first, scaled units, no launch__ metrics, no stall family, a template kernel name."""
    header = LAUNCH + ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
                       "dram__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_sector_hit_rate.pct"]
    units = [""] * len(LAUNCH) + ["usecond", "Mbyte", "Mbyte", "%", "%"]
    rows = [[str(0), "12345", "python", "x3005c0s7b0n0", FFT, "1", "7", "(256, 1, 1)", "(4,096, 1, 1)", "0", "8.0",
             "201.1", "134.22", "134.22", "85.5", "47.0"]]
    write(path, header, units, rows, preamble=["==WARNING== Some chatter before the table", ""])


def run(args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "summarize_ncu.py")] + args,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def main():
    with tempfile.TemporaryDirectory() as d:
        main_csv, fft_csv, out = (os.path.join(d, n) for n in ("ncu_ab3_raw.csv", "ncu_fft_raw.csv", "s.json"))
        build_main(main_csv)
        build_fft(fft_csv)
        rc, txt = run([main_csv, fft_csv, "--out", out])
        assert rc == 0 and "NCU_SUMMARY_OK" in txt and "ERROR" not in txt, txt
        assert len([l for l in txt.splitlines() if l.strip()]) <= 10, txt
        with open(out) as fh:
            s = json.load(fh)
        K = s["kernels"]
        assert set(K) == {AB3, MUL, FFT}, sorted(K)
        assert [i["tag"] for i in s["inputs"]] == ["ab3", "fft"] and s["bad_inputs"] == 0, s["inputs"]

        a, m, d_ = K[AB3], K[AB3]["metrics"], K[AB3]["derived"]
        assert a["tag"] == "ab3" and a["instances"] == 2, a
        assert m["duration_ns"] == 3270000.0 and d_["duration_us"] == 3270.0, (m["duration_ns"], d_)
        assert d_["duration_us_min"] == 3268.0 and d_["duration_us_max"] == 3272.0, d_
        assert d_["dram_bytes_total"] == 1.27e9 and d_["dram_read_fraction"] == 0.787, d_
        assert d_["achieved_GBps"] == 388.4 and m["dram_pct_of_peak"] == 25.0, d_        # 1.27e9 B / 3.270 ms
        assert d_["peak_GBps_implied"] == 1553.6, d_
        assert m["memory_pct_of_peak"] == 26.1 and m["sm_pct_of_peak"] == 41.0, m
        assert m["occupancy_theoretical_pct"] == 100.0 and m["occupancy_achieved_pct"] == 60.5, m
        assert d_["occupancy_limiter"] == "registers" and d_["occupancy_limit_blocks_per_sm"] == 4.0, d_
        assert m["registers_per_thread"] == 64.0 and m["block_size"] == 256.0 and m["grid_size"] == 32768.0, m
        assert d_["threads_total"] == 256.0 * 32768 and a["block_size_xyz"] == "(256, 1, 1)", (d_, a)
        assert m["local_load_inst"] == 0.0 and d_["local_memory_used"] is False, (m, d_)
        assert m["fp64_pipe_pct"] == 12.5 and m["l1_hit_pct"] == 33.3, m
        assert m["l2_hit_pct"] is None and "l2_hit_pct" in a["missing"] and "l2_hit_pct" in s["missing_any"], a["missing"]
        assert "duration_ns" not in a["missing"] and a["metric_columns"]["duration_ns"] == "gpu__time_duration.sum"
        assert a["stall_family"] == "warp_latency_issue_stalled", a["stall_family"]
        assert [x["reason"] for x in a["stalls_top5"]] == ["long_scoreboard", "lg_throttle", "math_pipe_throttle",
                                                            "not_selected", "wait"], a["stalls_top5"]
        assert a["stalls_top5"][0]["value"] == 31.0, a["stalls_top5"]                  # mean of 30 and 32, not the .pct

        w, wd = K[MUL]["metrics"], K[MUL]["derived"]
        assert wd["duration_us"] == 308.0 and wd["dram_bytes_total"] == 402653184.0, wd
        assert wd["achieved_GBps"] == 1307.3 and wd["local_memory_used"] is True, wd  # 402653184 B / 308 us
        assert w["local_load_inst"] == 100.0 and w["local_store_inst"] == 50.0, w

        f, fm, fd = K[FFT], K[FFT]["metrics"], K[FFT]["derived"]
        assert f["tag"] == "fft" and f["instances"] == 1 and f["source"] == "ncu_fft_raw.csv", f
        assert fm["duration_ns"] == 201100.0 and fd["duration_us"] == 201.1, (fm["duration_ns"], fd)   # usecond scaled
        assert abs(fd["dram_bytes_total"] - 268.44e6) < 1 and fd["achieved_GBps"] == 1334.9, fd       # Mbyte scaled
        assert fm["block_size"] == 256.0 and fm["grid_size"] == 4096.0, fm            # "(x, y, z)" fallback, "4,096"
        assert f["metric_columns"]["grid_size"] == "Grid Size" and "grid_size" not in f["missing"], f
        assert fm["l2_hit_pct"] == 47.0 and fm["registers_per_thread"] is None, fm
        assert "registers_per_thread" in f["missing"] and fd["occupancy_limiter"] is None, f
        assert f["stall_family"] is None and f["stalls_top5"] == [], f

        # unusable inputs: a missing file alongside a good one is reported but does not stop the summary ...
        rc, txt = run([os.path.join(d, "nope.csv"), main_csv, "--out", out])
        assert rc == 0 and "ERROR NCU_BAD_CSV" in txt and "NCU_SUMMARY_OK" in txt, txt
        # ... a file without a header is unusable on its own
        junk = os.path.join(d, "junk.csv")
        with open(junk, "w") as fh:
            fh.write("==WARNING== nothing here\n")
        rc, txt = run([junk, "--out", out])
        assert rc != 0 and "ERROR NCU_BAD_CSV" in txt and "ERROR NCU_NO_KERNELS" in txt, txt
    print("NCU_TEST_OK")


if __name__ == "__main__":
    main()
