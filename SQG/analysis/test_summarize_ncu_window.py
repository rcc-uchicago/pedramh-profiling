"""Synthetic-input test of summarize_ncu_window.py (stdlib only, no ncu / GPU; runs on a login node).

  python3 SQG/analysis/test_summarize_ncu_window.py          # prints NCU_WINDOW_TEST_OK
Builds an `ncu --page raw --csv`-shaped window: 3 entry kernels, then a 4-launch loop body A B A' B (A and A' share a
name but not a grid) three times, rows written out of ID order. Checks: launches are sorted by ID; the period is 4,
not the 2 a name-only match would give; the last period is reported in order with its totals and per-step values
(steps_per_period 2); signatures group by (name, grid, block); sector efficiencies and DRAM MB come through; a window
with no repeat reports period None with a WARN line, not an error; an unreadable input gives a greppable ERROR line.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size",
          "Device", "CC"]
METRICS = ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
           "dram__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_sector_hit_rate.pct",
           "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
           "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct", "launch__registers_per_thread"]
UNITS = [""] * len(LAUNCH) + ["usecond", "Mbyte", "Mbyte", "%", "%", "%", "%", "register/thread"]
FFT = "void regular_fft_factor<(unsigned int)4096, EPT<(unsigned int)16>, float>(kernel_arguments_t<T9>)"
BODY = [("input_concatenate_fusion", "(16384, 1, 1)", "(128, 1, 1)", 200.0, 100.0, 134.0, 50.0),
        (FFT, "(513, 1, 1)", "(4, 256, 1)", 196.0, 140.0, 134.0, 50.0),
        ("input_concatenate_fusion", "(8192, 1, 1)", "(128, 1, 1)", 100.0, 50.0, 67.0, 100.0),
        (FFT, "(513, 1, 1)", "(4, 256, 1)", 190.0, 140.0, 134.0, 25.0)]


def write(path, rows):
    with open(path, "w", newline="") as fh:
        fh.write("==PROF== Disconnected from process 1\n")
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        w.writerow(LAUNCH + METRICS)
        w.writerow(UNITS)
        for r in rows:
            w.writerow(r)


def row(i, name, grid, block, us, rd, wr, ld_eff):
    return ([str(i), "1", "python", "x3005c0s7b0n0", name, "1", "7", block, grid, "0", "8.0"]
            + [f"{us:,.2f}", f"{rd:,.1f}", f"{wr:,.1f}", "80.0", "40.0", f"{ld_eff}", "100", "32"])


def main():
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(HERE, "summarize_ncu_window.py")
        good, out = os.path.join(d, "ncu_lean32_raw.csv"), os.path.join(d, "w.json")
        rows = [row(i, f"wrapped_convert_{i}", "(1000, 1, 1)", "(128, 1, 1)", 10.0, 1.0, 1.0, 100.0) for i in range(3)]
        i = 3
        for _ in range(3):
            for b in BODY:
                rows.append(row(i, *b))
                i += 1
        write(good, rows[::-1])                                      # out of ID order on purpose
        r = subprocess.run([sys.executable, script, good, "--out", out, "--steps-per-period", "2"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode == 0 and "NCU_WINDOW_OK" in r.stdout, r.stdout
        with open(out) as fh:
            s = json.load(fh)
        assert s["launches"] == 15 and s["period"] == 4, (s["launches"], s["period"])
        lp = s["last_period"]
        assert [x["id"] for x in lp] == [11, 12, 13, 14], [x["id"] for x in lp]
        assert [x["name"][:24] for x in lp] == [b[0][:24] for b in BODY]
        assert lp[0]["duration_us"] == 200.0 and lp[0]["dram_read_MB"] == 100.0 and lp[3]["ld_sector_eff_pct"] == 25.0
        assert lp[1]["grid"] == "(513, 1, 1)" and lp[1]["block"] == "(4, 256, 1)" and lp[1]["registers"] == 32.0
        t = s["period_totals"]
        assert t["duration_us"] == 686.0 and t["per_step_us"] == 343.0, t
        assert t["dram_read_MB"] == 430.0 and t["dram_write_MB"] == 469.0 and t["per_step_dram_MB"] == 449.5, t
        sig = {g["signature"]: g for g in s["by_signature"]}
        assert len(s["by_signature"]) == 3 + 3, list(sig)
        assert sig["input_concatenate_fusion (8192, 1, 1)x(128, 1, 1)"]["n"] == 3
        f = sig[f"{FFT} (513, 1, 1)x(4, 256, 1)"]
        assert f["n"] == 6 and f["duration_us"] == 193.0 and f["ld_sector_eff_pct"] == 37.5, f
        assert "period: 686.0 us" in r.stdout, r.stdout

        flat = os.path.join(d, "ncu_flat_raw.csv")                   # no repeat at all
        write(flat, [row(k, f"k{k}", "(1, 1, 1)", "(32, 1, 1)", 1.0, 1.0, 1.0, 100.0) for k in range(5)])
        r = subprocess.run([sys.executable, script, flat, "--out", out],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode == 0 and "WARN NCU_WINDOW_NO_PERIOD" in r.stdout and "NCU_WINDOW_OK" in r.stdout, r.stdout
        with open(out) as fh:
            assert json.load(fh)["period"] is None

        bad = os.path.join(d, "ncu_bad_raw.csv")
        with open(bad, "w") as fh:
            fh.write("not,a,raw,page\n1,2,3,4\n")
        r = subprocess.run([sys.executable, script, bad, "--out", out],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert r.returncode != 0 and "ERROR NCU_WINDOW_BAD_CSV" in r.stdout, r.stdout
    print("NCU_WINDOW_TEST_OK")


if __name__ == "__main__":
    main()
