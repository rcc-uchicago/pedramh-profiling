#!/usr/bin/env python3
"""Tests for ncu_summarize — run with plain python3 on the login node (stdlib only).

Every change ships its test (CLAUDE.md). What these pin down:

* the ==PROF== banner lines ncu prepends do not break the header sniff;
* the long format really is pivoted, so one launch's metrics land together;
* **the ideal-sectors constant is per dtype** — 4 for fp32, 8 for complex64.
  Getting that wrong would make a perfectly-coalesced complex64 copy read as 2x
  uncoalesced and invert item 7's decision, which is the whole deliverable.
"""
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ncu_summarize as ns

HDR = '"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
KC = ("void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast"
      "<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda]"
      "<c10::complex<float>>>>(int, T3)")
KF = ("void at::native::vectorized_elementwise_kernel<(int)4, at::native::"
      "direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda]<float>, "
      "at::detail::Array<char *, (int)2>>(int, T2, T3)")


def _csv(rows):
    out = io.StringIO()
    out.write("==PROF== Connected to process 12345\n")
    out.write("==PROF== Profiling \"direct_copy\" - 0: 0%\n")
    out.write(HDR + "\n")
    for lid, kname, metrics in rows:
        for mname, mval in metrics.items():
            unit = ("Mbyte" if "dram__bytes" in mname else
                    "%" if mname.endswith((".pct", "_elapsed", "_active")) else
                    "sector" if "sectors" in mname else "")
            val = mval / 1e6 if "dram__bytes" in mname else mval
            out.write(f'"{lid}","12345","{kname}","{mname}","{unit}","{val}"\n')
    return out.getvalue()


def _metrics(sec_ld, req_ld, sec_st, req_st, pk, sm, rd, wr, l2=12.5, occ=40.0):
    return {
        "dram__bytes_read.sum": rd,
        "dram__bytes_write.sum": wr,
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": pk,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": sm,
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": sec_ld,
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": req_ld,
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": sec_st,
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": req_st,
        "lts__t_sector_hit_rate.pct": l2,
        "sm__warps_active.avg.pct_of_peak_sustained_active": occ,
    }


def _write(text):
    fh = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    fh.write(text)
    fh.close()
    return fh.name


class TestParse(unittest.TestCase):
    def test_banner_lines_do_not_break_header_sniff(self):
        path = _write(_csv([("0", KC, _metrics(64, 8, 64, 8, 55.0, 4.0, 4e8, 4e8))]))
        launches = ns.load(path)
        self.assertEqual(len(launches), 1, "one launch expected")
        self.assertIn("dram__bytes_read.sum", launches[0][1])

    def test_pivot_groups_metrics_by_launch_not_by_kernel(self):
        # Two launches of the SAME kernel must stay separate rows, or per-launch
        # variance vanishes into a single averaged blob.
        rows = [("0", KC, _metrics(64, 8, 64, 8, 55.0, 4.0, 4e8, 4e8)),
                ("1", KC, _metrics(96, 8, 64, 8, 61.0, 4.0, 5e8, 4e8))]
        launches = ns.load(_write(_csv(rows)))
        self.assertEqual(len(launches), 2)

    def test_dtype_drives_ideal_sectors(self):
        self.assertEqual(ns.IDEAL_SECTORS[ns._dtype(KC)], 8)   # complex64
        self.assertEqual(ns.IDEAL_SECTORS[ns._dtype(KF)], 4)   # fp32

    def test_label_keeps_the_launch_path(self):
        # §4.5's 4x under-count came from confusing these two paths. The label has
        # to name which one it was, or the same mistake is invisible again.
        self.assertIn("nocast", ns._short(KC))
        self.assertIn("vec", ns._short(KF))
        self.assertIn("complex64", ns._short(KC))

    def test_unrecognised_csv_exits_loudly(self):
        path = _write("a,b,c\n1,2,3\n")
        with self.assertRaises(SystemExit) as cm:
            ns.load(path)
        self.assertIn("NCU_CSV_UNRECOGNISED", str(cm.exception))

    def test_coalesced_and_uncoalesced_are_distinguishable_end_to_end(self):
        # The two readings item 7 must separate, run through the real CLI:
        #   coalesced   -> sec/req at the complex64 ideal of 8
        #   uncoalesced -> 4x that, and DRAM bytes inflated to match
        good = _write(_csv([("0", KC, _metrics(64, 8, 64, 8, 24.0, 3.0, 3.8e8, 3.8e8))]))
        bad = _write(_csv([("0", KC, _metrics(256, 8, 64, 8, 88.0, 3.0, 1.5e9, 3.8e8))]))
        run = lambda p: subprocess.run(
            [sys.executable, str(Path(__file__).parent / "ncu_summarize.py"), p],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=True).stdout
        og, ob = run(good), run(bad)
        self.assertIn("8.00", og, og)     # sec/req ld == ideal
        self.assertIn("32.00", ob, ob)    # 256/8 == 4x ideal
        self.assertIn("24.0%", og)
        self.assertIn("88.0%", ob)

    def test_l2_hit_rate_is_reported_and_disambiguates_a_low_dram_peak(self):
        # The case this column exists for: identical LOW dram %peak, opposite meanings.
        # High L2hit% = the cache absorbed the traffic (DRAM %peak is the wrong
        # denominator). Low L2hit% = the kernel really is not moving bytes.
        cached = _write(_csv([("0", KC, _metrics(64, 8, 64, 8, 9.0, 3.0, 4e7, 4e7,
                                                l2=94.0, occ=55.0))]))
        idle = _write(_csv([("0", KC, _metrics(64, 8, 64, 8, 9.0, 3.0, 4e7, 4e7,
                                              l2=3.0, occ=12.0))]))
        run = lambda p: subprocess.run(
            [sys.executable, str(Path(__file__).parent / "ncu_summarize.py"), p],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=True).stdout
        oc, oi = run(cached), run(idle)
        self.assertIn("L2hit%", oc)
        self.assertIn("94.0%", oc, oc)
        self.assertIn("3.0%", oi, oi)
        self.assertIn("55.0%", oc)      # occupancy column present too
        # And the reader is told to check L2 first, since %peak alone misleads here.
        self.assertIn("READ L2hit% BEFORE %peak", oc)


    def test_units_are_normalised_not_ignored(self):
        # ncu auto-scales per row: the SAME metric returns Mbyte on one launch and
        # Gbyte on the next. Reading Metric Value alone adds 1.5 (GB) to 132.72 (MB).
        # This is the bug that would have silently corrupted item 7's headline number.
        hdr = HDR + "\n"
        body = (f'"0","1","{KC}","dram__bytes_read.sum","Mbyte","500"\n'
                f'"0","1","{KC}","dram__bytes_write.sum","Gbyte","1.5"\n')
        launches = ns.load(_write("==PROF== x\n" + hdr + body))
        m = launches[0][1]
        self.assertEqual(m["dram__bytes_read.sum"], 500e6)
        self.assertEqual(m["dram__bytes_write.sum"], 1.5e9)

    def test_unknown_unit_refuses_to_guess(self):
        hdr = HDR + "\n"
        body = f'"0","1","{KC}","dram__bytes_read.sum","furlong","500"\n'
        with self.assertRaises(SystemExit) as cm:
            ns.load(_write(hdr + body))
        self.assertIn("NCU_UNKNOWN_UNIT", str(cm.exception))

    def test_no_python38_only_calls(self):
        # statistics.fmean is 3.8+; the Polaris login node is 3.6.15. This exact
        # trap has been fixed twice in this repo already.
        src = (Path(__file__).parent / "ncu_summarize.py").read_text()
        self.assertNotIn("fmean(", src)          # the call
        self.assertNotIn("import statistics", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
