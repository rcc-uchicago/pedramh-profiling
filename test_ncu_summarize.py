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
            out.write(f'"{lid}","12345","{kname}","{mname}","","{mval}"\n')
    return out.getvalue()


def _metrics(sec_ld, req_ld, sec_st, req_st, pk, sm, rd, wr):
    return {
        "dram__bytes_read.sum": rd,
        "dram__bytes_write.sum": wr,
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": pk,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": sm,
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": sec_ld,
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": req_ld,
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": sec_st,
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": req_st,
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

    def test_no_python38_only_calls(self):
        # statistics.fmean is 3.8+; the Polaris login node is 3.6.15. This exact
        # trap has been fixed twice in this repo already.
        src = (Path(__file__).parent / "ncu_summarize.py").read_text()
        self.assertNotIn("fmean(", src)          # the call
        self.assertNotIn("import statistics", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
