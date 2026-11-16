"""test_summarize_ncu_wu.py -- summarize_ncu_wu.py on a synthetic `--page raw` export (stdlib). Prints
WU_NCU_PARSER_TEST_OK."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize_ncu_wu as S                                         # noqa: E402

HEAD = ["ID", "Kernel Name", S.M["time"], S.M["dram_rd"], S.M["dram_wr"], S.M["ld_sec"], S.M["ld_req"], S.M["st_sec"],
        S.M["st_req"], S.M["ld_use"], S.M["st_use"], S.M["regs"], S.M["occ"]]
UNITS = ["", "", "usecond", "Mbyte", "Mbyte", "sector", "", "sector", "", "%", "%", "register/thread", "%"]
ROWS = [["0", "upd_compact_abc_cuda_kernel_forward", "200", "300", "90", "8,000,000", "1,000,000", "4000000",
         "500000", "100", "100", "40", "80"],
        ["1", "upd_compact_abc_cuda_kernel_forward", "220", "300", "90", "8000000", "1000000", "4000000", "500000",
         "100", "100", "40", "80"],
        ["2", "loop_fusion_7", "1", "1", "1", "16", "1", "8", "1", "50", "50", "32", "60"]]


def main():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "raw.csv")
    with open(p, "w") as f:
        for r in [HEAD, UNITS] + ROWS:
            f.write(",".join('"%s"' % x for x in r) + "\n")
    out = S.summarize(p)
    k = out["upd_compact_abc_cuda_kernel_forward"]
    assert k["instances"] == 2 and abs(k["time"] - 210) < 1e-9, k
    assert abs(k["ld_sec_per_req"] - 8.0) < 1e-12 and abs(k["st_sec_per_req"] - 8.0) < 1e-12, k
    assert abs(k["dram_rd"] - 300) < 1e-9, k
    assert abs(out["loop_fusion_7"]["ld_sec_per_req"] - 16) < 1e-12
    print("WU_NCU_PARSER_TEST_OK")


if __name__ == "__main__":
    main()
