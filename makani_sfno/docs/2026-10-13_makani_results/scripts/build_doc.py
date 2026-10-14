"""Splice tables.md (from gen_tables.py) into results_template.md -> ../2026-10-13_makani_results.md."""
import os

HERE = os.environ.get("MAKANI_RESULTS_WORK", os.path.dirname(os.path.abspath(__file__)))
blocks = open(os.path.join(HERE, "tables.md")).read().strip("\n").split("\n\n")
fill = {"{{T2}}": blocks[0], "{{T3}}": blocks[1], "{{T4}}": blocks[2], "{{T4b}}": blocks[3],
        "{{T5}}": "\n\n".join(blocks[4:]),
        "{{T6}}": open(os.path.join(HERE, "table6.md")).read().strip("\n"),
        "{{T7}}": open(os.path.join(HERE, "table7.md")).read().strip("\n")}
s = open(os.path.join(HERE, "results_template.md")).read()
for k, v in fill.items():
    assert s.count(k) == 1, k
    s = s.replace(k, v)
assert "{{" not in s
open(os.path.join(HERE, "..", "2026-10-13_makani_results.md"), "w").write(s)
print("BUILD_DOC_OK")
