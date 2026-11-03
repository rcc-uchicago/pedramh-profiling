"""Build the text views and battery specs for the 2026-11-02 emitall audit (polaris_emitall_audit.pbs).

Views (plain text, fed to battery.py's "txt" input; never its "tex" path, which strips text after '%'):
  knowledge      makani_sfno/KNOWLEDGE.md without sections 9-10, which quote retired claims on purpose
  changelog      CHANGELOG.md decisions-log entries dated >= 2026-10-07 (after the 10-06 retractions); the
                 append-only history before that legitimately contains the claims later retracted
  writeup_1013   the 2026-10-13 results write-up, whole
  configs_1102   the 2026-11-02 configurations doc, whole
Also writes lint_targets.txt (one path per line). Stdlib only.
"""
import json
import os
import re
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = "/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling"
KW = REPO + "/.claude/worktrees/makani-knowledge"
EV = "/lus/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval"
CUTOFF = "2026-10-07"


def knowledge_view():
    L = open(KW + "/makani_sfno/KNOWLEDGE.md").read().split("\n")
    out, skip = [], False
    for l in L:
        if l.startswith("## "):
            skip = bool(re.match(r"## (9|10)\b", l))
        if not skip:
            out.append(l)
    return "\n".join(out)


def changelog_view():
    t = open(REPO + "/CHANGELOG.md").read()
    log = t[t.index("## Decisions / changes log"):]
    log = log[:log.index("\n## ", 5)] if "\n## " in log[5:] else log
    out, keep = [], False
    for l in log.split("\n"):
        m = re.match(r"^- \*\*(\d{4}-\d{2}-\d{2})", l)
        if m:
            keep = m.group(1) >= CUTOFF
        if keep:
            out.append(l)
    return "\n".join(out)


VIEWS = {
    "knowledge": knowledge_view,
    "changelog": changelog_view,
    "writeup_1013": lambda: open(KW + "/makani_sfno/docs/2026-10-13_makani_results/2026-10-13_makani_results.md").read(),
    "configs_1102": lambda: open(KW + "/makani_sfno/docs/2026-11-02_makani_configurations/"
                                      "2026-11-02_makani_configurations.md").read(),
}


def main():
    R = json.load(open(os.path.join(OUT, "battery_rows.json")))
    ctx = R["retraction_context"]
    for name, fn in VIEWS.items():
        text = fn()
        open(os.path.join(OUT, "view_%s.txt" % name), "w").write(text)
        rows = []
        for r in R["rows"]:
            r = dict(r)
            if r["class"] == "NEVER":
                r["unless_context"] = ctx
            rows.append(r)
        if name == "changelog":
            rows += R["frozen_recent_changelog"]
        spec = {"label": "retraction-guard-" + name, "base": OUT, "txt": "view_%s.txt" % name, "rows": rows}
        json.dump(spec, open(os.path.join(OUT, "battery_%s.json" % name), "w"), indent=1, ensure_ascii=False)
        print("view %-13s %7d chars -> battery_%s.json" % (name, len(text), name))
    targets = [
        EV + "/receipts/build_index.py",
        EV + "/g4d_e63_2026-11-02/compare_g4d_e63.py",
        EV + "/g4d_e63_2026-11-02/maps",
        EV + "/depth_vs_length_2026-10-21/compare_depth_length.py",
        EV + "/g4d16_k56_lagged_2026-10-19/compare_g4d16_vs_a.py",
        EV + "/g4d_dryair_twin_2026-10-19/g4d_compare.py",
        KW + "/makani_sfno/docs/2026-11-02_makani_configurations/extract_configs.py",
        KW + "/makani_sfno/docs/2026-11-02_makani_configurations/build_claims.py",
        KW + "/makani_sfno/docs/2026-10-13_makani_results/scripts",
        REPO + "/ACE2_retrain/polaris/make_results_table.py",
        REPO + "/ACE2_retrain/polaris/parse_ace2_scaling.py",
        REPO + "/ACE2_retrain/polaris/parse_nccl_tests.py",
        REPO + "/ACE2_retrain/polaris/compare_resume_trace.py",
    ]
    # the evaluation scorers, minus their tests: test fixtures embed synthetic log text, which lint's digit-literal
    # check reads as hand-typed numbers (run 1: 19 such hits, all in test_parse_makani_scaling.py)
    pol = REPO + "/.claude/worktrees/eval-protocol/makani_sfno/polaris"
    targets += sorted(os.path.join(pol, f) for f in os.listdir(pol) if f.endswith(".py") and not f.startswith("test_"))
    missing = [p for p in targets if not os.path.exists(p)]
    if missing:
        print("WARN LINT_TARGET_MISSING %s" % missing)
    open(os.path.join(OUT, "lint_targets.txt"), "w").write("\n".join(p for p in targets if os.path.exists(p)) + "\n")
    print("AUDIT_VIEWS_OK views=%d lint_targets=%d" % (len(VIEWS), len(targets) - len(missing)))


if __name__ == "__main__":
    sys.exit(main())
