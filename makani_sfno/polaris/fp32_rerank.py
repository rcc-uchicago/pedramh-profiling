"""Re-rank the fp32-rolled pool (docs/2026-10-13_fp32_reroll_prereg.md, docs/makani-knowledge).

Pool = fp32-rolled arms of the reference score tags (G1E*, G1E*fix, G1D*, FSEF*) + every arm of the new tag.
Gates 1-3 from the budget read-out; rank passers by A (median over members of the per-member A); the leader
wins outright if it beats every other passer by > 3.3 paired SE, else the passers within 3.3 SE tie and the tie
goes to fewer drifting groups, then lower W. Also reports each re-rolled arm's paired bf16 -> fp32 change.

Stdlib only.  PASS = FP32_RERANK_OK winner=<arm>
"""
import argparse
import csv
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precision_cross_analysis as pca  # noqa: E402  (budgets, healthy, t_pvalue)

REF_TAGS = ("g1fs", "g1fix", "g1dry", "g1dry_c", "g1dry_d", "pxc")
FP32_REF = re.compile(r"^(G1E\d\d(fix)?|G1D\d\d|FSEF\d\d)$")      # arms of REF_TAGS that were rolled in fp32
ORIG_TAGS = ("g1fs", "fd1", "ref1", "ref2")                        # where the bf16 originals were scored
CURRENT_PICK = "G1D39"
SIG = 3.3


def original(arm):
    """bf16-rolled original of a re-rolled arm (prereg table), or None."""
    for pat, rep in ((r"^FSEF(\d\d)$", r"FSE\1"), (r"^FDEF(\d\d)$", r"FDE\1"), (r"^BF(\d\d)$", r"B\1"),
                     (r"^BF(\d\d)fix$", r"B\1dryair"), (r"^DRYAIRF(\d\d)$", r"DRYAIR\1")):
        if re.match(pat, arm):
            return re.sub(pat, rep, arm)
    return None


def load_tag(scores_dir, tag):
    """{arm: {"A_m": {m: A}, "G": {m: {group: mean r}}}} from <tag>_channels.csv."""
    out = {}
    path = os.path.join(scores_dir, tag + "_channels.csv")
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            m = re.search(r"(\d\d)\.nc$", r["member"]).group(1)
            out.setdefault(r["arm"], {}).setdefault(m, {}).setdefault(r["group"], []).append(float(r["r"]))
    res = {}
    for arm, mem in out.items():
        G = {m: {g: sum(v) / len(v) for g, v in gs.items()} for m, gs in mem.items()}
        res[arm] = {"A_m": {m: sum(g.values()) / len(g) for m, g in G.items()}, "G": G, "tag": tag}
    return res


def census(scores_dir, tag):
    out = {}
    path = os.path.join(scores_dir, tag + "_census.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            out[r["arm"]] = int(r["drifting_groups"])
    return out


def paired(xm, ym, keep=None):
    common = [m for m in sorted(set(xm) & set(ym)) if keep is None or keep(m)]
    d = [xm[m] - ym[m] for m in common]
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), len(d)
    mean = sum(d) / len(d)
    se = st.stdev(d) / math.sqrt(len(d))
    t = mean / se if se > 0 else (math.copysign(float("inf"), mean) if mean else 0.0)
    return mean, se, pca.t_pvalue(t, len(d) - 1), len(d)


def gates(bud, arm):
    b = bud.get(arm, {})
    if len(b) < 8:
        return False, False, False
    g1 = all(v[3] == pca.N_STEPS0 - pca.STRIDE * int(m) for m, v in b.items())
    g2 = all(v[1] > 0 and v[2] <= 1.0 for v in b.values())
    g3 = all(math.isfinite(v[0]) and abs(v[0]) <= 1.0 for v in b.values())
    return g1, g2, g3


def run(scores_dir, new_tag, out_path):
    pool, drift, bud = {}, {}, {}
    for tag in REF_TAGS + (new_tag,):
        arms = load_tag(scores_dir, tag)
        for arm, v in arms.items():
            if (tag == new_tag or FP32_REF.match(arm)) and arm not in pool:
                pool[arm] = v
        for arm, n in census(scores_dir, tag).items():
            drift.setdefault(arm, n)
        for arm, mem in pca.budgets(os.path.join(scores_dir, tag + "_budget.txt")).items():
            bud.setdefault(arm, mem)
    rows = {}
    for arm, v in pool.items():
        A = st.median(v["A_m"].values())
        groups = sorted(next(iter(v["G"].values())))
        W = max(st.median(v["G"][m][g] for m in v["G"]) for g in groups)
        g1, g2, g3 = gates(bud, arm)
        rows[arm] = dict(A=A, W=W, g1=g1, g2=g2, g3=g3, ok=g1 and g2 and g3, drift=drift.get(arm), A_m=v["A_m"])
    passers = sorted((a for a in rows if rows[a]["ok"]), key=lambda a: rows[a]["A"])
    lines = ["fp32 re-rank (prereg 2026-10-13): pool %d arms, %d pass gates 1-3" % (len(rows), len(passers))]
    winner, outright, tie = None, False, []
    if passers:
        lead = passers[0]
        tie = [lead]
        for c in passers[1:]:
            mean, se, p, n = paired(rows[c]["A_m"], rows[lead]["A_m"])
            if not (mean > SIG * se):
                tie.append(c)
        outright = len(tie) == 1
        winner = min(tie, key=lambda a: (rows[a]["drift"] if rows[a]["drift"] is not None else 99, rows[a]["W"]))
    lines.append("gate 1-3 passers by A (A minus leader: paired mean +- SE; p; members):")
    for a in passers:
        r = rows[a]
        if a == passers[0]:
            d = "leader"
        else:
            mean, se, p, n = paired(r["A_m"], rows[passers[0]]["A_m"])
            d = "%+.3f +- %.3f; p=%.2g; n%d" % (mean, se, p, n)
        lines.append("  %-10s A=%.3f W=%.2f drifting=%s %s%s" % (a, r["A"], r["W"], r["drift"], d,
                                                              "  [tie set]" if a in tie and not outright else ""))
    failed = sorted((a for a in rows if not rows[a]["ok"]), key=lambda a: rows[a]["A"])
    lines.append("not passing gates 1-3 (lowest A first): " + ", ".join(
        "%s A=%.2f fails %s" % (a, rows[a]["A"], ",".join(g for g, k in (("1", "g1"), ("2", "g2"), ("3", "g3"))
                                                          if not rows[a][k])) for a in failed[:12]))
    # bf16 -> fp32 change for each re-rolled arm (members healthy in both)
    orig_rows = {}
    for tag in ORIG_TAGS:
        for arm, v in load_tag(scores_dir, tag).items():
            orig_rows.setdefault(arm, v)
        for arm, mem in pca.budgets(os.path.join(scores_dir, tag + "_budget.txt")).items():
            bud.setdefault(arm, mem)
    lines.append("bf16 -> fp32 rollout change for the re-rolled arms (fp32 minus bf16, healthy members in both):")
    for arm in sorted(a for a in rows if original(a) and rows[a].get("A_m") and pool[a]["tag"] == new_tag):
        o = original(arm)
        if o not in orig_rows:
            lines.append("  %-10s original %s not found" % (arm, o))
            continue
        mean, se, p, n = paired(rows[arm]["A_m"], orig_rows[o]["A_m"],
                                keep=lambda m, a=arm, b=o: pca.healthy(bud, a, m) and pca.healthy(bud, b, m))
        lines.append("  %-10s vs %-10s A %.2f -> %.2f  change %+.3f +- %.3f; p=%.2g; members %d" % (
            arm, o, st.median(orig_rows[o]["A_m"].values()), rows[arm]["A"], mean, se, p, n))
    if winner:
        lines.append("decision: %s %s (%s)" % (winner, "wins outright" if outright else "wins the tie-break",
                                              "same as the current pick" if winner == CURRENT_PICK
                                              else "REPLACES the current pick %s" % CURRENT_PICK))
        lines.append("FP32_RERANK_OK winner=%s outright=%d tie_set=%s" % (winner, int(outright), ",".join(tie)))
    else:
        lines.append("ERROR FP32_RERANK_NO_PASSERS")
    text = "\n".join(lines) + "\n"
    if out_path:
        open(out_path, "w").write(text)
    print(text, end="")
    return winner, outright, tie, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--tag", required=True, help="score tag of the new fp32 re-rolls")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    run(a.scores_dir, a.tag, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
