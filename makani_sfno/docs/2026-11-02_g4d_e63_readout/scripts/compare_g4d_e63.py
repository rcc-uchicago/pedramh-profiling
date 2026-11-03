"""G4D after 63 epochs: K=56 length gain + gap to A, and the 5-yr pool re-rank (prereg.md here).

Part 1 reads only k56_summary.csv (as compare_depth_length.py, 10-21). Part 2 reuses the fp32 re-rank's loaders
and the precision cross's health rule / t p-value (eval-protocol @ 5848643d), as g4d_compare.py did (10-19).
System python3 (3.6), stdlib only. Env overrides: CMP_E, CMP_<ARM> (a K=56 scores dir), NF_SD (protocol_scores),
CMP_OUT. Writes results.txt (prose) and results.json (the same numbers, keyed, as the receipt for emitall claims
specs; ratio names use "_over_" for "/" so they are JSON-pointer safe). PASS = G4D_E63_OK ...
"""
import csv
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, "/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/"
                   "eval-protocol/makani_sfno/polaris")
import fp32_rerank as fr  # noqa: E402
import precision_cross_analysis as pca  # noqa: E402

E = os.environ.get("CMP_E", "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval")
K_ARMS = (("A243", "prod1n_b32_sgdr_K56"), ("A16", "A16_K56"), ("G4D16", "G4D16_K56"), ("G4D36", "G4D36_K56"),
          ("G4D56", "G4D56_K56"), ("G4D43", "G4D43_K56"), ("G4D63", "G4D63_K56"))
KDIRS = {a: os.environ.get("CMP_" + a, os.path.join(E, d, "scores")) for a, d in K_ARMS}
LEADS = (6, 12, 24, 36, 48, 72, 126, 168, 336)
RULE_LEADS = (6, 24)
CLOSE_LEADS = (6, 12, 24)
SD = os.environ.get("NF_SD", E + "/protocol_scores")
NEW = "g4d63"
TAGS = ("g1fs", "g1fix", "g1dry", "g1dry_c", "g1dry_d", "pxc", "f32a", "g4e", "g4fix", "g4d", NEW)
PICK = "G4D16"       # g4d_dryair_twin_2026-10-19/results.txt; fr.CURRENT_PICK still says G1D39
NEW_ARMS = ("G4D36", "G4D56", "G4D43", "G4D63")
PAIRS = (("G4D36", "G4D16"), ("G4D56", "G4D16"), ("G4D43", "G4D23"), ("G4D63", "G4D23"))
SIG = fr.SIG

out = []
P = out.append


# ---------------------------------------------------------------- part 1: K=56
def k56(d):
    res = {}
    for r in csv.DictReader(open(os.path.join(d, "k56_summary.csv"))):
        res.setdefault((r["metric"], int(r["lead_hours"])), {})[r["channel"]] = float(r["mean_over_ic"])
    return res


def contrast(num, den, chs):
    r = [num[c] / den[c] for c in chs]
    return {"med": st.median(r), "worse": sum(x > 1 for x in r), "better": sum(x < 1 for x in r),
            "mlog": sum(math.log(x) for x in r) / len(r)}


def rule1(c):
    """Pre-registered rule 1 on one lead's contrast (99 channels => 2/3 = 66)."""
    need = math.ceil(2 * (c["worse"] + c["better"]) / 3.0)
    if c["med"] <= 0.97 and c["better"] >= need:
        return "HELPS"
    if c["med"] >= 1.03 and c["worse"] >= need:
        return "HURTS"
    return "FLAT"


def rule2(s):
    if not math.isfinite(s):
        return "n/a"
    return "CLOSED" if s >= 1 else "MOSTLY" if s >= 0.5 else "PARTLY" if s >= 0.25 else "NO"


J = {"part1": None, "part2": None}       # emitall receipt
present = {a: os.path.isfile(os.path.join(d, "k56_summary.csv")) for a, d in KDIRS.items()}
k_ok = all(present.values())
R1 = {}
if not k_ok:
    P("== PART 1 (K=56) INCOMPLETE: missing %s ==" % ", ".join(a for a in present if not present[a]))
else:
    K = {a: k56(d) for a, d in KDIRS.items()}
    com = sorted(set.intersection(*[set(K[a][("rmse", 6)]) for a in K]))
    for a in K:
        for l in LEADS:
            bad = [c for c in com if not (K[a][("rmse", l)][c] > 0 and math.isfinite(K[a][("rmse", l)][c]))]
            if bad:
                raise SystemExit("ERROR NONPOSITIVE_RMSE arm=%s lead=%d channels=%s" % (a, l, bad[:3]))
    rm = {a: {l: K[a][("rmse", l)] for l in LEADS} for a in K}
    ROWS = (("G4D36/G4D16", "G4D36", "G4D16"), ("G4D56/G4D16", "G4D56", "G4D16"), ("G4D43/G4D16", "G4D43", "G4D16"),
            ("G4D63/G4D16", "G4D63", "G4D16"), ("G4D16/A243", "G4D16", "A243"), ("G4D56/A243", "G4D56", "A243"),
            ("G4D63/A243", "G4D63", "A243"), ("A16/A243", "A16", "A243"))
    C = {name: {l: contrast(rm[n][l], rm[d][l], com) for l in LEADS} for name, n, d in ROWS}
    P("== PART 1. K=56 RMSE ratio by lead, median over the %d common channels (>1 = numerator worse) ==" % len(com))
    P("  %-13s" % "lead h" + "".join("%8d" % l for l in LEADS))
    for name, _, _ in ROWS:
        P("  %-13s" % name + "".join("%8.3f" % C[name][l]["med"] for l in LEADS))
    P("  numerator worse on (of %d channels):" % len(com))
    for name, _, _ in ROWS:
        P("  %-13s" % name + "".join("%8d" % C[name][l]["worse"] for l in LEADS))

    P("")
    P("== gap closed: channel-mean ln(G4D16/X) / channel-mean ln(G4D16/A243), where G4D16's median gap > 1.02 ==")
    close = {}
    for x in ("G4D36", "G4D56", "G4D63"):
        cells = []
        for l in CLOSE_LEADS:
            g = C["G4D16/A243"][l]
            s = (-C["%s/G4D16" % x][l]["mlog"] / g["mlog"] + 0.0) if g["med"] > 1.02 and g["mlog"] > 0 else float("nan")
            close[(x, l)] = s
            cells.append("%3d h %s" % (l, "%.2f (%s)" % (s, rule2(s)) if math.isfinite(s) else "n/a"))
        P("  %-6s " % x + "   ".join(cells))

    P("")
    P("== 14-day accuracy, median over the %d common channels ==" % len(com))
    P("  %-6s %9s %9s %9s %9s" % ("", "NRMSE126", "NRMSE336", "ACC126", "ACC336"))
    for a in ("A243", "A16", "G4D16", "G4D36", "G4D56", "G4D43", "G4D63"):
        P("  %-6s " % a + " ".join("%9.3f" % st.median(K[a][(m, l)][c] for c in com)
                                   for m in ("nrmse", "acc") for l in (126, 336)))

    P("")
    P("== pre-registered read-out ==")
    for l in RULE_LEADS:
        c = C["G4D56/G4D16"][l]
        R1[l] = rule1(c)
        P("  rule 1, %2d h: G4D56/G4D16 %.3f, better on %d/%d -> %s   (G4D36/G4D16 %.3f -> %s)" % (
            l, c["med"], c["better"], len(com), R1[l], C["G4D36/G4D16"][l]["med"], rule1(C["G4D36/G4D16"][l])))
    for l in CLOSE_LEADS:
        P("  rule 2, %2d h: gap closed by G4D56 %s -> %s" % (
            l, "%.2f" % close[("G4D56", l)] if math.isfinite(close[("G4D56", l)]) else "n/a",
            rule2(close[("G4D56", l)])))
    R1["close6"] = rule2(close[("G4D56", 6)])
    key = lambda name: name.replace("/", "_over_")
    J["part1"] = {
        "channels": len(com), "leads_h": list(LEADS),
        "ratio_median": {key(nm): {str(l): C[nm][l]["med"] for l in LEADS} for nm, _, _ in ROWS},
        "worse_count": {key(nm): {str(l): C[nm][l]["worse"] for l in LEADS} for nm, _, _ in ROWS},
        "gap_closed": {x: {str(l): close[(x, l)] for l in CLOSE_LEADS} for x in ("G4D36", "G4D56", "G4D63")},
        "acc14": {a: {"%s%d" % (m, l): st.median(K[a][(m, l)][c] for c in com) for m in ("nrmse", "acc")
                      for l in (126, 336)} for a in ("A243", "A16", "G4D16", "G4D36", "G4D56", "G4D43", "G4D63")},
        "rule1": {str(l): R1[l] for l in RULE_LEADS},
        "rule2": {str(l): rule2(close[("G4D56", l)]) for l in CLOSE_LEADS}}

# ---------------------------------------------------------------- part 2: 5-yr protocol
P("")
arms, bud, drift = {}, {}, {}
for tag in TAGS:
    if not os.path.isfile(os.path.join(SD, tag + "_channels.csv")):
        continue
    for a, v in fr.load_tag(SD, tag).items():
        arms.setdefault(a, v)
    for a, n in fr.census(SD, tag).items():
        drift.setdefault(a, n)
    for a, mem in pca.budgets(os.path.join(SD, tag + "_budget.txt")).items():
        bud.setdefault(a, mem)
p_ok = any(arms[a]["tag"] == NEW for a in arms)
winner = None
if not p_ok:
    P("== PART 2 (5-yr) INCOMPLETE: no %s_channels.csv in %s ==" % (NEW, SD))
else:
    def row(a):
        v = arms.get(a)
        if v is None:
            return None
        groups = sorted(next(iter(v["G"].values())))
        g = fr.gates(bud, a)
        dps = [abs(x[0]) for x in bud.get(a, {}).values() if math.isfinite(x[0])]
        return dict(A=st.median(v["A_m"].values()), A_m=v["A_m"], g=g, ok=all(g), drift=drift.get(a),
                    W=max(st.median(v["G"][m][k] for m in v["G"]) for k in groups),
                    h=sum(pca.healthy(bud, a, m) for m in v["A_m"]), dps=st.median(dps) if dps else float("nan"))

    def gstr(r):
        return "pass" if r["ok"] else "fail " + ",".join(str(i + 1) for i, k in enumerate(r["g"]) if not k)

    def fpaired(xm, ym):
        """fr.paired with p = 1 at t = 0 (its t_pvalue takes log(0) there; g4d_compare.py hit the same)."""
        d = [xm[m] - ym[m] for m in sorted(set(xm) & set(ym))]
        if len(d) < 2:
            return float("nan"), float("nan"), float("nan"), len(d)
        mean, se = sum(d) / len(d), st.stdev(d) / math.sqrt(len(d))
        t = mean / se if se > 0 else (math.copysign(float("inf"), mean) if mean else 0.0)
        return mean, se, (1.0 if t == 0 else pca.t_pvalue(t, len(d) - 1)), len(d)

    def paired_healthy(x, y):
        """x - y by start index, healthy in both: mean +- SE, t p-value, n (as g4d_compare.pooled, one pair)."""
        xm, ym = arms[x]["A_m"], arms[y]["A_m"]
        d = [xm[m] - ym[m] for m in set(xm) & set(ym) if pca.healthy(bud, x, m) and pca.healthy(bud, y, m)]
        if len(d) < 2:
            return float("nan"), float("nan"), float("nan"), len(d)
        mean, se = sum(d) / len(d), st.stdev(d) / math.sqrt(len(d))
        t = mean / se if se > 0 else (math.copysign(float("inf"), mean) if mean else 0.0)
        return mean, se, (1.0 if t == 0 else pca.t_pvalue(t, len(d) - 1)), len(d)

    P("== PART 2. 5-yr protocol, new arms (fp32 rollouts, trained-in dry air) ==")
    for a in NEW_ARMS + ("G4D16", "G4D23"):
        r = row(a)
        if r is None:
            P("  %-6s missing" % a)
            continue
        P("  %-6s %d/8 healthy  %-11s A=%6.3f W=%5.2f drifting=%s med|dPS|=%.2f hPa%s" % (
            a, r["h"], gstr(r), r["A"], r["W"], r["drift"], r["dps"], "" if a in NEW_ARMS else "   (reference)"))
    P("  paired dA (by start date, healthy in both):")
    for x, y in PAIRS:
        if x in arms and y in arms:
            P("    %s - %s = %+.3f +- %.3f (p=%.2g, n%d)" % ((x, y) + paired_healthy(x, y)))
        else:
            P("    %s - %s: missing" % (x, y))

    pool = [a for a in arms if (fr.FP32_REF.match(a) and arms[a]["tag"] in fr.REF_TAGS)
            or arms[a]["tag"] in ("f32a", "g4e", "g4fix", "g4d", NEW)]
    rows = {a: row(a) for a in pool}
    passers = sorted((a for a in rows if rows[a]["ok"]), key=lambda a: rows[a]["A"])
    P("")
    P("== parent sec.5 rule: fp32 pool + f32a + g4e + g4fix + g4d + %s: %d arms, %d pass gates 1-3 ==" % (
        NEW, len(rows), len(passers)))
    tie = []
    if passers:
        lead = passers[0]
        tie = [lead]
        for c in passers[1:]:
            m, se = fpaired(rows[c]["A_m"], rows[lead]["A_m"])[:2]
            if not m > SIG * se:
                tie.append(c)
        winner = min(tie, key=lambda a: (rows[a]["drift"] if rows[a]["drift"] is not None else 99, rows[a]["W"]))
        for a in passers[:20]:
            d = "leader" if a == lead else "%+.3f +- %.3f; p=%.2g; n%d" % fpaired(rows[a]["A_m"], rows[lead]["A_m"])
            P("  %-10s A=%.3f W=%.2f drifting=%s %s%s" % (a, rows[a]["A"], rows[a]["W"], rows[a]["drift"], d,
                                                        "  [tie set]" if a in tie and len(tie) > 1 else ""))
        if len(passers) > 20:
            P("  ... %d more passers, none in the tie set" % (len(passers) - 20) if all(a not in tie for a in passers[20:])
              else "  ... %d more passers" % (len(passers) - 20))
        P("winner: %s (%s)%s" % (winner, "outright" if len(tie) == 1 else "tie-break among " + ",".join(tie),
                                 "  (unchanged)" if winner == PICK else "  ** REPLACES %s **" % PICK))
    newp = [a for a in passers if arms[a]["tag"] == NEW]
    P("%s gate 1-3 passers: %s" % (NEW, ", ".join(newp) or "none"))
    J["part2"] = {
        "arms": {a: {"A": r["A"], "W": r["W"], "drifting": r["drift"], "healthy": r["h"], "gates_1_3": r["ok"],
                     "med_abs_dps_hPa": r["dps"]}
                 for a in NEW_ARMS + ("G4D16", "G4D23") for r in [row(a)] if r is not None},
        "paired": {"%s-%s" % (x, y): dict(zip(("mean", "se", "p", "n"), paired_healthy(x, y)))
                   for x, y in PAIRS if x in arms and y in arms},
        "pool": {"n_arms": len(rows), "n_passers": len(passers), "leader": passers[0] if passers else None,
                 "tie_set": tie, "winner": winner, "replaces_pick": None if winner is None else winner != PICK,
                 "pick_before": PICK, "new_passers": newp}}

P("")
done = k_ok and p_ok
P("%s parts=%s%s rule1_6h=%s rule1_24h=%s close6=%s winner=%s replaces=%s" % (
    "G4D_E63_OK" if done else "G4D_E63_PARTIAL", "K56" if k_ok else "-", "+5yr" if p_ok else "",
    R1.get(6, "-"), R1.get(24, "-"), R1.get("close6", "-"), winner or "-",
    "-" if winner is None else int(winner != PICK)))
text = "\n".join(out) + "\n"
txt_path = os.environ.get("CMP_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")
open(txt_path, "w").write(text)
J["token"] = out[-1]
json.dump(J, open(os.path.splitext(txt_path)[0] + ".json", "w"), indent=1, sort_keys=True)
print(text, end="")
