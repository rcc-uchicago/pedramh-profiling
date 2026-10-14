"""Recompute protocol scores per member from *_channels.csv, gates from *_budget.txt, and paired SEs.

Writes arms.json (per-arm: A, W, per-member A, legacy, gates) and prints a check table.
Pure python3.6 stdlib: no numpy on the login node.
"""
import csv
import json
import math
import os
import re
import statistics as st

D = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/protocol_scores"
TAGS = ["ref1", "ref2", "fd1", "g1fs", "dry1", "g1fix", "g1dry", "g1dry_c", "g1dry_d"]
OUT = os.environ.get("MAKANI_RESULTS_WORK", os.path.dirname(os.path.abspath(__file__)))

arms = {}
for tag in TAGS:
    reported = {r["arm"]: r for r in csv.DictReader(open(os.path.join(D, tag + "_scores.csv")))}
    per = {}
    for r in csv.DictReader(open(os.path.join(D, tag + "_channels.csv"))):
        m = re.search(r"(\d\d)\.nc$", r["member"]).group(1)
        per.setdefault(r["arm"], {}).setdefault(m, []).append(r)
    for arm, members in per.items():
        if arm in arms:
            continue  # same arm re-scored in a later file (FDE04-20, FSE23=FSEMA): keep the first
        a_m, leg_m, groups_by_member = {}, {}, {}
        for m, rows in members.items():
            g = {}
            for r in rows:
                g.setdefault(r["group"], []).append(float(r["r"]))
            gm = {k: sum(v) / len(v) for k, v in g.items()}
            groups_by_member[m] = gm
            a_m[m] = sum(gm.values()) / len(gm)
            leg_m[m] = sum(float(r["rmse_sigma"]) for r in rows) / len(rows)
        gnames = sorted(next(iter(groups_by_member.values())))
        w = max(st.median(groups_by_member[m][g] for m in groups_by_member) for g in gnames)
        arms[arm] = dict(file=tag, A=st.median(a_m.values()), W=w, A_m=a_m,
                         legacy=st.median(leg_m.values()), legacy_m=leg_m,
                         A_rep=float(reported[arm]["A"]), W_rep=float(reported[arm]["W"]),
                         nch=len(next(iter(members.values()))))

# gates 2/3 from budget read-outs
B = re.compile(r"^(\S+?)(\d\d)\s+n=\s*(\d+)\s+dPS=\s*([-+0-9.eE]+).*?TMQ gm\s+([-+0-9.eE]+)->\s*([-+0-9.eE]+) \(min\s+([-+0-9.eE]+)\).*?lastmonth neg=\s*([-+0-9.eE]+)%")
for tag in TAGS:
    p = os.path.join(D, tag + "_budget.txt")
    if not os.path.exists(p):
        continue
    for line in open(p, errors="replace"):
        mm = B.match(line.strip())
        if not mm:
            continue
        arm, mem, n, dps, t0, t1, tmin, neg = mm.groups()
        a = arms.get(arm)
        if a is None:
            continue
        a.setdefault("budget", {})[mem] = dict(n=int(n), dps=float(dps), tmq_min=float(tmin), neg=float(neg))

for a in arms.values():
    b = a.get("budget", {})
    if len(b) == 8:
        a["dps_range"] = (min(x["dps"] for x in b.values()), max(x["dps"] for x in b.values()))
        a["gate2"] = all(x["tmq_min"] > 0 and x["neg"] <= 1.0 for x in b.values())
        a["gate2_bad_members"] = sum(1 for x in b.values() if not (x["tmq_min"] > 0 and x["neg"] <= 1.0))
        a["neg_max"] = max(x["neg"] for x in b.values())
        a["gate3"] = all(abs(x["dps"]) <= 1.0 for x in b.values())

# census (drifting groups, 2047-2045)
for tag in TAGS:
    p = os.path.join(D, tag + "_census.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r["arm"] in arms and "drifting" not in arms[r["arm"]]:
                arms[r["arm"]]["drifting"] = int(r["drifting_groups"])
                arms[r["arm"]]["census_file"] = tag
                try:
                    arms[r["arm"]]["trefht_drift_K"] = float(r["trefht_drift_K"])
                except (KeyError, ValueError):
                    pass

json.dump(arms, open(os.path.join(OUT, "arms.json"), "w"), indent=0)
worst = max(abs(a["A"] - a["A_rep"]) / max(1e-12, abs(a["A_rep"])) for a in arms.values())
print("arms", len(arms), "max rel |A_recomputed - A_reported| =", "%.2e" % worst)
for k, a in sorted(arms.items(), key=lambda kv: kv[1]["A"]):
    if a["A"] > 50:
        continue
    print("%-12s %-7s A=%.3f W=%.2f leg=%.3f g2=%s(%s bad, neg<=%.1f%%) g3=%s dPS=%s drift=%s" % (
        k, a["file"], a["A"], a["W"], a["legacy"], a.get("gate2"), a.get("gate2_bad_members"),
        a.get("neg_max", float("nan")), a.get("gate3"),
        ("%.2f..%.2f" % a["dps_range"]) if "dps_range" in a else "-", a.get("drifting")))


def paired(x, y):
    """mean and SE of per-member A differences x - y, paired by start index."""
    ax, ay = arms[x]["A_m"], arms[y]["A_m"]
    common = sorted(set(ax) & set(ay))
    d = [ax[m] - ay[m] for m in common]
    mean = sum(d) / len(d)
    se = st.stdev(d) / math.sqrt(len(d))
    return mean, se, len(d), sum(1 for v in d if v < 0)


PAIRS = [("FSEMA", "B22"), ("B24", "B22"), ("DRYAIR24", "B22"), ("B22dryair", "B22"), ("B22posclamp", "B22"),
         ("G1E15", "FSE15"), ("G1E16", "FSE16"), ("G1E18", "FSE18"), ("G1E19", "FSE19"),
         ("G1D15", "G1E15"), ("G1D16", "G1E16"), ("G1D17", "G1E17"), ("G1D18", "G1E18"), ("G1D19", "G1E19"),
         ("G1D39", "G1E15"), ("G1D59", "G1D39"), ("G1D39", "FDE32"), ("FDE32", "FSE30"), ("FSE30", "FSEMA"),
         ("FDE32", "B22"), ("G1E15", "B22"), ("G1D39", "B22"), ("G1E15fix", "G1E15"), ("G1E19fix", "G1E19"),
         ("FD22", "FSEMA"), ("FD22dryair", "FD22"), ("FD24dryair", "FD24")]
print("\npaired per-member A differences (x - y): mean, SE, n, members where x better, t")
pairs = {}
for x, y in PAIRS:
    mean, se, n, nb = paired(x, y)
    pairs["%s-%s" % (x, y)] = dict(mean=mean, se=se, n=n, x_better=nb)
    print("%-11s - %-6s %+.3f +- %.3f  n=%d  x better in %d/8  t=%+.1f" % (x, y, mean, se, n, nb, mean / se if se else float("nan")))
json.dump(pairs, open(os.path.join(OUT, "pairs.json"), "w"), indent=0)
print("\ngates (budget) for arms with 8 budget rows:")
for k, a in sorted(arms.items(), key=lambda kv: kv[1]["A"]):
    if "gate2" in a and a["A"] < 50:
        print("%-12s A=%.3f g2=%s (%d bad, neg max %.1f%%) g3=%s dPS %.2f..%.2f drift=%s/%s" % (
            k, a["A"], a["gate2"], a["gate2_bad_members"], a["neg_max"], a["gate3"], a["dps_range"][0], a["dps_range"][1], a.get("drifting"), a.get("census_file")))
json.dump(arms, open(os.path.join(OUT, "arms.json"), "w"), indent=0)
