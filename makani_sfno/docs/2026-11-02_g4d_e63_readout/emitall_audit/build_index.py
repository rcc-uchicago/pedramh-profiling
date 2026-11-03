"""Re-key the protocol-score CSVs into one JSON receipt for emitall specs: {tag: {arm: {...}}}.

No derivation: every value is the CSV's own field, parsed as float/int (scores: A, W, W_group, members;
census: drifting_groups, trefht_drift_K, ps_drift_hPa). Exists because emitall can address CSV rows only by
position, which breaks silently when rows reorder. Re-run whenever a *_scores.csv / *_census.csv changes; specs
declare this file `newer_than` the CSVs, so a stale index is itself a finding. System python3, stdlib only.
"""
import csv
import glob
import json
import os

SD = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/protocol_scores"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protocol_scores_index.json")
idx = {}
for sc in sorted(glob.glob(os.path.join(SD, "*_scores.csv"))):
    tag = os.path.basename(sc)[:-len("_scores.csv")]
    arms = idx.setdefault(tag, {})
    for r in csv.DictReader(open(sc)):
        arms[r["arm"]] = {"A": float(r["A"]), "W": float(r["W"]), "W_group": r["W_group"],
                          "members": int(r["members"])}
    ce = os.path.join(SD, tag + "_census.csv")
    if os.path.exists(ce):
        for r in csv.DictReader(open(ce)):
            d = arms.setdefault(r["arm"], {})
            d["drifting_groups"] = int(r["drifting_groups"])
            d["trefht_drift_K"] = float(r["trefht_drift_K"])
            d["ps_drift_hPa"] = float(r["ps_drift_hPa"])
json.dump(idx, open(OUT, "w"), indent=1, sort_keys=True)
print("INDEX_OK tags=%d arms=%d -> %s" % (len(idx), sum(len(v) for v in idx.values()), OUT))
