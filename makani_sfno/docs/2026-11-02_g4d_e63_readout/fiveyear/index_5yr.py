"""emitall receipt for the five-year figures: summary_5yr_index.json (stdlib, python3.6+).

Re-keys figures/summary_5yr.csv and trefht_band_bias.csv by path, and parses the extraction and truth gates out of
their logs, so a claims spec can point at /csv/<figure>/<variable>/<model>/<year>/<field>, /band/<arm>/<year>/<col>
and /gates/... . Derived fields (maxima, counts) are computed here, not in the spec.
Usage: index_5yr.py [DIR]      PASS = INDEX_5YR_OK
"""
import csv
import json
import os
import re
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def main():
    out = {"csv": {}, "band": {}, "gates": {"reroll": {}, "truth": {}}, "derived": {}}
    for r in csv.DictReader(open(os.path.join(D, "figures", "summary_5yr.csv"))):
        key = r["year"] or r["window"]
        node = out["csv"].setdefault(r["figure"], {}).setdefault(r["variable"], {}).setdefault(r["model"], {})
        node[key] = {k: num(r[k]) for k in ("value", "ci_lo", "ci_hi", "sig_area_pct", "significant", "unit", "n")}
    for r in csv.DictReader(open(os.path.join(D, "trefht_band_bias.csv"))):
        out["band"].setdefault(r["arm"], {})[r["year"]] = {k: num(v) for k, v in r.items() if k not in ("arm", "year")}

    ext = open(os.path.join(D, "extract_7809283.log")).read()
    for m in re.finditer(r"^(G4D\d+): (\d+) members.*?max rel ([0-9.e+-]+), (\d+)/(\d+) float32 cells identical; "
                         r"gate 2 vs \S+: (\d+) values, max rel ([0-9.e+-]+)", ext, re.M):
        out["gates"]["reroll"][m.group(1)] = {"members": int(m.group(2)), "maps_max_rel": float(m.group(3)),
                                              "cells_identical": int(m.group(4)), "cells": int(m.group(5)),
                                              "e_values": int(m.group(6)), "e_max_rel": float(m.group(7))}
    tru = open(os.path.join(D, "truth_series.log")).read()
    for m in re.finditer(r"^\s+(\d{4}) rc=0: gate: time mean vs \S+: (\d+) shared channels, max rel diff ([0-9.e+-]+)",
                         tru, re.M):
        out["gates"]["truth"][m.group(1)] = {"channels": int(m.group(2)), "max_rel": float(m.group(3))}

    der = out["derived"]
    by = out["csv"]["bias_by_year"]
    der["max_abs_annual_global_bias"] = {
        v: max(abs(by[v][a][y]["value"]) for a in by[v] for y in by[v][a] if y != "2045-2049") for v in by}
    der["n_trend_ci_excluding_0"] = {
        v: sum(1 for a in by[v] if (by[v][a]["2045-2049"]["ci_lo"] > 0 or by[v][a]["2045-2049"]["ci_hi"] < 0))
        for v in by}
    ann = out["csv"]["annual_bias_TREFHT"]["TREFHT"]
    der["trefht_annual_rms"] = {}
    for y in ("2045", "2046", "2047", "2048", "2049"):
        vals = [ann[a][y]["value"] for a in ann]
        der["trefht_annual_rms"][y] = {"min": min(vals), "max": max(vals)}
    later = [ann[a][y]["value"] for a in ann for y in ("2046", "2047", "2048", "2049")]
    der["trefht_annual_rms"]["2046-2049"] = {"min": min(later), "max": max(later)}
    der["south60_mean_bias"] = {}
    for y in ("2045", "2046"):
        vals = [out["band"][a][y]["south of 60S_mean_bias_K"] for a in out["band"]]
        der["south60_mean_bias"][y] = {"min": min(vals), "max": max(vals)}
    hp = out["csv"]["hovmoller_PRECT"]["PRECT"]
    der["hovmoller_PRECT_max_sig_pct"] = max(hp[a]["2044-11 to 2049-12"]["sig_area_pct"] for a in hp)
    json.dump(out, open(os.path.join(D, "summary_5yr_index.json"), "w"), indent=1, sort_keys=True)
    print("INDEX_5YR_OK reroll_gates=%d truth_gates=%d csv_figures=%d"
          % (len(out["gates"]["reroll"]), len(out["gates"]["truth"]), len(out["csv"])))


if __name__ == "__main__":
    main()
