"""Write fiveyear/claims_window.json: the numbers quoted about the 2048-2049 bias maps (chat 2026-11-04, README)."""
import json

F = "/lus/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/g4d_e63_2026-11-02/fiveyear"
ARMS = "{G4D16|G4D36|G4D43|G4D56|G4D63}"
VARS = ("TREFHT", "PRECT", "U10", "U_l17", "Z3_l12")
claims = []


def c(label, quoted, expr, section="window"):
    claims.append({"section": section, "label": label, "quoted": quoted, "expr": expr})


table = {"TREFHT": ("0.690", "0.821", "0.912"), "PRECT": ("0.380", "0.464", "0.464"),
         "U10": ("0.228", "0.245", "0.258"), "U_l17": ("0.356", "0.408", "0.417"), "Z3_l12": ("66.3", "77.8", "92.9")}
for v, qs in table.items():
    for win, q in zip(("2045-2047", "2046-2047", "2048-2049"), qs):
        c("G4D16 %s RMS bias %s" % (v, win), q, "f('w','/%s/G4D16/%s/value')" % (v, win), "g4d16 table")
ranges = {"TREFHT": ("1.093", "1.178"), "PRECT": ("0.985", "1.159"), "U10": ("0.995", "1.163"),
          "U_l17": ("1.022", "1.242"), "Z3_l12": ("1.141", "1.218")}
for v, (lo, hi) in ranges.items():
    c("%s ratio 2048-49/2046-47 min over arms" % v, lo, "min(vals('w','/%s/%s/ratio_report_vs_like'))" % (v, ARMS))
    c("%s ratio 2048-49/2046-47 max over arms" % v, hi, "max(vals('w','/%s/%s/ratio_report_vs_like'))" % (v, ARMS))
c("ratios above 1", "23", "len(where(vals('w','/{%s}/%s/ratio_report_vs_like'), '>', 1))" % ("|".join(VARS), ARMS))
c("PRECT G4D16 ratio", "1.000", "f('w','/PRECT/G4D16/ratio_report_vs_like')")
c("ratios in total", "25", "len(vals('w','/{%s}/%s/ratio_report_vs_like'))" % ("|".join(VARS), ARMS))
c("variables where G4D43 has the largest ratio", "5",
  " + ".join("(f('w','/%s/G4D43/ratio_report_vs_like') == max(vals('w','/%s/%s/ratio_report_vs_like')))"
             % (v, v, ARMS) for v in VARS))
c("Z3_l12 2048-49 G4D16", "92.9", "f('w','/Z3_l12/G4D16/2048-2049/value')")
c("Z3_l12 2048-49 other arms min", "74.0", "min(vals('w','/Z3_l12/{G4D36|G4D43|G4D56|G4D63}/2048-2049/value'))")
c("Z3_l12 2048-49 other arms max", "81.5", "max(vals('w','/Z3_l12/{G4D36|G4D43|G4D56|G4D63}/2048-2049/value'))")
# annual pattern RMS (figures/summary_5yr.csv via summary_5yr_index.json)
A = "/csv/annual_bias_%s/%s/%s/%d/value"
for v, arm, y, q in (("Z3_l12", "G4D16", 2045, "44.6"), ("Z3_l12", "G4D16", 2049, "95.5"),
                     ("Z3_l12", "G4D43", 2045, "35.1"), ("Z3_l12", "G4D43", 2049, "77.8"),
                     ("TREFHT", "G4D16", 2045, "0.552"), ("TREFHT", "G4D16", 2046, "0.833"),
                     ("TREFHT", "G4D16", 2048, "0.948"), ("TREFHT", "G4D43", 2046, "0.755"),
                     ("TREFHT", "G4D43", 2049, "0.931")):
    c("annual %s %s %d" % (v, arm, y), q, "f('a','%s')" % (A % (v, v, arm, y)), "annual")
c("Z3_l12 annual RMS rises every year, arm-steps", "20",
  " + ".join("(f('a','%s') > f('a','%s'))" % (A % ("Z3_l12", "Z3_l12", arm, y + 1), A % ("Z3_l12", "Z3_l12", arm, y))
             for arm in ARMS.strip("{}").split("|") for y in range(2045, 2049)), "annual")
c("TREFHT 2048 or 2049 above 2046, arms", "5",
  " + ".join("(max(f('a','%s'), f('a','%s')) > f('a','%s'))"
             % (A % ("TREFHT", "TREFHT", arm, 2048), A % ("TREFHT", "TREFHT", arm, 2049), A % ("TREFHT", "TREFHT", arm, 2046))
             for arm in ARMS.strip("{}").split("|")), "annual")
spec = {"campaign": "G4D 2048-2049 bias maps", "register": "reporting window 2048-2049, descriptive",
        "quote_source": "Claude's report 2026-11-04 and the read-out README", "base": F,
        "receipts": {"w": {"path": "figures/summary_window_bias.json", "newer_than": ["figures/summary_window_bias.csv"]},
                     "a": {"path": "summary_5yr_index.json", "newer_than": ["figures/summary_5yr.csv"]}},
        "claims": claims}
json.dump(spec, open(F + "/claims_window.json", "w"), indent=1)
print("SPEC_OK claims=%d" % len(claims))
