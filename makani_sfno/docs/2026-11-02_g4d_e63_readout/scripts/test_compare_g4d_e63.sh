#!/bin/bash
# Self-test of compare_g4d_e63.py on data that already exists, before any new job.
#   K=56 case 1: every new arm := G4D16. Length rows must be exactly 1.000; the G4D16/A243 and A16/A243 rows must
#                reproduce depth_vs_length_2026-10-21/results.txt; gap closed 0.00 (NO); rule 1 FLAT.
#   K=56 case 2: G4D56 := A243. G4D56/A243 must be 1.000; G4D56/G4D16 at 6 h = 1/1.385 = 0.722 (odd n: the median
#                of reciprocals is the reciprocal of the median); gap closed 1.00 (CLOSED); rule 1 HELPS.
#   5-yr stand-in: g4d63 := g4d with G4D16->G4D36, G4D19->G4D56, G4D23->G4D43, G4D15->G4D63. Then
#                G4D36-G4D16 = G4D43-G4D23 = +0.000 +- 0.000, G4D56-G4D16 = -(G4D16-G4D19) = -0.022 +- 0.016
#                (g4d_dryair_twin results), and the winner stays G4D16 or its copy G4D36.
# PASS = TEST_G4D_E63_OK
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
E=/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval
SD=${E}/protocol_scores
G="${E}/G4D16_K56/scores"; A="${E}/prod1n_b32_sgdr_K56/scores"
T="$(mktemp -d "${CLAUDE_JOB_DIR:-/tmp}/tmp/g4de63test.XXXX")"
mkdir "${T}/sd"
for f in "${SD}"/*; do case "$(basename "$f")" in g4d63_*) ;; *) ln -s "$f" "${T}/sd/";; esac; done
for s in channels.csv census.csv budget.txt scores.csv; do
    sed -e 's/G4D16/G4D36/g' -e 's/G4D19/G4D56/g' -e 's/G4D23/G4D43/g' -e 's/G4D15/G4D63/g' "${SD}/g4d_${s}" \
        > "${T}/sd/g4d63_${s}"
done
o1="$(CMP_G4D36="${G}" CMP_G4D56="${G}" CMP_G4D43="${G}" CMP_G4D63="${G}" NF_SD="${T}/sd" CMP_OUT="${T}/r1.txt" \
      python3 "${D}/compare_g4d_e63.py")"
o2="$(CMP_G4D36="${G}" CMP_G4D56="${A}" CMP_G4D43="${G}" CMP_G4D63="${G}" NF_SD="${T}/sd" CMP_OUT="${T}/r2.txt" \
      python3 "${D}/compare_g4d_e63.py")"
rm -rf "${T}"
row() { printf '%s\n' "$1" | sed -n "s/^  $2 *//p" | head -1 | tr -s ' '; }
fail=0
chk() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }
ones="1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000"
chk "case1 G4D36/G4D16 == 1" "$(row "${o1}" "G4D36\/G4D16")" "${ones}"
chk "case1 G4D63/G4D16 == 1" "$(row "${o1}" "G4D63\/G4D16")" "${ones}"
chk "case1 gap row = 10-21" "$(row "${o1}" "G4D16\/A243")" "1.385 1.223 1.082 1.023 0.977 0.935 0.919 0.918 0.939"
chk "case1 length row = 10-21" "$(row "${o1}" "A16\/A243")" "1.142 1.145 1.156 1.170 1.177 1.177 1.159 1.140 1.049"
chk "case1 rule 1 6 h" "$(printf '%s\n' "${o1}" | sed -n 's/^  rule 1,  6 h:.*-> \([A-Z]*\) .*/\1/p')" "FLAT"
chk "case1 rule 2 6 h" "$(printf '%s\n' "${o1}" | sed -n 's/^  rule 2,  6 h: gap closed by G4D56 \(.*\)$/\1/p')" "0.00 -> NO"
chk "case2 G4D56/A243 == 1" "$(row "${o2}" "G4D56\/A243")" "${ones}"
chk "case2 G4D56/G4D16 6 h" "$(row "${o2}" "G4D56\/G4D16" | cut -d' ' -f1)" "0.722"
chk "case2 rule 1 6 h" "$(printf '%s\n' "${o2}" | sed -n 's/^  rule 1,  6 h:.*-> \([A-Z]*\) .*/\1/p')" "HELPS"
chk "case2 rule 2 6 h" "$(printf '%s\n' "${o2}" | sed -n 's/^  rule 2,  6 h: gap closed by G4D56 \(.*\)$/\1/p')" "1.00 -> CLOSED"
pd() { printf '%s\n' "${o1}" | sed -n "s/^    $1 = \([-+0-9.]* +- [0-9.]*\).*/\1/p"; }
chk "5yr G4D36-G4D16" "$(pd "G4D36 - G4D16")" "+0.000 +- 0.000"
chk "5yr G4D43-G4D23" "$(pd "G4D43 - G4D23")" "+0.000 +- 0.000"
chk "5yr G4D56-G4D16" "$(pd "G4D56 - G4D16")" "-0.022 +- 0.016"
w=$(printf '%s\n' "${o1}" | sed -n 's/^winner: \([A-Z0-9]*\) .*/\1/p')
case "${w}" in G4D16|G4D36) echo "ok   5yr winner ${w}";; *) echo "FAIL 5yr winner [${w}]"; fail=1;; esac
printf '%s\n' "${o1}" | tail -1
[ "${fail}" = 0 ] && echo "TEST_G4D_E63_OK" || { echo "ERROR TEST_G4D_E63"; exit 1; }
