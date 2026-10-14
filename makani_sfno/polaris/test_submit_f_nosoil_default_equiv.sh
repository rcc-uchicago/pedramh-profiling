#!/bin/bash
# Default-path equivalence test for submit_f_nosoil.sh (arm/g1-fp32 vs bac79ea9).
# qsub is replaced by an echo stub and M= points at a scratch tree, so nothing is
# submitted and no real provenance file is touched.
# PASS = LAUNCHER_DEFAULT_EQUIV_OK
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="${TMPDIR:-/tmp}/launcher_eq_$$"
mkdir -p "$T/bin" "$T/old/polaris" "$T/new/polaris" "$T/M/runs/makani_mn_scaling" \
         "$T/M/runs/makani_probe/surgical/7646690"
touch "$T/M/runs/makani_probe/surgical/7646690/best_ckpt_mp0_nosoil.tar"
printf '#!/bin/bash\necho "QSUB $*"\necho "1234567.polaris-pbs-01"\n' > "$T/bin/qsub"
chmod +x "$T/bin/qsub"
git -C "$WT" show bac79ea9:makani_sfno/polaris/submit_f_nosoil.sh > "$T/old/polaris/submit_f_nosoil.sh"
cp "$WT/polaris/submit_f_nosoil.sh" "$T/new/polaris/"
for v in old new; do
    sed -i "s|^M=/eagle/projects/lighthouse-uchicago/members/mehta5$|M=$T/M|" "$T/$v/polaris/submit_f_nosoil.sh"
    grep -q "^M=$T/M$" "$T/$v/polaris/submit_f_nosoil.sh" || { echo "ERROR M_NOT_REDIRECTED $v"; exit 2; }
done
rc=0
for args in "2 23 10:00:00 scratch" "2 43 12:00:00 warm" "1 23 05:00:00 scratch"; do
    for v in old new; do
        (cd "$T/$v" && PATH="$T/bin:$PATH" bash polaris/submit_f_nosoil.sh $args 2>&1 \
            | grep -v -e 'submitted=' > "$T/$v.out")
    done
    if diff -q "$T/old.out" "$T/new.out" > /dev/null; then echo "SAME [$args]"
    else echo "DIFF [$args]"; diff "$T/old.out" "$T/new.out" | head -6; rc=1; fi
done
[ $rc = 0 ] && echo LAUNCHER_DEFAULT_EQUIV_OK
echo "--- AMP_MODE=none SPARE=0 DEPEND=7725884 (stubbed):"
(cd "$T/new" && AMP_MODE=none SPARE=0 DEPEND=7725884 PATH="$T/bin:$PATH" \
    bash polaris/submit_f_nosoil.sh 2 23 10:00:00 scratch 2>&1) \
    | grep -e '^QSUB' -e 'amp_mode' -e 'precision' -e 'F_QUEUED' | tr ',' '\n' \
    | grep -e QSUB -e AMP_MODE -e amp_mode -e precision -e F_QUEUED -e RUN_NUM
exit $rc
