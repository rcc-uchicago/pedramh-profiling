#!/usr/bin/env python3
"""build_claims.py -- write claims_configs.json (an emitall spec) from the doc's own tables.

Every quoted value is taken from the markdown as printed: each table row is split into
cells, each cell's numbers are pulled out in order, and each number is paired with the
configs.json pointer that should reproduce it. A cell whose number count does not match
its field list aborts the build, so a number cannot slip into a covered table unchecked.
Covered: Tables 1a, 1b, 1c (the main training tables) and Table 3 (protocols). Label
columns (role/run label, run tag, acronyms, git hash, script names) are names, not numbers,
and are skipped.

Stdlib only, python3.6-compatible.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "2026-11-02_makani_configurations.md")
CONF = os.path.join(HERE, "configs.json")
# numbers not glued to a letter/digit/underscore/dot on the left or a letter/digit on the right
NUM = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?![A-Za-z0-9])")


def cells(line):
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def tables(md):
    """List of (header cells, rows) in document order."""
    out, cur = [], None
    for line in md.split("\n"):
        if line.startswith("|"):
            cs = cells(line)
            if cur is None:
                cur = (cs, [])
                out.append(cur)
            elif set(line.replace("|", "").strip()) <= set("-: "):
                continue
            else:
                cur[1].append(cs)
        else:
            cur = None
    return out


def table(ts, first, second=None):
    hits = [rows for h, rows in ts if h[0] == first and (second is None or h[1].startswith(second))]
    if len(hits) != 1:
        raise SystemExit("ERROR TABLE_NOT_UNIQUE %s %s (%d)" % (first, second, len(hits)))
    return hits[0]


def label_of(cell):
    mm = re.match(r"^\*\*([^*]+)\*\*", cell)
    return mm.group(1) if mm else cell


def sched_fields(r):
    if r["scheduler"] == "CosineAnnealingWarmRestarts":
        return ["scheduler_T_0", "scheduler_T_mult", "scheduler_min_lr"]
    if r["scheduler"] == "CosineAnnealingLR":
        return ["scheduler_T_max", "scheduler_min_lr"]
    return ["scheduler_factor", "scheduler_patience"]


def main():
    md = open(DOC).read()
    conf = json.load(open(CONF))
    runs = conf["runs"]
    T = tables(md)
    claims = []

    def add(section, label, quoted, ptr):
        claims.append({"section": section, "label": label, "quoted": quoted, "expr": "f('c','" + ptr + "')"})

    def take(section, lab, colname, cell, ptrs):
        nums = NUM.findall(cell.replace("UNVERIFIED", ""))
        if len(nums) != len(ptrs):
            raise SystemExit("ERROR CELL_COUNT %s %s %s: %r has %d numbers, %d fields"
                             % (section, lab, colname, cell, len(nums), len(ptrs)))
        for q, p in zip(nums, ptrs):
            add(section, lab + " " + colname, q, p)

    # ---- Table 1a
    rows = table(T, "role in plain words")
    for cs in rows:
        lab = label_of(cs[0])
        r = runs[lab]
        b = "/runs/" + lab + "/"
        take("Table 1a", lab, "job ids", cs[2], [b + "jobs/%d" % i for i in range(len(r["jobs"]))])
        take("Table 1a", lab, "initialised from", cs[3], [])
        take("Table 1a", lab, "output channels", cs[4], [b + "output_channels"])
        take("Table 1a", lab, "loss depth", cs[5], [b + "multistep_count", b + "n_future"])
        take("Table 1a", lab, "dry air", cs[6], [])
        take("Table 1a", lab, "amp_mode", cs[7], [])
        take("Table 1a", lab, "EMA", cs[8], [b + "ema_decay"] if r["ema_decay"] else [])
        if ("soil yes" in cs[4]) != r["soil_outputs"]:
            raise SystemExit("ERROR SOIL_FLAG " + lab)
        for word, key in (("bf16", "bf16"), ("none (fp32", "none")):
            if cs[7].startswith(word) and r["amp_mode"] != key:
                raise SystemExit("ERROR AMP " + lab)
        if ("trained in (`conserve_dry_air True`)" in cs[6]) != r["conserve_dry_air"]:
            raise SystemExit("ERROR DRYAIR_FLAG " + lab)
    # string claims (exact mode) for the categorical columns
    for cs in rows:
        lab = label_of(cs[0])
        claims.append({"section": "Table 1a", "label": lab + " run tag", "quoted": cs[1].strip("`"),
                       "expr": "f('c','/runs/" + lab + "/run_tag')", "mode": "exact"})
        claims.append({"section": "Table 1a", "label": lab + " amp_mode", "quoted": cs[7].split(" ")[0],
                       "expr": "f('c','/runs/" + lab + "/amp_mode')", "mode": "exact"})

    # ---- Table 1b
    for cs in table(T, "run", "optimizer"):
        lab = label_of(cs[0])
        r = runs[lab]
        b = "/runs/" + lab + "/"
        take("Table 1b", lab, "optimizer", cs[1], [b + "optimizer_beta2", b + "optimizer_max_grad_norm"])
        take("Table 1b", lab, "peak LR", cs[2], [b + "lr"])
        take("Table 1b", lab, "LR schedule", cs[3], [b + f for f in sched_fields(r)])
        take("Table 1b", lab, "warmup epochs", cs[4], [b + "lr_warmup_steps"])
        take("Table 1b", lab, "local batch", cs[5], [b + "local_batch"])
        take("Table 1b", lab, "GPUs", cs[6], [b + "gpus"])
        take("Table 1b", lab, "nodes", cs[7], [b + "nodes"])
        take("Table 1b", lab, "global batch", cs[8], [b + "global_batch"])
        take("Table 1b", lab, "h x w", cs[9], [b + "h_parallel_size", b + "w_parallel_size"])
        take("Table 1b", lab, "epochs trained", cs[10], [b + "epochs_trained"])
        take("Table 1b", lab, "updates", cs[11], [b + "updates"])
        take("Table 1b", lab, "training time (h)", cs[12], [b + "training_h"])
        take("Table 1b", lab, "node-hours", cs[13], [b + "node_hours"])
        take("Table 1b", lab, "median epoch time (s)", cs[14], [b + "median_epoch_time_s"])
        take("Table 1b", lab, "scale_factor", cs[15], [b + "scale_factor"])
        claims.append({"section": "Table 1b", "label": lab + " optimizer", "quoted": cs[1].split(",")[0],
                       "expr": "f('c','/runs/" + lab + "/optimizer_type')", "mode": "exact"})

    # ---- Table 1c
    seen = {}
    for cs in table(T, "run", "job"):
        lab = cs[0]
        if True:
            i = seen.get(lab, 0)
            seen[lab] = i + 1
            b = "/runs/" + lab + "/segments/%d/" % i
            take("Table 1c", lab, "job (segment %d)" % i, cs[1], ["/runs/" + lab + "/jobs/%d" % i])
            take("Table 1c", lab, "epochs (segment %d)" % i, cs[2], [b + "epoch_first", b + "epoch_last"])
            take("Table 1c", lab, "nodes (segment %d)" % i, cs[3], [b + "nodes"])
            take("Table 1c", lab, "max_epochs (segment %d)" % i, cs[4], [b + "max_epochs_printed"])
            take("Table 1c", lab, "training s (segment %d)" % i, cs[5], [b + "training_s"])
            take("Table 1c", lab, "killed elapsed (segment %d)" % i, cs[7],
                 [b + "stdout/killed_walltime_elapsed_s"] if NUM.findall(cs[7]) else [])
            if (cs[6] == "epoch-time sum") != (runs[lab]["segments"][i]["training_s_basis"] != "Total training time line"):
                raise SystemExit("ERROR BASIS " + lab)
    n1c = sum(len(r["segments"]) for r in runs.values())
    if sum(seen.values()) != n1c:
        raise SystemExit("ERROR TABLE_1C_ROWS %d vs %d" % (sum(seen.values()), n1c))

    # ---- Table 3 (protocols)
    P = "/protocols/"
    fy, k, lg, sc, sco = P + "five_year/", P + "k56/", P + "lagged/", P + "screen/", P + "score/"
    t3 = {r[0]: r for r in table(T, "protocol")}
    spec3 = {
        "**5-yr protocol**": [
            [fy + "N_MEMBERS"],
            [fy + "start_year", fy + "START_FRAME0", fy + "MEMBER_STRIDE", fy + "start_year"],
            [fy + "N_STEPS0", fy + "MEMBER_STRIDE", fy + "N_STEPS0", fy + "last_member_steps",
             fy + "end_date/0", fy + "end_date/1", fy + "end_date/2"],
            [fy + "selection_window/0", fy + "selection_window/1", fy + "reporting_window_sealed/0",
             fy + "reporting_window_sealed/1", fy + "reporting_set_R_frames/0", fy + "reporting_set_R_frames/1",
             fy + "member_pairs", sco + "n_groups", sco + "n_surface_groups", sco + "n_level_families",
             sco + "n_bands", fy + "gate2_neg_tmq_area_max_pct", fy + "gate3_abs_dps_max_hpa",
             fy + "gate4_max_drifting_groups", fy + "gate4_drift_threshold_sd"]],
        "**K=56 sweep**": [
            [k + "N_IC", k + "n_test_files", P + "data_split/test/first", P + "data_split/test/last", k + "n_ics"],
            [],
            [k + "K", k + "horizon_h", k + "horizon_days"],
            []],
        "**Lagged ensemble**": [
            [lg + "N_TARGETS", lg + "members_per_target", lg + "STRIDE"],
            [k + "FIRST_START", "/k56_arms/G4D16_lagged_K56_d4/test_file_year"],
            [lg + "K", "/k56_arms/G4D16_lagged_K56_d4/depth_min", "/k56_arms/G4D16_lagged_K56_d4/depth_max"],
            ["/k56_arms/G4D16_lagged_K56_d4/rollouts"]],
        "**1-yr screen**": [
            [],
            [sc + "START_YEAR", sc + "START_FRAME", sc + "second_start_frame"],
            [sc + "N_LEADS"],
            ["/counts/screen_jobs", "/counts/screen_labels"]],
    }
    for rowkey, cols in spec3.items():
        cs = t3[rowkey]
        for j, ptrs in enumerate(cols):
            take("Table 3", rowkey.strip("*"), "col%d" % (j + 1), cs[j + 1], ptrs)

    spec = {
        "campaign": "makani SFNO configurations 2026-11-02",
        "register": "configuration tables (Tables 1a, 1b, 1c, 3) of 2026-11-02_makani_configurations.md",
        "quote_source": "2026-11-02_makani_configurations.md, as printed (parsed by build_claims.py)",
        "base": HERE,
        "receipts": {"c": {"path": "configs.json"}},
        "claims": claims,
    }
    with open(os.path.join(HERE, "claims_configs.json"), "w") as f:
        json.dump(spec, f, indent=1, ensure_ascii=False)
    print("CLAIMS_BUILT n=%d" % len(claims))


if __name__ == "__main__":
    main()
