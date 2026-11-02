#!/usr/bin/env python3
"""build_claims.py -- write claims_configs.json (an emitall spec) from the doc's own tables.

Every quoted value is taken from the markdown as printed: each table row is split into
cells, each cell's numbers are pulled out in order, and each number is paired with the
configs.json pointer that should reproduce it. A cell whose number count does not match
its field list aborts the build, so a number cannot slip into a covered table unchecked.
Covered: Tables 1a, 1b, 1c (the main training tables, each split into narrow parts that are joined
back by run, so columns are found by header name, not position) and Table 3 (protocols). Label
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


def cols(ts, must):
    """Rows (as {header: cell}) of the one table whose header contains every name in `must`."""
    hits = [(h, rows) for h, rows in ts if all(m in h for m in must)]
    if len(hits) != 1:
        raise SystemExit("ERROR TABLE_NOT_UNIQUE %s (%d)" % (must, len(hits)))
    h, rows = hits[0]
    return [dict(zip(h, r)) for r in rows]


def joined(ts, parts, key="run"):
    """Merge the parts of a split table row by row, keyed on the first column; every part must list the same rows."""
    merged = None
    for must in parts:
        rows = cols(ts, must)
        if merged is None:
            merged = rows
            continue
        if [r[key] for r in rows] != [m[key] for m in merged]:
            raise SystemExit("ERROR PART_ROWS_DIFFER %s" % must)
        for m, r in zip(merged, rows):
            m.update(r)
    return merged


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

    # ---- Table 1a (parts 1-3)
    rows = joined(T, [["run", "role", "initialised from", "checkpoint acronyms"],
                      ["run", "run tag", "PBS job id(s)"],
                      ["run", "output channels (count; soil)", "EMA (decay)"]])
    for c in rows:
        lab = label_of(c["run"])
        r = runs[lab]
        b = "/runs/" + lab + "/"
        take("Table 1a", lab, "job ids", c["PBS job id(s)"], [b + "jobs/%d" % i for i in range(len(r["jobs"]))])
        take("Table 1a", lab, "initialised from", c["initialised from"], [])
        take("Table 1a", lab, "output channels", c["output channels (count; soil)"], [b + "output_channels"])
        take("Table 1a", lab, "loss depth", c["loss depth (multistep_count / n_future)"],
             [b + "multistep_count", b + "n_future"])
        take("Table 1a", lab, "dry air", c["dry-air constraint in training"], [])
        take("Table 1a", lab, "amp_mode", c["training precision (amp_mode)"], [])
        take("Table 1a", lab, "EMA", c["EMA (decay)"], [b + "ema_decay"] if r["ema_decay"] else [])
        if ("soil yes" in c["output channels (count; soil)"]) != r["soil_outputs"]:
            raise SystemExit("ERROR SOIL_FLAG " + lab)
        for word, key in (("bf16", "bf16"), ("none (fp32", "none")):
            if c["training precision (amp_mode)"].startswith(word) and r["amp_mode"] != key:
                raise SystemExit("ERROR AMP " + lab)
        if ("trained in (`conserve_dry_air True`)" in c["dry-air constraint in training"]) != r["conserve_dry_air"]:
            raise SystemExit("ERROR DRYAIR_FLAG " + lab)
    # string claims (exact mode) for the categorical columns
    for c in rows:
        lab = label_of(c["run"])
        claims.append({"section": "Table 1a", "label": lab + " run tag", "quoted": c["run tag"].strip("`"),
                       "expr": "f('c','/runs/" + lab + "/run_tag')", "mode": "exact"})
        claims.append({"section": "Table 1a", "label": lab + " amp_mode",
                       "quoted": c["training precision (amp_mode)"].split(" ")[0],
                       "expr": "f('c','/runs/" + lab + "/amp_mode')", "mode": "exact"})

    # ---- Table 1b (parts 1-4)
    for c in joined(T, [["run", "optimizer", "β₂", "max grad norm"], ["run", "peak LR"],
                        ["run", "local batch per GPU", "global batch"], ["run", "epochs trained", "node-hours"]]):
        lab = label_of(c["run"])
        r = runs[lab]
        b = "/runs/" + lab + "/"
        take("Table 1b", lab, "optimizer", c["β₂"] + " " + c["max grad norm"],
             [b + "optimizer_beta2", b + "optimizer_max_grad_norm"])
        take("Table 1b", lab, "peak LR", c["peak LR"], [b + "lr"])
        take("Table 1b", lab, "LR schedule", c["LR schedule (type, T₀ / T_mult / T_max, min LR)"],
             [b + f for f in sched_fields(r)])
        take("Table 1b", lab, "warmup epochs", c["warmup epochs"], [b + "lr_warmup_steps"])
        take("Table 1b", lab, "local batch", c["local batch per GPU"], [b + "local_batch"])
        take("Table 1b", lab, "GPUs", c["GPUs"], [b + "gpus"])
        take("Table 1b", lab, "nodes", c["nodes"], [b + "nodes"])
        take("Table 1b", lab, "global batch", c["global batch"], [b + "global_batch"])
        take("Table 1b", lab, "h x w", c["spatial parallel (h × w)"], [b + "h_parallel_size", b + "w_parallel_size"])
        take("Table 1b", lab, "epochs trained", c["epochs trained"], [b + "epochs_trained"])
        take("Table 1b", lab, "updates", c["updates"], [b + "updates"])
        take("Table 1b", lab, "training time (h)", c["wall-clock training time (h)"], [b + "training_h"])
        take("Table 1b", lab, "node-hours", c["node-hours"], [b + "node_hours"])
        take("Table 1b", lab, "median epoch time (s)", c["median epoch time (s)"], [b + "median_epoch_time_s"])
        take("Table 1b", lab, "scale_factor", c["scale_factor"], [b + "scale_factor"])
        claims.append({"section": "Table 1b", "label": lab + " optimizer", "quoted": c["optimizer"],
                       "expr": "f('c','/runs/" + lab + "/optimizer_type')", "mode": "exact"})

    # ---- Table 1c (parts 1-2), rows keyed on (run, job)
    p1 = cols(T, ["run", "job", "epochs in this job", "max_epochs printed"])
    p2 = cols(T, ["run", "job", "training time (s)", "basis"])
    if [(r["run"], r["job"]) for r in p1] != [(r["run"], r["job"]) for r in p2]:
        raise SystemExit("ERROR TABLE_1C_PARTS_DIFFER")
    seen = {}
    for a, z in zip(p1, p2):
        c = dict(a, **z)
        lab = c["run"]
        i = seen.get(lab, 0)
        seen[lab] = i + 1
        b = "/runs/" + lab + "/segments/%d/" % i
        take("Table 1c", lab, "job (segment %d)" % i, c["job"], ["/runs/" + lab + "/jobs/%d" % i])
        take("Table 1c", lab, "epochs (segment %d)" % i, c["epochs in this job"], [b + "epoch_first", b + "epoch_last"])
        take("Table 1c", lab, "nodes (segment %d)" % i, c["nodes"], [b + "nodes"])
        take("Table 1c", lab, "max_epochs (segment %d)" % i, c["max_epochs printed"], [b + "max_epochs_printed"])
        take("Table 1c", lab, "training s (segment %d)" % i, c["training time (s)"], [b + "training_s"])
        k = c["PBS-killed elapsed (s)"]
        take("Table 1c", lab, "killed elapsed (segment %d)" % i, k,
             [b + "stdout/killed_walltime_elapsed_s"] if NUM.findall(k) else [])
        if (c["basis"] == "epoch-time sum") != (runs[lab]["segments"][i]["training_s_basis"] != "Total training time line"):
            raise SystemExit("ERROR BASIS " + lab)
    n1c = sum(len(r["segments"]) for r in runs.values())
    if sum(seen.values()) != n1c:
        raise SystemExit("ERROR TABLE_1C_ROWS %d vs %d" % (sum(seen.values()), n1c))

    # ---- Table 3 (protocols)
    P = "/protocols/"
    fy, k, lg, sc, sco = P + "five_year/", P + "k56/", P + "lagged/", P + "screen/", P + "score/"
    t3 = {r[0]: r for r in table(T, "protocol")}      # Table 3 was not split (6 columns)
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
    for rowkey, colspec in spec3.items():
        cs = t3[rowkey]
        for j, ptrs in enumerate(colspec):
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
