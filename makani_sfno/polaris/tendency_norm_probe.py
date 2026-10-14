#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial probe: is the `temp_diff_normalization` story actually true?

`polaris_makani_ace2_ports_handoff.md` §3 argues that our loss is nearly blind to
slowly-varying channels, that this is why `SOILWATER_10CM` drifts, and that
`temp_diff_normalization: True` is therefore the indicated fix.  **That argument has
never been measured.**  It rests on a code quote in
`docs/2026-09-10_rollout_spectra_and_drift.md` §3 and on an assumption about how the
ratio `delta_c / sigma_c` is distributed across our 101 channels.

This probe is written to FALSIFY it, not to confirm it.  Five hypotheses, each with a
threshold fixed HERE, before any number is computed, so the verdict cannot move after
the fact (same discipline as `docs/2026-09-20_k56_readout_prereg.md`, whose own
disclosed defect is the reason thresholds get written down first).

    H1  MECHANISM.   The installed makani really does implement
                     `temp_diff_normalization` as weight = sigma_c / delta_c with a
                     1e-4 clamp on the denominator.
                     FALSIFIED if the key is absent from makani/utils/loss.py or the
                     block does not have that shape.

    H2  SPREAD.      r_c = delta_c / sigma_c spans at least 1.5 orders of magnitude
                     across the 101 channels (max/min >= 30).
                     FALSIFIED if max/min < 30 -- then every channel is penalised
                     within a factor of ~30 of every other and "slow channels are
                     invisible to the loss" is simply wrong.

    H3  RESERVOIRS.  `SOILWATER_10CM` and `TSOI_10CM` sit in the BOTTOM QUARTILE of
                     r_c, i.e. they really are the slow ones.
                     FALSIFIED if either is above the 25th percentile.

    H4  MECHANISM->DRIFT.  Channels with small r_c carry more of their 336 h error as
                     systematic bias.  Spearman(r_c, bias^2/MSE @336h) <= -0.30.
                     FALSIFIED if the correlation is >= 0 or |rho| < 0.30 -- then the
                     drift is not concentrated where the loss is blind, and port B
                     loses its measured justification regardless of H2/H3.

    H5  COMPETING EXPLANATION.  The 336 h bias is instead explained by the train->test
                     climate shift (train 2015-2044, test 2048-2049, a warming
                     scenario; global_means/stds and time_means are all train-split).
                     Spearman(delta_mean_c, bias_c @336h), both in sigma units.
                     If |rho| here EXCEEDS |rho| from H4, the simplest reading is a
                     normalization/split artefact, not a loss-weighting failure, and
                     port B is demoted rather than promoted.

H1 and H2 are independent of each other; H4 is the one that matters, because H2 and H3
can both hold while the drift still lives somewhere else entirely.

⚠ PASS = `TENDENCY_PROBE_OK`, and that token means ONLY that the probe ran to
completion on real data.  **It is not a verdict.**  The verdicts are the five
`H<n> ... SUPPORTED|FALSIFIED` lines, and a probe that runs green while falsifying
three of five hypotheses is a successful probe (CLAUDE.md #14).

Run on a compute node under the SFNO venv -- it imports makani, hence torch
(CLAUDE.md #3).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# Pre-registered thresholds. Changing these after seeing the numbers forfeits the point.
H2_MIN_SPREAD = 30.0
H3_QUANTILE = 0.25
H4_MIN_ABS_RHO = 0.30
CLAMP = 1e-4
RESERVOIRS = ("SOILWATER_10CM", "TSOI_10CM")
WATCH = ("SOILWATER_10CM", "TSOI_10CM", "Z3_l17", "U10", "PRECT", "TREFHT", "PS", "TMQ")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack-root", required=True, type=Path,
                   help="e3sm_makani_alldata_production root (has train/, test/, stats/)")
    p.add_argument("--train-year", type=int, default=2044)
    p.add_argument("--test-year", type=int, default=2048)
    p.add_argument("--k56-h5", type=Path, default=None,
                   help="k56_metrics.h5; H4/H5 are skipped without it")
    p.add_argument("--windows", type=int, default=4,
                   help="evenly spaced windows in the year -- 4 samples all seasons, "
                        "so delta_c is not a single season's number")
    p.add_argument("--window-len", type=int, default=60, help="samples per window")
    p.add_argument("--channel-chunk", type=int, default=16, help="bounds peak memory")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--provenance", default="")
    return p.parse_args()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy (the venv's scipy is not guaranteed)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(np.float64)
    rb = np.argsort(np.argsort(b[ok])).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def check_h1() -> dict:
    """Read the INSTALLED makani, not the doc that quoted it three weeks ago."""
    out: dict = {"verdict": "FALSIFIED", "notes": []}
    try:
        import makani  # noqa: F401  (imports torch -- compute node only)
    except Exception as exc:                                  # pragma: no cover
        out["notes"].append(f"import makani failed: {exc!r}")
        return out
    root = Path(makani.__file__).parent
    out["makani_path"] = str(root)
    loss_py = root / "utils" / "loss.py"
    if not loss_py.exists():
        cands = sorted(str(p) for p in root.rglob("loss*.py"))
        out["notes"].append(f"utils/loss.py absent; candidates: {cands[:8]}")
        return out
    src = loss_py.read_text(errors="ignore")
    lines = src.split("\n")
    hits = [i for i, l in enumerate(lines) if "temp_diff_normalization" in l]
    if not hits:
        out["notes"].append("no 'temp_diff_normalization' in makani/utils/loss.py")
        return out
    i = hits[0]
    block = "\n".join(lines[max(0, i - 3): i + 10])
    out["block"] = block
    out["block_line"] = i + 1
    # The three things the handoff's arithmetic depends on.
    has_stds = "get_time_diff_stds" in block or "time_diff" in block
    has_clamp = "clamp" in block
    has_ratio = bool(re.search(r"/\s*time_diff_scale", block))
    out.update(has_get_time_diff_stds=has_stds, has_clamp=has_clamp,
               has_sigma_over_delta=has_ratio)
    # Which params key supplies the file -- the handoff says DO NOT assume the name.
    try:
        from makani.utils.loss import get_time_diff_stds  # type: ignore
        gsrc = inspect.getsource(get_time_diff_stds)
        out["get_time_diff_stds_src"] = gsrc
        keys = sorted(set(re.findall(r"params[\.\[]\s*['\"]?(\w+)", gsrc)))
        out["params_keys_read"] = keys
        out["normalized_by_global_stds"] = ("global_stds" in gsrc or "scale" in gsrc)
    except Exception as exc:
        out["notes"].append(f"get_time_diff_stds not importable: {exc!r}")
    if has_stds and has_clamp and has_ratio:
        out["verdict"] = "SUPPORTED"
    return out


def measure_delta(pack: Path, year: int, windows: int, wlen: int,
                  chunk: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-channel std of the one-step change, and of the field, over sampled windows.

    Diffs are taken WITHIN a window only. A diff across a window boundary would span
    months and would not be a 6-hour tendency at all -- the same class of error as a
    rollout crossing a file boundary, which sfno_ensemble refuses outright.
    """
    import h5py

    path = pack / "train" / f"{year}.h5"
    with h5py.File(path, "r") as f:
        st, dg = f["fields_state"], f["fields_diagnostic"]
        T, n_state = st.shape[0], st.shape[1]
        n_chan = n_state + dg.shape[1]
        starts = np.linspace(0, T - wlen - 1, windows).astype(int)
        sum_d2 = np.zeros(n_chan)
        sum_d = np.zeros(n_chan)
        sum_x2 = np.zeros(n_chan)
        sum_x = np.zeros(n_chan)
        n_d = n_x = 0
        for c0 in range(0, n_chan, chunk):
            c1 = min(c0 + chunk, n_chan)
            for s in starts:
                if c1 <= n_state:
                    blk = st[s:s + wlen, c0:c1].astype(np.float64)
                elif c0 >= n_state:
                    blk = dg[s:s + wlen, c0 - n_state:c1 - n_state].astype(np.float64)
                else:                       # chunk straddles the state/diag boundary
                    blk = np.concatenate(
                        [st[s:s + wlen, c0:n_state].astype(np.float64),
                         dg[s:s + wlen, 0:c1 - n_state].astype(np.float64)], axis=1)
                d = np.diff(blk, axis=0)
                sum_d[c0:c1] += d.sum(axis=(0, 2, 3))
                sum_d2[c0:c1] += (d ** 2).sum(axis=(0, 2, 3))
                sum_x[c0:c1] += blk.sum(axis=(0, 2, 3))
                sum_x2[c0:c1] += (blk ** 2).sum(axis=(0, 2, 3))
                if c0 == 0:
                    n_d += d.shape[0] * d.shape[2] * d.shape[3]
                    n_x += blk.shape[0] * blk.shape[2] * blk.shape[3]
    delta = np.sqrt(np.maximum(sum_d2 / n_d - (sum_d / n_d) ** 2, 0.0))
    sig_s = np.sqrt(np.maximum(sum_x2 / n_x - (sum_x / n_x) ** 2, 0.0))
    return delta, sig_s, n_d


def main() -> int:
    a = _args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"provenance": a.provenance, "thresholds": {
        "H2_MIN_SPREAD": H2_MIN_SPREAD, "H3_QUANTILE": H3_QUANTILE,
        "H4_MIN_ABS_RHO": H4_MIN_ABS_RHO, "CLAMP": CLAMP}}

    print("=" * 72)
    print("H1 -- does the mechanism exist in the INSTALLED makani?")
    print("=" * 72)
    h1 = check_h1()
    res["H1"] = h1
    print(f"makani: {h1.get('makani_path')}")
    if "block" in h1:
        print(f"--- makani/utils/loss.py:{h1['block_line']} ---")
        print(h1["block"])
    if "get_time_diff_stds_src" in h1:
        print("--- get_time_diff_stds ---")
        print(h1["get_time_diff_stds_src"])
        print(f"params keys read      : {h1.get('params_keys_read')}")
        print(f"normalized internally : {h1.get('normalized_by_global_stds')}")
    for n in h1.get("notes", []):
        print(f"  note: {n}")
    print(f"H1 MECHANISM ................ {h1['verdict']}")
    print()

    # --- channel names, from the converter itself (the trainer-side half is pinned) ---
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convert_e3sm_to_makani_alldata as C  # type: ignore
    chan = list(C.TARGET_CHANNELS)

    stds = np.load(a.pack_root / "stats" / "global_stds.npy").ravel()
    means = np.load(a.pack_root / "stats" / "global_means.npy").ravel()
    if len(chan) != stds.size:
        print(f"ERROR CHANNEL_COUNT_MISMATCH: {len(chan)} names vs {stds.size} stds")
        return 2

    print("=" * 72)
    print(f"H2/H3 -- measuring delta_c on {a.windows} x {a.window_len} samples "
          f"of train/{a.train_year}.h5")
    print("=" * 72)
    delta, sig_sample, n_d = measure_delta(a.pack_root, a.train_year, a.windows,
                                           a.window_len, a.channel_chunk)
    r = delta / np.maximum(stds, 1e-30)
    res["n_diff_cells"] = int(n_d)

    order = np.argsort(r)
    print(f"{'channel':18s} {'sigma_c':>12s} {'delta_c':>12s} {'r=d/s':>10s} "
          f"{'weight 1/r':>11s}")
    for i in list(order[:8]) + list(order[-5:]):
        print(f"{chan[i]:18s} {stds[i]:12.5g} {delta[i]:12.5g} {r[i]:10.5g} "
              f"{1.0 / max(r[i], 1e-30):11.4g}")
    spread = float(r.max() / max(r.min(), 1e-30))
    h2 = "SUPPORTED" if spread >= H2_MIN_SPREAD else "FALSIFIED"
    print(f"\nr_c: min {r.min():.5g}  median {np.median(r):.5g}  max {r.max():.5g}"
          f"   max/min = {spread:.4g}")
    print(f"H2 SPREAD (>= {H2_MIN_SPREAD}) ........... {h2}")

    q25 = float(np.quantile(r, H3_QUANTILE))
    h3_detail = {}
    for name in RESERVOIRS:
        j = chan.index(name) if name in chan else None
        h3_detail[name] = {"present": j is not None,
                           "r": float(r[j]) if j is not None else None,
                           "pct": float((r < r[j]).mean() * 100) if j is not None else None}
    h3 = ("SUPPORTED" if all(v["present"] and v["r"] <= q25 for v in h3_detail.values())
          else "FALSIFIED")
    for name, v in h3_detail.items():
        print(f"  {name:18s} r={v['r']}  percentile={v['pct']}")
    print(f"H3 RESERVOIRS ARE SLOW ...... {h3}   (25th pct of r = {q25:.5g})")

    n_clamp_phys = int((delta <= CLAMP).sum())
    n_clamp_norm = int((r <= CLAMP).sum())
    print(f"H1b CLAMP: channels with delta_c <= {CLAMP}: {n_clamp_phys} in physical "
          f"units, {n_clamp_norm} in sigma units")
    print("     (which one bites depends on H1's 'normalized internally' answer)")
    print("\nWatch list:")
    for name in WATCH:
        if name in chan:
            j = chan.index(name)
            print(f"  {name:18s} sigma={stds[j]:.5g} delta={delta[j]:.5g} "
                  f"r={r[j]:.5g} weight={1.0 / max(r[j], 1e-30):.4g}")
    res.update(H2={"verdict": h2, "spread": spread, "r_min": float(r.min()),
                   "r_med": float(np.median(r)), "r_max": float(r.max())},
               H3={"verdict": h3, "q25": q25, "detail": h3_detail},
               clamp={"physical": n_clamp_phys, "sigma_units": n_clamp_norm})
    print()

    # --- H4 / H5 need the K=56 curves -------------------------------------------
    bias_frac = rho4 = rho5 = None
    if a.k56_h5 and a.k56_h5.exists():
        import h5py
        with h5py.File(a.k56_h5, "r") as f:
            keys = list(f)
            if "bias_mean" not in f or "rmse_mean" not in f:
                print(f"ERROR K56_KEYS_UNEXPECTED: {keys}")
                return 3
            bias = np.asarray(f["bias_mean"][:], dtype=np.float64)      # (K, C)
            rmse = np.asarray(f["rmse_mean"][:], dtype=np.float64)
            kchan = [c.decode() if isinstance(c, bytes) else str(c)
                     for c in f["channel"][:]]
        if kchan != chan:
            print("ERROR CHANNEL_ORDER_MISMATCH: k56 channel order differs; "
                  "a positional join would correlate the wrong pairs")
            return 4
        with np.errstate(divide="ignore", invalid="ignore"):
            bias_frac = (bias[-1] ** 2) / np.maximum(rmse[-1] ** 2, 1e-300)
        print("=" * 72)
        print("H4 -- is the drift concentrated where the loss is blind?")
        print("=" * 72)
        rho4 = _spearman(r, bias_frac)
        h4 = ("SUPPORTED" if (np.isfinite(rho4) and rho4 <= -H4_MIN_ABS_RHO)
              else "FALSIFIED")
        print(f"Spearman(r_c, bias^2/MSE @336h) = {rho4:+.4f}   "
              f"(need <= {-H4_MIN_ABS_RHO})")
        print(f"H4 MECHANISM -> DRIFT ....... {h4}")

        print("\n" + "=" * 72)
        print("H5 -- or is it the train->test climate shift?")
        print("=" * 72)
        import h5py
        tp = a.pack_root / "test" / f"{a.test_year}.h5"
        with h5py.File(tp, "r") as f:
            st, dg = f["fields_state"], f["fields_diagnostic"]
            idx = np.linspace(0, st.shape[0] - 1, 64).astype(int)
            m_state = np.stack([st[i].mean(axis=(1, 2)) for i in idx]).mean(axis=0)
            m_diag = np.stack([dg[i].mean(axis=(1, 2)) for i in idx]).mean(axis=0)
        m_test = np.concatenate([m_state, m_diag])
        dmean = (m_test - means) / np.maximum(stds, 1e-30)
        bias_sig = bias[-1] / np.maximum(stds, 1e-30)
        rho5 = _spearman(dmean, bias_sig)
        print(f"Spearman(delta_mean_c, bias_c @336h), both in sigma units "
              f"= {rho5:+.4f}")
        stronger = (np.isfinite(rho5) and np.isfinite(rho4)
                    and abs(rho5) > abs(rho4))
        print(f"H5 COMPETING EXPLANATION .... "
              f"{'STRONGER THAN H4 -- demote port B' if stronger else 'weaker than H4'}")
        res.update(H4={"verdict": h4, "spearman": rho4},
                   H5={"spearman": rho5, "stronger_than_H4": bool(stronger)})
    else:
        print("H4/H5 SKIPPED -- no k56_metrics.h5 given. The mechanism->drift link "
              "is the hypothesis that matters; without it H2/H3 prove only that the "
              "channels differ in speed, which was never in doubt.")

    # --- artefacts ---------------------------------------------------------------
    import csv
    with (a.out_dir / "tendency_norm_probe.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "sigma_c", "delta_c", "r", "weight_1_over_r",
                    "sigma_sample", "bias_frac_336h"])
        for i, name in enumerate(chan):
            w.writerow([name, f"{stds[i]:.8g}", f"{delta[i]:.8g}", f"{r[i]:.8g}",
                        f"{1.0 / max(r[i], 1e-30):.8g}", f"{sig_sample[i]:.8g}",
                        "" if bias_frac is None else f"{bias_frac[i]:.8g}"])
    (a.out_dir / "tendency_norm_probe.json").write_text(
        json.dumps(res, indent=2, sort_keys=True, default=str) + "\n")

    print("\n" + "=" * 72)
    print("VERDICTS (the probe running is NOT one of them)")
    print("=" * 72)
    print(f"  H1 mechanism exists ....... {h1['verdict']}")
    print(f"  H2 r_c spread >= {H2_MIN_SPREAD} ...... {h2}")
    print(f"  H3 reservoirs are slow .... {h3}")
    print(f"  H4 drift tracks blindness . {res.get('H4', {}).get('verdict', 'SKIPPED')}")
    print(f"  H5 competing explanation .. "
          f"{'stronger' if res.get('H5', {}).get('stronger_than_H4') else 'weaker/NA'}")
    print(f"wrote {a.out_dir}/tendency_norm_probe.{{csv,json}}")
    print(f"TENDENCY_PROBE_OK channels={len(chan)} cells={n_d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
