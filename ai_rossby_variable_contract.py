#!/usr/bin/env python
"""The PanguWeather-parity variable contract, and the checks that enforce it.

One module, three jobs:

* ``PLANNED`` — the variable contract the ai-rossby PanguPlasim run is built to.
  This is the decision record: everything downstream (model config, dataset
  config, H5->Zarr converter, store attrs) is derived from it, never restated.
* ``--check-ground-truth`` — assert ``PLANNED`` equals the contract of jesswan's
  actually-trained PanguWeather E3SM run, parsed out of its YAML. This is the
  Step-0 gate (see ``ai_rossby_panguweather_variable_parity.md``).
* ``--check-artifacts`` — assert the produced ai-rossby model config, dataset
  config and Zarr store attrs equal ``PLANNED``. This is the runtime preflight;
  channel order is a silent failure mode (tensors stack in store-attrs order
  while fills and the loss build from the model-config lists, and ``torch.cat``
  cannot catch a mismatch), so order is compared, not just membership.

Exit code 0 = every group PASS. Non-zero + a greppable ``ERROR`` line otherwise.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# --- The contract ---------------------------------------------------------
#
# 108 fields = 5 upper-air x 18 levels (90) + 18 surface-type
#            = 90 + surface 6 + land 2 + ocean 0 + diagnostic 3
#              + constant_boundary 4 + varying_boundary 3.
#
# Order is load-bearing in two places and matches PanguWeather's YAML exactly:
#   * within each group, so store attrs and model-config lists agree channel
#     for channel;
#   * across surface/land/ocean, because PanguPlasimLegacy slices its surface
#     tensor as [surface | land | ocean] (pangu_plasim_legacy.py:673-678) -- so
#     the store's folded `surface_variables` attr must be exactly that order.
PLANNED = {
    "upper_air_variables": ["T", "U", "V", "Z3", "RELHUM"],
    "surface_variables": ["TREFHT", "U10", "RHREFHT", "PS", "PSL", "TMQ"],
    "diagnostic_variables": ["FSNTOA", "FSNT", "PRECT"],
    "land_variables": ["SOILWATER_10CM", "TSOI_10CM"],
    "ocean_variables": [],
    "constant_boundary_variables": ["PCT_GLACIER", "PFTDATA_MASK", "PCT_NATVEG", "TOPO"],
    "varying_boundary_variables": ["SST", "ICE", "sol_in"],
    # The 18 E3SM hybrid levels, in hPa, at the archive's full precision. These
    # are the values embedded in the H5 keys (e.g. `T_998.4964394917621`), which
    # PanguWeather selects on via `sigma_levels` (`use_sigma_levels: True`).
    "levels": [
        4.714998332947841,
        10.655023096474308,
        19.235455601758737,
        28.79458853709195,
        50.11779996521295,
        69.59908688413749,
        96.46377266572703,
        145.04282239200347,
        200.99889546355382,
        256.72368590525895,
        302.21364012188303,
        385.999023919911,
        492.46857402252755,
        608.6437744215842,
        713.7046383204334,
        849.6612491105952,
        925.5197481473349,
        998.4964394917621,
    ],
    # Parity fills: reuse PanguWeather's `mask_fill` verbatim. SST and TSOI_10CM
    # are filled in-distribution at 270; every other masked field fills at 0.
    # (A masked-stats recompute with SST=-1.8 is the documented fast-follow.)
    "fills": {
        "SOILWATER_10CM": 0.0,
        "TSOI_10CM": 270.0,
        "PCT_GLACIER": 0.0,
        "PFTDATA_MASK": 0.0,
        "PCT_NATVEG": 0.0,
        "TOPO": 0.0,
        "SST": 270.0,
        "ICE": 0.0,
    },
}

VAR_GROUPS = [
    "upper_air_variables",
    "surface_variables",
    "diagnostic_variables",
    "land_variables",
    "ocean_variables",
    "constant_boundary_variables",
    "varying_boundary_variables",
]

# The store has no land/ocean groups of its own: land folds into the surface
# group, last, in the order the model slices it.
STORE_SURFACE = (
    PLANNED["surface_variables"] + PLANNED["land_variables"] + PLANNED["ocean_variables"]
)

GROUND_TRUTH_YAML = REPO / "PanguWeather/v2.0/config/E3SM_SFNO_H5_STAMPEDE_jsw.yaml"


def n_channels(contract: dict) -> int:
    """Total field count implied by a contract."""
    return len(contract["upper_air_variables"]) * len(contract["levels"]) + sum(
        len(contract[g]) for g in VAR_GROUPS if g != "upper_air_variables"
    )


# --- Parsing --------------------------------------------------------------
#
# PanguWeather's YAML uses merge keys and repeated anchors that PyYAML's safe
# loader mangles, and the ai-rossby model YAML uses OmegaConf interpolation.
# Both are read with targeted regexes instead: this module must run on a login
# node with nothing installed but the standard library.


def _yaml_list(text: str, key: str) -> list:
    m = re.search(rf"^\s*{re.escape(key)}:\s*(\[.*?\])", text, re.M | re.S)
    if not m:
        raise KeyError(f"{key} not found")
    return ast.literal_eval(m.group(1))


def _yaml_scalar_list(text: str, key: str) -> list:
    """A YAML list that may span lines without brackets closing on one line."""
    return _yaml_list(text, key)


def parse_panguweather(path: Path) -> dict:
    """The ground-truth contract, read out of a PanguWeather E3SM config."""
    text = path.read_text()
    out = {g: _yaml_list(text, g) for g in VAR_GROUPS}
    # `use_sigma_levels: True` means the *sigma_levels* list is the level
    # identity that reaches the data loader; `levels` is the nominal hPa label.
    if re.search(r"^\s*use_sigma_levels:\s*True", text, re.M):
        out["levels"] = _yaml_scalar_list(text, "sigma_levels")
    else:
        out["levels"] = _yaml_scalar_list(text, "levels")
    m = re.search(r"^\s*mask_fill:\s*\{(.*?)\}", text, re.M | re.S)
    if not m:
        raise KeyError("mask_fill not found")
    out["fills"] = {
        k: float(v)
        for k, v in re.findall(r"'([^']+)':\s*([-\d.eE]+)", m.group(1))
    }
    return out


def parse_ai_rossby_model(path: Path) -> dict:
    text = path.read_text()
    out = {g: _yaml_list(text, g) for g in VAR_GROUPS}
    out["levels"] = _yaml_scalar_list(text, "levels")
    return out


def parse_ai_rossby_dataset(path: Path) -> dict:
    """`nan_fill_values` + `nan_fill_default`, expanded to a full fill map.

    The dataset config only names the non-default fills, so the default is
    expanded across the rest. "The rest" is taken to be exactly the fields
    PanguWeather's `mask_fill` covers — i.e. the ones that actually carry a
    mask. It is deliberately NOT every boundary field: `sol_in` has 0% NaN, so
    a fill for it would never fire, and demanding one would fail a correct
    config.
    """
    text = path.read_text()
    m = re.search(r"^\s*nan_fill_default:\s*([-\d.eE]+)", text, re.M)
    if not m:
        raise KeyError("nan_fill_default not found")
    default = float(m.group(1))
    # `nan_fill_values` may be flow style (`{SST: 270.0}`) or block style
    # (indented `SST: 270.0` lines). Accept both — the ai-rossby configs use one
    # and ours uses the other, and a parser that silently sees an empty map here
    # would report "all fills at default" as a PASS-shaped answer.
    explicit: dict[str, float] = {}
    m = re.search(r"^\s*nan_fill_values:\s*\{(.*?)\}", text, re.M | re.S)
    if m:
        explicit = {
            k: float(v) for k, v in re.findall(r"([A-Za-z_0-9]+)\s*:\s*([-\d.eE]+)", m.group(1))
        }
    else:
        m = re.search(r"^(\s*)nan_fill_values:\s*$", text, re.M)
        if m:
            indent = len(m.group(1))
            for line in text[m.end():].splitlines()[1:]:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if len(line) - len(line.lstrip()) <= indent:
                    break
                k, _, v = line.strip().partition(":")
                explicit[k.strip()] = float(v.split("#")[0].strip())
    maskable = list(PLANNED["fills"])
    fills = {v: explicit.get(v, default) for v in maskable}
    # An explicit fill for a field PanguWeather does not mask is a real
    # divergence, not a harmless extra — surface it rather than dropping it.
    for k, v in explicit.items():
        if k not in fills:
            fills[k] = v
    return {"fills": fills}


def parse_converter_channels(path: Path) -> dict:
    """`PANGU_E3SM_CHANNELS` out of the H5->Zarr converter, without importing it.

    The converter pulls in h5py/xarray, which are absent from the login-node
    environment; the literal is extracted from the AST instead.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PANGU_E3SM_CHANNELS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError("PANGU_E3SM_CHANNELS not found")


# --- Comparison -----------------------------------------------------------


def _cmp(label: str, expected, actual, results: list) -> None:
    ok = expected == actual
    results.append((label, ok, expected, actual))


def _cmp_levels(label: str, expected: list, actual: list, results: list) -> None:
    """Compare level lists after a float32 cast.

    The Zarr `pressure_level` coord is stored float32, so a store (or a config
    written to match one) differs from the archive's float64 literals at ~1e-7
    relative. That is a storage artifact, not a different level. Both consumers
    already tolerate it (train.py's subset guard only fires for a strict subset;
    the normalizer matches nearest-value at atol=1e-3), so the check does too —
    while still catching a genuinely different level, a reordering, or a
    different count.
    """
    def f32(xs):
        return [struct.unpack("f", struct.pack("f", float(x)))[0] for x in xs]

    ok = len(expected) == len(actual) and f32(expected) == f32(actual)
    results.append((label, ok, expected, actual))


def report(results: list, title: str) -> int:
    print(f"=== {title} ===")
    failed = 0
    for label, ok, expected, actual in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failed += 1
            print(f"       expected: {expected}")
            print(f"       actual:   {actual}")
    if failed:
        print(f"ERROR VARIABLE_PARITY_FAIL {failed}/{len(results)} checks failed")
        return 1
    print(f"VARIABLE_PARITY_OK {len(results)}/{len(results)} checks passed")
    return 0


def check_ground_truth(path: Path) -> int:
    gt = parse_panguweather(path)
    results: list = []
    for g in VAR_GROUPS:
        _cmp(f"{g} (names + order)", gt[g], PLANNED[g], results)
    _cmp_levels("levels (values + order)", gt["levels"], PLANNED["levels"], results)
    _cmp("mask fills", gt["fills"], PLANNED["fills"], results)
    _cmp("total field count", n_channels(gt), n_channels(PLANNED), results)
    return report(results, f"PLANNED vs ground truth ({path.name})")


def check_artifacts(model: Path, dataset: Path, converter: Path, store: Path | None) -> int:
    results: list = []

    m = parse_ai_rossby_model(model)
    for g in VAR_GROUPS:
        _cmp(f"model.{g}", PLANNED[g], m[g], results)
    _cmp_levels("model.levels", PLANNED["levels"], m["levels"], results)

    d = parse_ai_rossby_dataset(dataset)
    _cmp("dataset fills", PLANNED["fills"], d["fills"], results)

    c = parse_converter_channels(converter)
    _cmp("converter.surface_variables (folded)", STORE_SURFACE, c["surface_variables"], results)
    _cmp(
        "converter.pressure_upper_air_variables",
        PLANNED["upper_air_variables"],
        c["pressure_upper_air_variables"],
        results,
    )
    for g in ("diagnostic_variables", "constant_boundary_variables", "varying_boundary_variables"):
        _cmp(f"converter.{g}", PLANNED[g], c[g], results)
    _cmp("converter.sigma_upper_air_variables", [], c["sigma_upper_air_variables"], results)
    _cmp_levels("converter.pressure_levels", PLANNED["levels"], c["pressure_levels"], results)

    if store is not None:
        results.extend(_check_store(store))

    return report(results, "PLANNED vs produced artifacts")


def _check_store(store: Path) -> list:
    """Compare a produced Zarr store's attrs against PLANNED.

    Read straight off ``zarr.json`` / ``.zattrs`` so the check needs no zarr
    install -- it has to be runnable from the login node too.
    """
    import numpy as np  # noqa: F401  (only needed for the level cast below)

    results: list = []
    meta = store / "zarr.json"
    if meta.exists():
        attrs = json.loads(meta.read_text()).get("attributes", {})
    else:
        attrs = json.loads((store / ".zattrs").read_text())

    _cmp("store.surface_variables (folded)", STORE_SURFACE, list(attrs.get("surface_variables", [])), results)
    _cmp(
        "store.pressure_upper_air_variables",
        PLANNED["upper_air_variables"],
        list(attrs.get("pressure_upper_air_variables", [])),
        results,
    )
    for g in ("diagnostic_variables", "constant_boundary_variables", "varying_boundary_variables"):
        _cmp(f"store.{g}", PLANNED[g], list(attrs.get(g, [])), results)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check-ground-truth", action="store_true")
    p.add_argument("--check-artifacts", action="store_true")
    p.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_YAML)
    p.add_argument("--model-config", type=Path)
    p.add_argument("--dataset-config", type=Path)
    p.add_argument("--converter", type=Path)
    p.add_argument("--store", type=Path, default=None, help="Optional produced Zarr store.")
    p.add_argument("--dump", action="store_true", help="Print PLANNED as JSON and exit.")
    a = p.parse_args(argv)

    if a.dump:
        print(json.dumps({**PLANNED, "store_surface_variables": STORE_SURFACE,
                          "n_channels": n_channels(PLANNED)}, indent=2))
        return 0

    rc = 0
    if a.check_ground_truth:
        rc |= check_ground_truth(a.ground_truth)
    if a.check_artifacts:
        missing = [n for n, v in (("--model-config", a.model_config),
                                  ("--dataset-config", a.dataset_config),
                                  ("--converter", a.converter)) if v is None]
        if missing:
            print(f"ERROR MISSING_ARGS {' '.join(missing)}")
            return 2
        rc |= check_artifacts(a.model_config, a.dataset_config, a.converter, a.store)
    if not (a.check_ground_truth or a.check_artifacts):
        p.error("pick --check-ground-truth and/or --check-artifacts (or --dump)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
