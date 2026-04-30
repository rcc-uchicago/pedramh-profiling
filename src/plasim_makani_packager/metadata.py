#!/usr/bin/env python3
"""metadata.py — Render Makani-consumable metadata + config from packaged HDF5.

Reads lat/lon and time-spacing from a sample training file and writes:

    {output-root}/metadata/data.json
    {output-root}/config/{config-name}.yaml       -- packager template,
                                                     {{OUTPUT_ROOT}} / {{EXP_DIR}}
                                                     substituted

By default this renders only the v10 (zgplev) baseline config — the four
other trainer-side configs (smoke/tiny/short/full) are hand-curated and
live at ``src/sfno_training/config/plasim_sim52_zgplev_*.yaml`` (per
plan §3.4). The packager-side template only differs from those in the
top-level config-name key, so ``metadata.py`` cannot semantically
generate them; it just renders the baseline.

For v9 (sigma-level zg) regeneration, check out the
``plasim-makani-packager-v9-final`` git tag and run from there.

CLI
---
metadata.py --output-root {root}
            [--variant {zgplev,astro64x128}]
            [--dataset-name plasim-sim52-astro-64x128-zgplev]
            [--config-name plasim_sim52_zgplev_baseline]
            [--exp-dir /scratch/.../runs/sim52_astro_64x128_zgplev]
            [--rsdt-method astronomical]
            [--sst-land-fill-k 271.35]
            [--train-years 3 100 --valid-years 101 120 --test-years 121 128]
            [-v]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py

from plasim_makani_packager.channels import (
    DIAGNOSTIC_CHANNELS,
    FORCING_CHANNELS,
    STATE_CHANNELS,
    TARGET_CHANNELS,
)

logger = logging.getLogger("plasim_makani_packager.metadata")

# v10 defaults (zgplev variant) — codebase is v10-only post-merge per L8.
DEFAULT_VARIANT: str = "zgplev"
DEFAULT_DATASET_NAME: str = "plasim-sim52-astro-64x128-zgplev"
DEFAULT_CONFIG_NAME: str = "plasim_sim52_zgplev_baseline"
DEFAULT_TEMPLATE_NAME: str = "plasim_64x128_zgplev.yaml"
# Top-level key inside the packager template, used as the default for
# the renamed config-name substitution. Must match the template file.
DEFAULT_TEMPLATE_TOP_KEY: str = "plasim_sim52_astro_64x128_zgplev"
DEFAULT_PACKAGER_VERSION: str = "sim52_astro_64x128_zgplev"
DEFAULT_EXP_DIR: Path = Path(
    "/scratch/11114/zhixingliu/AI-RES/runs/sim52_astro_64x128_zgplev"
)
TRAINER_PATCH_CONTRACT_URL: str = (
    "docs/plasim_makani_packager_plan.md#trainer-patch-contract"
)
V9_FREEZE_TAG: str = "plasim-makani-packager-v9-final"


def _pick_sample_file(output_root: Path) -> Path:
    for split in ("train", "valid", "test"):
        for p in sorted((output_root / split).glob("MOST.*.h5")):
            return p
    raise RuntimeError(
        f"no packaged files found under {output_root}/{{train,valid,test}}"
    )


def build_metadata(
    output_root: Path,
    *,
    dataset_name: str,
    train_years: tuple[int, int],
    valid_years: tuple[int, int],
    test_years: tuple[int, int],
    sst_land_fill_k: float,
    rsdt_method: str,
    packager_version: str,
) -> dict:
    sample_path = _pick_sample_file(output_root)
    with h5py.File(sample_path, "r") as f:
        lat = f["lat"][...].tolist()
        lon = f["lon"][...].tolist()

    return {
        "dataset_name": dataset_name,
        "h5_path": "fields_state",
        "diagnostic_h5_path": "fields_diagnostic",
        "forcing_h5_path": "forcing",
        "dims": ["time", "channel", "lat", "lon"],
        "dhours": 6,
        "coords": {
            "grid_type": "legendre-gauss",
            "lat": lat,
            "lon": lon,
            "channel": list(TARGET_CHANNELS),
            "channel_state": list(STATE_CHANNELS),
            "channel_diagnostic": list(DIAGNOSTIC_CHANNELS),
            "channel_forcing": list(FORCING_CHANNELS),
        },
        "attrs": {
            "description": (
                "PlaSim sim52 postproc 64x128, astronomical rsdt, "
                "three-dataset layout for patched Makani"
            ),
            "source_postproc_root": "/scratch/11114/zhixingliu/AI-RES/data/postproc/sim52",
            "source_boundary_root": "/scratch/11114/zhixingliu/AI-RES/data/boundary_astro/sim52",
            "rsdt_method": rsdt_method,
            "sst_land_fill_K": float(sst_land_fill_k),
            "train_years": list(train_years),
            "valid_years": list(valid_years),
            "test_years": list(test_years),
            "packager_version": packager_version,
            "requires_patched_makani": True,
            "trainer_patch_contract_url": TRAINER_PATCH_CONTRACT_URL,
        },
    }


def render_yaml(
    template_path: Path,
    *,
    output_root: Path,
    exp_dir: Path,
    config_name: str,
    default_config_name: str = DEFAULT_TEMPLATE_TOP_KEY,
) -> str:
    text = template_path.read_text()
    text = text.replace("{{OUTPUT_ROOT}}", str(output_root.resolve()))
    text = text.replace("{{EXP_DIR}}", str(exp_dir.resolve()))
    if config_name != default_config_name:
        text = text.replace(
            f"{default_config_name}:",
            f"{config_name}:",
            1,
        )
    return text


def write_outputs(
    output_root: Path,
    *,
    metadata: dict,
    rendered_yaml: str,
    config_name: str,
) -> tuple[Path, Path]:
    meta_dir = output_root / "metadata"
    cfg_dir = output_root / "config"
    meta_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    meta_path = meta_dir / "data.json"
    cfg_path = cfg_dir / f"{config_name}.yaml"

    meta_path.write_text(json.dumps(metadata, indent=2))
    cfg_path.write_text(rendered_yaml)
    return meta_path, cfg_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument(
        "--variant",
        choices=("zgplev", "astro64x128"),
        default=DEFAULT_VARIANT,
        help="Channel-list variant. 'zgplev' is the v10 default; "
        "'astro64x128' (v9) raises with a pointer to the freeze tag.",
    )
    p.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    p.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    p.add_argument(
        "--exp-dir",
        type=Path,
        default=DEFAULT_EXP_DIR,
    )
    p.add_argument("--rsdt-method", default="astronomical")
    p.add_argument("--sst-land-fill-k", type=float, default=271.35)
    p.add_argument(
        "--train-years", type=int, nargs=2, default=[3, 100], metavar=("S", "E")
    )
    p.add_argument(
        "--valid-years", type=int, nargs=2, default=[101, 120], metavar=("S", "E")
    )
    p.add_argument(
        "--test-years", type=int, nargs=2, default=[121, 128], metavar=("S", "E")
    )
    p.add_argument(
        "--packager-version",
        default=DEFAULT_PACKAGER_VERSION,
        help="Free-form tag written into metadata.attrs.packager_version",
    )
    p.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Override path to the YAML template (defaults to packaged template).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.variant == "astro64x128":
        sys.exit(
            "error: --variant astro64x128 is no longer supported by this "
            f"codebase. The codebase is v10-only after the zg sigma → zg_plev "
            f"migration (see docs/plasim_zg_plev_migration_plan.md L8). "
            f"To regenerate v9 artifacts, check out the "
            f"`{V9_FREEZE_TAG}` git tag and run from there."
        )

    template = args.template or (
        Path(__file__).resolve().parent / "templates" / DEFAULT_TEMPLATE_NAME
    )
    if not template.exists():
        sys.exit(f"error: template not found at {template}")

    metadata = build_metadata(
        args.output_root,
        dataset_name=args.dataset_name,
        train_years=tuple(args.train_years),
        valid_years=tuple(args.valid_years),
        test_years=tuple(args.test_years),
        sst_land_fill_k=args.sst_land_fill_k,
        rsdt_method=args.rsdt_method,
        packager_version=args.packager_version,
    )
    rendered = render_yaml(
        template,
        output_root=args.output_root,
        exp_dir=args.exp_dir,
        config_name=args.config_name,
    )
    meta_path, cfg_path = write_outputs(
        args.output_root,
        metadata=metadata,
        rendered_yaml=rendered,
        config_name=args.config_name,
    )
    logger.info("wrote %s", meta_path)
    logger.info("wrote %s", cfg_path)


if __name__ == "__main__":
    main()
