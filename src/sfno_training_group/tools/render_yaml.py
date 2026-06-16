"""Render production / smoke YAML by substituting placeholders from the calendar manifest.

Phase B.5 (smoke) + Phase F.E (production) of the group-code track.

Substitutes:
- ``{{DATA_DIR}}``, ``{{EXP_DIR}}`` -> CLI args.
- ``{{TRAIN_END_EXCL_<year>}}`` -> manifest entry's ``train_end_exclusive_dt`` (smoke / per-year).
- ``{{VAL_END_EXCL_<year>}}``   -> manifest entry's ``val_end_exclusive_dt_for_max_lead_K``.
- ``{{TRAIN_DATA_SET_ENTRIES}}`` (Phase F) -> a multi-line block of N entries, one per
  train year in ``--train-years``. Each entry is rendered as
  ``      - ['<year>-01-01 00:00:00', '<train_end_exclusive_dt>']``
  preserving the indent of the placeholder line. This lets production YAMLs
  reference 100 years without 100 hand-written placeholder lines.

The end-exclusive variants are required because group's ``partition_date_range``
(``utils/data_loader_multifiles.py:240``) treats ``[start, end)`` exclusively on
end. Emitting ``last_train_init_dt`` directly would silently drop the final
valid init.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("render_yaml")


def _expand_train_data_set_block(
    template: str, train_years: list[int], by_year: dict[int, dict],
) -> str:
    """Expand `{{TRAIN_DATA_SET_ENTRIES}}` into a multi-line YAML block.

    The placeholder line's leading whitespace is preserved on every emitted line,
    so the result drops cleanly into a YAML mapping value:

        train_data_sets:
          "{{DATA_DIR}}":
            {{TRAIN_DATA_SET_ENTRIES}}        <- this line is replaced

    becomes

        train_data_sets:
          "{{DATA_DIR}}":
            - ['0012-01-01 00:00:00', '0012-12-30 12:00:00']
            - ['0013-01-01 00:00:00', '0013-12-30 12:00:00']
            ... 100 entries total ...
    """
    marker = "{{TRAIN_DATA_SET_ENTRIES}}"
    lines = template.splitlines(keepends=False)
    out_lines: list[str] = []
    expanded = False
    for line in lines:
        idx = line.find(marker)
        if idx < 0:
            out_lines.append(line)
            continue
        if expanded:
            raise RuntimeError("more than one TRAIN_DATA_SET_ENTRIES placeholder found")
        prefix = line[:idx]  # leading whitespace + anything before marker
        # Allow only whitespace before marker (placeholder must own its line).
        if prefix.strip():
            raise RuntimeError(
                f"TRAIN_DATA_SET_ENTRIES must be on its own line, found: {line!r}"
            )
        for year in train_years:
            if year not in by_year:
                raise RuntimeError(f"manifest missing train year {year}")
            end_excl = by_year[year]["train_end_exclusive_dt"]
            out_lines.append(
                f"{prefix}- ['{year:04d}-01-01 00:00:00', '{end_excl}']"
            )
        expanded = True
    return "\n".join(out_lines) + ("\n" if template.endswith("\n") else "")


def render(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, val in mapping.items():
        out = out.replace(key, val)
    # Detect any unrendered placeholders.
    leftovers: list[str] = []
    i = 0
    while True:
        a = out.find("{{", i)
        if a < 0:
            break
        b = out.find("}}", a)
        if b < 0:
            break
        leftovers.append(out[a:b + 2])
        i = b + 2
    if leftovers:
        raise RuntimeError(f"Unrendered placeholders remain: {leftovers}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpl", required=True, type=Path,
                        help="Input YAML template with {{...}} placeholders.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output YAML path.")
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--exp-dir", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Path to _v10_calendar_manifest.json.")
    parser.add_argument("--train-years", type=int, nargs="+", required=True,
                        help="Years referenced by {{TRAIN_END_EXCL_<year>}}.")
    parser.add_argument("--val-years", type=int, nargs="+", required=True,
                        help="Years referenced by {{VAL_END_EXCL_<year>}}.")
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s %(message)s")

    manifest = json.loads(args.manifest.read_text())
    by_year = {y["year"]: y for y in manifest["years"]}

    mapping: dict[str, str] = {
        "{{DATA_DIR}}": args.data_dir,
        "{{EXP_DIR}}": args.exp_dir,
    }
    for year in args.train_years:
        if year not in by_year:
            raise RuntimeError(f"manifest missing year {year}")
        mapping[f"{{{{TRAIN_END_EXCL_{year}}}}}"] = by_year[year]["train_end_exclusive_dt"]
    for year in args.val_years:
        if year not in by_year:
            raise RuntimeError(f"manifest missing year {year}")
        mapping[f"{{{{VAL_END_EXCL_{year}}}}}"] = by_year[year]["val_end_exclusive_dt_for_max_lead_K"]

    template = args.tpl.read_text()
    # Phase F: expand TRAIN_DATA_SET_ENTRIES block first (if marker present),
    # then run scalar substitutions. Block-expansion handles the 100-train-year
    # case without 100 hand-written {{TRAIN_END_EXCL_NN}} lines.
    if "{{TRAIN_DATA_SET_ENTRIES}}" in template:
        template = _expand_train_data_set_block(template, args.train_years, by_year)
    out_text = render(template, mapping)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_text)
    logger.info("Rendered %s -> %s with %d substitutions", args.tpl, args.out, len(mapping))
    return 0


if __name__ == "__main__":
    sys.exit(main())
