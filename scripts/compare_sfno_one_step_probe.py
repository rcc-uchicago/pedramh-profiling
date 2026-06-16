#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def short(value: Any, n: int = 18) -> str:
    text = str(value)
    return text if len(text) <= n else text[:n] + "..."


def indexed_by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in items if "name" in item}


def first_activation_mismatch(a: dict[str, Any], b: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None, dict[str, Any] | None]:
    acts_a = a.get("activations", [])
    acts_b = b.get("activations", [])
    n = min(len(acts_a), len(acts_b))
    for idx in range(n):
        left = acts_a[idx]
        right = acts_b[idx]
        if left.get("name") != right.get("name") or left.get("output") != right.get("output"):
            return idx, left, right
    if len(acts_a) != len(acts_b):
        return n, acts_a[n] if n < len(acts_a) else None, acts_b[n] if n < len(acts_b) else None
    return None, None, None


def tensor_hash(summary: Any) -> str | None:
    if isinstance(summary, dict):
        return summary.get("sha256")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two SFNO one-step probe JSON dumps.")
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    left = read_json(Path(args.left))
    right = read_json(Path(args.right))
    lines: list[str] = [
        "# SFNO One-Step Probe Comparison",
        "",
        f"Left: `{args.left_label}` `{Path(args.left)}`",
        f"Right: `{args.right_label}` `{Path(args.right)}`",
        "",
        "## Top-Level Verdicts",
        "",
        f"- Input aggregate identical: `{left.get('input_aggregate_sha256') == right.get('input_aggregate_sha256')}`",
        f"- Parameter aggregate identical: `{left.get('parameter_aggregate_sha256') == right.get('parameter_aggregate_sha256')}`",
        f"- Surface output identical: `{tensor_hash(left.get('outputs', {}).get('surface')) == tensor_hash(right.get('outputs', {}).get('surface'))}`",
        f"- Upper-air output identical: `{tensor_hash(left.get('outputs', {}).get('upper_air')) == tensor_hash(right.get('outputs', {}).get('upper_air'))}`",
        "",
    ]

    lines += [
        "## Inputs",
        "",
        "| tensor | identical | left sha | right sha | left shape | right shape |",
        "|---|---:|---|---|---|---|",
    ]
    input_names = sorted(set(left.get("inputs", {})) | set(right.get("inputs", {})))
    for name in input_names:
        l_item = left.get("inputs", {}).get(name, {})
        r_item = right.get("inputs", {}).get(name, {})
        l_sha = tensor_hash(l_item)
        r_sha = tensor_hash(r_item)
        lines.append(
            f"| `{name}` | `{l_sha == r_sha}` | `{short(l_sha)}` | `{short(r_sha)}` | "
            f"`{l_item.get('shape')}` | `{r_item.get('shape')}` |"
        )
    lines.append("")

    l_params = indexed_by_name(left.get("parameters", []))
    r_params = indexed_by_name(right.get("parameters", []))
    param_names = sorted(set(l_params) | set(r_params))
    param_diffs = [
        name for name in param_names
        if l_params.get(name, {}).get("sha256") != r_params.get(name, {}).get("sha256")
        or l_params.get(name, {}).get("shape") != r_params.get(name, {}).get("shape")
    ]
    lines += [
        "## Parameters",
        "",
        f"- Parameter names left/right: `{len(l_params)}` / `{len(r_params)}`",
        f"- Differing or missing parameter entries: `{len(param_diffs)}`",
        "",
        "| first differing parameter | left sha | right sha | left shape | right shape |",
        "|---|---|---|---|---|",
    ]
    for name in param_diffs[:40]:
        l_item = l_params.get(name, {})
        r_item = r_params.get(name, {})
        lines.append(
            f"| `{name}` | `{short(l_item.get('sha256'))}` | `{short(r_item.get('sha256'))}` | "
            f"`{l_item.get('shape')}` | `{r_item.get('shape')}` |"
        )
    if not param_diffs:
        lines.append("| none | | | | |")
    lines.append("")

    mismatch_idx, act_l, act_r = first_activation_mismatch(left, right)
    lines += [
        "## Activations",
        "",
        f"- Activation count left/right: `{len(left.get('activations', []))}` / `{len(right.get('activations', []))}`",
        f"- First activation mismatch index: `{mismatch_idx}`",
    ]
    if mismatch_idx is not None:
        lines += [
            f"- Left activation: `{None if act_l is None else act_l.get('name')}`",
            f"- Right activation: `{None if act_r is None else act_r.get('name')}`",
            "",
            "```json",
            json.dumps({"left": act_l, "right": act_r}, indent=2, sort_keys=True)[:6000],
            "```",
            "",
        ]
    else:
        lines.append("")

    lines += [
        "## Imports",
        "",
        "| module | same file | same version | left file | right file |",
        "|---|---:|---:|---|---|",
    ]
    l_imports = {item["module"]: item for item in left.get("imports", [])}
    r_imports = {item["module"]: item for item in right.get("imports", [])}
    for name in sorted(set(l_imports) | set(r_imports)):
        l_item = l_imports.get(name, {})
        r_item = r_imports.get(name, {})
        lines.append(
            f"| `{name}` | `{l_item.get('file') == r_item.get('file')}` | "
            f"`{l_item.get('version') == r_item.get('version')}` | "
            f"`{l_item.get('file')}` | `{r_item.get('file')}` |"
        )
    lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
