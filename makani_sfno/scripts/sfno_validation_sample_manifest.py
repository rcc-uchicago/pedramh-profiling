#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np


def read_yparams(upstream_tree: Path, yaml_config: Path, config: str) -> Any:
    sys.path.insert(0, str(upstream_tree))
    from utils.YParams import YParams  # noqa: PLC0415

    return YParams(str(yaml_config), config)


def datetime_class_from_calendar(upstream_tree: Path, calendar: str) -> Any:
    sys.path.insert(0, str(upstream_tree))
    from utils.data_loader_multifiles import datetime_class_from_calendar as _factory  # noqa: PLC0415

    return _factory(calendar)


def get_out_path(root_dir: str, year: int, idx: int) -> str:
    return os.path.join(root_dir, f"{year}_{idx:04}.h5")


def h5_path_for_time(params: Any, data_dir: str, dt: Any) -> str:
    datetime_class = datetime_class_from_calendar(Path(params["_upstream_tree"]), params.calendar)
    idx = int((dt - datetime_class(dt.year, 1, 1, hour=0, has_year_zero=params.has_year_zero)).total_seconds()) // 3600 // params.data_timedelta_hours
    return get_out_path(data_dir, dt.year, idx)


def simulate_distributed_sampler(dataset_len: int, world_size: int, batch_size: int) -> dict[str, Any]:
    # torch.utils.data.DistributedSampler(drop_last=False, shuffle=False) semantics.
    num_samples = int(np.ceil(dataset_len / world_size))
    total_size = num_samples * world_size
    indices = list(range(dataset_len))
    padding_size = total_size - len(indices)
    if padding_size <= len(indices):
        indices += indices[:padding_size]
    else:
        indices += (indices * int(np.ceil(padding_size / len(indices))))[:padding_size]

    by_rank = {}
    used = []
    dropped_by_rank = {}
    for rank in range(world_size):
        rank_indices = indices[rank:total_size:world_size]
        usable = (len(rank_indices) // batch_size) * batch_size
        by_rank[str(rank)] = rank_indices
        used.extend(rank_indices[:usable])
        dropped_by_rank[str(rank)] = rank_indices[usable:]

    return {
        "sampler_num_samples_per_rank": num_samples,
        "sampler_total_size_with_padding": total_size,
        "padding_indices": indices[dataset_len:],
        "rank_indices": by_rank,
        "dropped_by_rank_due_to_dataloader_drop_last": dropped_by_rank,
        "used_dataset_indices_in_metric_order": used,
        "used_unique_dataset_indices": sorted(set(used)),
        "used_count_with_repeats": len(used),
        "used_unique_count": len(set(used)),
        "candidate_indices_not_used": [i for i in range(dataset_len) if i not in set(used)],
    }


def sample_record(params: Any, data_dir: str, dataset_index: int, inference_idx: int) -> dict[str, Any]:
    datetime_class = datetime_class_from_calendar(Path(params["_upstream_tree"]), params.calendar)
    start_date = datetime_class(params.val_year_start, 1, 1, has_year_zero=params.has_year_zero)
    start_time = start_date + timedelta(hours=int(inference_idx) * params.data_timedelta_hours)
    max_lead = max(params.forecast_lead_times)
    target_1 = start_time + timedelta(hours=params.timedelta_hours)
    target_max = start_time + timedelta(hours=params.timedelta_hours * max_lead)
    return {
        "dataset_index": int(dataset_index),
        "inference_idx": int(inference_idx),
        "start_time": str(start_time),
        "init_h5": h5_path_for_time(params, data_dir, start_time),
        "varying_boundary_step0_h5": h5_path_for_time(params, data_dir, start_time),
        "target_step1_h5": h5_path_for_time(params, data_dir, target_1),
        f"target_step{max_lead}_h5": h5_path_for_time(params, data_dir, target_max),
    }


def build_manifest(params: Any, data_dir: str, label: str, world_size: int) -> dict[str, Any]:
    datetime_class = datetime_class_from_calendar(Path(params["_upstream_tree"]), params.calendar)
    year_start = int(params.val_year_start)
    year_end = int(params.val_year_end)
    start_date = datetime_class(year_start, 1, 1, has_year_zero=params.has_year_zero)
    end_date = datetime_class(year_end, 1, 1, has_year_zero=params.has_year_zero)
    dates = np.arange(0, int((end_date - start_date).total_seconds() // 3600), params.data_timedelta_hours)
    lead_time_offset = max(params.forecast_lead_times) * params.timedelta_hours // params.data_timedelta_hours
    max_inference_idx = len(dates) - lead_time_offset
    inference_idxs = np.linspace(0, max_inference_idx, num=int(params.num_inferences) + 1, dtype=int)
    ddp = simulate_distributed_sampler(len(inference_idxs), world_size, int(params.batch_size))

    first_dataset_indices = list(range(min(12, len(inference_idxs))))
    first_rank0_batch = ddp["rank_indices"]["0"][: int(params.batch_size)]
    used_head = ddp["used_dataset_indices_in_metric_order"][:16]
    used_tail = ddp["used_dataset_indices_in_metric_order"][-16:]

    return {
        "label": label,
        "data_dir": data_dir,
        "validation_year_start": year_start,
        "validation_year_end": year_end,
        "interpreted_years": list(range(year_start, year_end)),
        "interpretation": f"Python-exclusive range({year_start}, {year_end}) => years {list(range(year_start, year_end))}",
        "start_date": str(start_date),
        "end_date_exclusive": str(end_date),
        "date_grid_count_6h": int(len(dates)),
        "date_grid_first_hour": int(dates[0]),
        "date_grid_last_hour": int(dates[-1]),
        "forecast_lead_times": list(map(int, params.forecast_lead_times)),
        "lead_time_offset_steps": int(lead_time_offset),
        "max_inference_idx": int(max_inference_idx),
        "num_inferences_config": int(params.num_inferences),
        "candidate_validation_ic_count": int(len(inference_idxs)),
        "candidate_inference_idxs_head": inference_idxs[:16].astype(int).tolist(),
        "candidate_inference_idxs_tail": inference_idxs[-16:].astype(int).tolist(),
        "batch_size_per_rank": int(params.batch_size),
        "world_size": int(world_size),
        "distributed_sampler": ddp,
        "first_candidate_samples": [sample_record(params, data_dir, i, int(inference_idxs[i])) for i in first_dataset_indices],
        "rank0_first_batch_samples": [sample_record(params, data_dir, i, int(inference_idxs[i])) for i in first_rank0_batch],
        "metric_used_head_samples": [sample_record(params, data_dir, i, int(inference_idxs[i])) for i in used_head],
        "metric_used_tail_samples": [sample_record(params, data_dir, i, int(inference_idxs[i])) for i in used_tail],
        "not_used_samples": [sample_record(params, data_dir, i, int(inference_idxs[i])) for i in ddp["candidate_indices_not_used"]],
    }


def write_markdown(path: Path, march17: dict[str, Any], local: dict[str, Any]) -> None:
    same_keys = [
        "validation_year_start",
        "validation_year_end",
        "interpreted_years",
        "date_grid_count_6h",
        "date_grid_first_hour",
        "date_grid_last_hour",
        "lead_time_offset_steps",
        "max_inference_idx",
        "num_inferences_config",
        "candidate_validation_ic_count",
        "batch_size_per_rank",
        "world_size",
    ]
    lines = [
        "# SFNO 5410 Validation Sample Manifest",
        "",
        "This checks sample-set provenance only. It does not run model inference.",
        "",
        "## Summary",
        "",
        "| item | March17 | Local rerun | same |",
        "|---|---|---|---:|",
    ]
    for key in same_keys:
        lines.append(f"| `{key}` | `{march17[key]}` | `{local[key]}` | `{march17[key] == local[key]}` |")
    lines += [
        "",
        f"- March17 data root: `{march17['data_dir']}`",
        f"- Local data root: `{local['data_dir']}`",
        f"- Year interpretation: `{local['interpretation']}`",
        f"- Candidate ICs from `np.linspace(..., num=num_inferences+1)`: `{local['candidate_validation_ic_count']}`",
        f"- ICs used by 4-rank DistributedSampler + DataLoader `drop_last=True`: `{local['distributed_sampler']['used_unique_count']}` unique / `{local['distributed_sampler']['used_count_with_repeats']}` with repeats",
        f"- Candidate ICs not used in metric aggregation: `{local['distributed_sampler']['candidate_indices_not_used']}`",
        "",
        "## First Candidate Samples",
        "",
        "| dataset idx | inference idx | start time | March17 init H5 | local init H5 | target step 60 |",
        "|---:|---:|---|---|---|---|",
    ]
    for m, l in zip(march17["first_candidate_samples"], local["first_candidate_samples"]):
        lines.append(
            f"| {l['dataset_index']} | {l['inference_idx']} | `{l['start_time']}` | "
            f"`{m['init_h5']}` | `{l['init_h5']}` | `{l['target_step60_h5']}` |"
        )

    lines += [
        "",
        "## Rank 0 First Batch",
        "",
        "This is the batch that produced the first-forward diagnostic in `out.log`.",
        "",
        "| dataset idx | inference idx | start time | local init H5 |",
        "|---:|---:|---|---|",
    ]
    for sample in local["rank0_first_batch_samples"]:
        lines.append(f"| {sample['dataset_index']} | {sample['inference_idx']} | `{sample['start_time']}` | `{sample['init_h5']}` |")

    lines += [
        "",
        "## Tail Of Used Metric Samples",
        "",
        "| dataset idx | inference idx | start time | local init H5 | target step 60 |",
        "|---:|---:|---|---|---|",
    ]
    for sample in local["metric_used_tail_samples"]:
        lines.append(
            f"| {sample['dataset_index']} | {sample['inference_idx']} | `{sample['start_time']}` | "
            f"`{sample['init_h5']}` | `{sample['target_step60_h5']}` |"
        )

    lines += [
        "",
        "## Not Used Candidate",
        "",
        "| dataset idx | inference idx | start time | local init H5 | target step 60 | reason |",
        "|---:|---:|---|---|---|---|",
    ]
    for sample in local["not_used_samples"]:
        lines.append(
            f"| {sample['dataset_index']} | {sample['inference_idx']} | `{sample['start_time']}` | "
            f"`{sample['init_h5']}` | `{sample['target_step60_h5']}` | rank-local incomplete batch dropped |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-tree", required=True)
    parser.add_argument("--yaml-config", required=True)
    parser.add_argument("--config", default="SFNO")
    parser.add_argument("--local-data-dir", required=True)
    parser.add_argument("--march17-data-dir", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    upstream_tree = Path(args.upstream_tree).resolve()
    params = read_yparams(upstream_tree, Path(args.yaml_config).resolve(), args.config)
    params["_upstream_tree"] = str(upstream_tree)
    if not hasattr(params, "has_diagnostic"):
        params["has_diagnostic"] = hasattr(params, "diagnostic_variables") and len(params.diagnostic_variables) > 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    march17 = build_manifest(params, args.march17_data_dir, "march17", args.world_size)
    local = build_manifest(params, args.local_data_dir, "local", args.world_size)

    (out_dir / "march17_validation_sample_manifest.json").write_text(json.dumps(march17, indent=2, sort_keys=True) + "\n")
    (out_dir / "local_validation_sample_manifest.json").write_text(json.dumps(local, indent=2, sort_keys=True) + "\n")
    write_markdown(out_dir / "validation_sample_manifest_compare.md", march17, local)
    print(out_dir / "validation_sample_manifest_compare.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
