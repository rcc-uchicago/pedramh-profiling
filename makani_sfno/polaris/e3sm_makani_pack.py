"""Shared code for the two E3SM-to-makani converters.

convert_e3sm_to_makani.py (52 state channels) and convert_e3sm_to_makani_alldata.py
(100 state channels) read the same archive and write the same file layout. This
module holds what they share: the grid, reading one sample, writing one year, the
normalization statistics, the metadata file and the read-back check. Each converter
keeps its own channel list and its own way of computing the statistics.

Input: {e3sm_root}/h5/plev_data/{year}_{idx:04d}.h5. There is one file per 6-hour
sample and 1460 per year (365-day years, 2015-2049). Each file has a group "input"
holding one (180, 360) float32 array per channel. Row 0 is the southernmost latitude
(-89.5) and column 0 is longitude 0.5.

Output, under output_root:
    {split}/{year}.h5, where split is train, valid or test:
        fields_state       (T, C_state, 180, 360) float32: predicted, fed back as input
        fields_diagnostic  (T, 1, 180, 360) float32: predicted, used only in the loss
        forcing            (T, C_forcing, 180, 360) float32: given, never predicted
        timestamp          (T,) int64 seconds, 21600 apart, counted from the split start
        time_plasim        (T,) float64 days, kept for the PlaSim file layout
        channel_state, channel_diagnostic, channel_forcing, lat, lon
    stats/global_means.npy, global_stds.npy, time_means.npy, and forcing_* versions
    metadata/data.json

Rows are flipped so row 0 is the northernmost latitude, as in the rest of the repo
(docs/2026-06-02_eval_inference_latitude_flip.md). The trainer's
PlasimForcingDataset finds timestamps, channel names, lat and lon through h5py
dimension scales, so these are attached to every 4D dataset.
"""

import glob
import json
import os
import sys

import h5py
import numpy as np

STEP_SECONDS = 21600  # 6 hours
SAMPLES_PER_YEAR = 1460  # 365-day years, no leap days
H, W = 180, 360
LAT = np.arange(89.5, -90.0, -1.0)  # north to south, matching the flipped rows
LON = np.arange(0.5, 360.0, 1.0)
SPLITS = ("train", "valid", "test")
STD_FLOOR = 1e-12  # keeps a constant channel from dividing by zero
ZERO_VAR_STD = 1e-9  # a channel with a smaller std is reported as constant

# Written on every year file after source_root, year and converter.
FILE_ATTRS = {
    "lat_order": "descending (row 0 = +89.5)",
    "sst_units": "degC, land filled with -1.8",
}


def channel_names(specs):
    """Return the channel names from a list of (name, archive key, fill) entries."""
    return [name for name, _, _ in specs]


def add_common_args(parser):
    """Add the command-line arguments both converters take."""
    parser.add_argument("--e3sm-root", required=True,
                        help="archive root; the files are under h5/plev_data/")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-years", type=int, nargs=2, default=[2015, 2015],
                        metavar=("FIRST", "LAST"),
                        help="inclusive; `1 0` packs no train years")
    parser.add_argument("--valid-years", type=int, nargs=2, default=[2016, 2016],
                        metavar=("FIRST", "LAST"))
    parser.add_argument("--test-years", type=int, nargs=2, default=[2017, 2017],
                        metavar=("FIRST", "LAST"))
    parser.add_argument("--max-samples-per-year", type=int, default=None,
                        help="pack only the first N samples of each year (small test packs)")
    parser.add_argument("--overwrite", action="store_true",
                        help="repack years whose output file already exists")
    parser.add_argument("--validate", action="store_true",
                        help="read the finished pack back and check it")


def split_years(args):
    """Turn the --train/valid/test-years arguments into lists of years.

    E3SM is one continuous run, so a year in both train and valid (or test) would
    put training data into the evaluation. Valid and test may overlap: the small
    smoke packs use the same year for both.

    Returns:
        Dict from split name to its years, e.g. {"train": [2015, 2016], ...}.
    """
    ranges = (args.train_years, args.valid_years, args.test_years)
    years = {split: list(range(first, last + 1)) for split, (first, last) in zip(SPLITS, ranges)}
    for other in ("valid", "test"):
        shared = sorted(set(years["train"]) & set(years[other]))
        if shared:
            sys.exit(f"ERROR SPLIT_OVERLAP: train and {other} share {shared}")
    return years


def year_files(e3sm_root, year, max_samples=None):
    """Return a year's archive files in time order, or the first max_samples of them."""
    files = sorted(glob.glob(os.path.join(e3sm_root, "h5", "plev_data", f"{year}_*.h5")))
    if not files:
        raise RuntimeError(f"no input files for year {year} under {e3sm_root}/h5/plev_data")
    return files[:max_samples] if max_samples else files


def _stack(group, specs, path):
    """Read one channel group from an open archive file into a (C, H, W) array."""
    out = np.empty((len(specs), H, W), dtype=np.float32)
    for i, (_, key, fill) in enumerate(specs):
        arr = group[key][...]
        if arr.shape != (H, W):
            raise RuntimeError(f"{path}:{key} shape {arr.shape} != {(H, W)}")
        arr = arr[::-1, :]  # south-first to north-first
        if fill is not None:
            arr = np.where(np.isnan(arr), np.float32(fill), arr)
        elif np.isnan(arr).any():
            raise RuntimeError(f"{path}:{key} has NaN and no fill value")
        out[i] = arr
    return out


def read_sample(path, specs):
    """Read one archive file.

    Args:
        path: One per-sample archive file.
        specs: (state, diagnostic, forcing) channel lists. Each entry is
            (name, archive key, fill). Fill replaces NaN, for fields that exist only
            over land or only over ocean. A channel with fill None must have no NaN.

    Returns:
        A list of three float32 arrays, each (C, 180, 360) with row 0 at the north.

    Raises:
        RuntimeError: A channel has the wrong shape, or has NaN and no fill value.
    """
    with h5py.File(path, "r") as f:
        return [_stack(f["input"], group, path) for group in specs]


def write_year(out_path, files, offset_seconds, specs, attrs, on_sample=None):
    """Pack one year of archive files into one output file.

    The file is written as out_path + ".tmp" and renamed at the end, so a job killed
    mid-year never leaves a partial file under the final name.

    Args:
        out_path: Output file, {output_root}/{split}/{year}.h5.
        files: The year's archive files, in time order.
        offset_seconds: Time of the first sample, counted from the start of its
            split, so timestamps run on from one year to the next.
        specs: (state, diagnostic, forcing) channel lists; see read_sample.
        attrs: Attributes to write on the file, in order.
        on_sample: If given, called as on_sample(state, diagnostic, forcing) after
            each sample is written.

    Returns:
        The number of samples written.
    """
    n = len(files)
    names = [channel_names(group) for group in specs]
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        ascii_str = h5py.string_dtype("ascii")
        scales = {}
        for scale_name, data in (
            ("timestamp", np.int64(offset_seconds) + np.arange(n, dtype=np.int64) * STEP_SECONDS),
            ("lat", LAT.astype(np.float64)),
            ("lon", LON.astype(np.float64)),
            ("channel_state", np.array(names[0], dtype=ascii_str)),
            ("channel_diagnostic", np.array(names[1], dtype=ascii_str)),
            ("channel_forcing", np.array(names[2], dtype=ascii_str)),
        ):
            scale = f.create_dataset(scale_name, data=data)
            scale.make_scale(scale_name)
            scales[scale_name] = scale
        f.create_dataset("time_plasim",
                         data=(np.arange(n, dtype=np.float64) * (STEP_SECONDS / 86400.0)
                               + offset_seconds / 86400.0))

        datasets = []
        for dset_name, channel_scale, group_names in (
            ("fields_state", "channel_state", names[0]),
            ("fields_diagnostic", "channel_diagnostic", names[1]),
            ("forcing", "channel_forcing", names[2]),
        ):
            c = len(group_names)
            dset = f.create_dataset(dset_name, shape=(n, c, H, W), dtype="float32",
                                    chunks=(1, c, H, W))
            for dim, scale_name in enumerate(("timestamp", channel_scale, "lat", "lon")):
                dset.dims[dim].attach_scale(scales[scale_name])
            datasets.append(dset)

        for key, value in attrs.items():
            f.attrs[key] = value

        for t, path in enumerate(files):
            arrays = read_sample(path, specs)
            for dset, arr in zip(datasets, arrays):
                dset[t] = arr
            if on_sample is not None:
                on_sample(*arrays)
            if t % 200 == 0:
                print(f"  [{attrs.get('year')}] {t}/{n}", flush=True)
    os.replace(tmp, out_path)
    print(f"wrote {out_path} (T={n})", flush=True)
    return n


def pack_splits(args, years, specs, converter, extra_attrs=None,
                train_offset_samples=0, on_train_sample=None):
    """Pack every requested year whose output file does not exist yet.

    Args:
        args: Parsed arguments from add_common_args.
        years: Output of split_years.
        specs: (state, diagnostic, forcing) channel lists; see read_sample.
        converter: Path of the calling script, written to each file's attributes.
        extra_attrs: Attributes written after FILE_ATTRS.
        train_offset_samples: Samples that come before the first train year here.
            Parallel runs that each pack part of the train split use it to keep
            timestamps running on across the split.
        on_train_sample: Passed to write_year for train years only.

    Returns:
        (packed, skipped): train years written by this run, and train years skipped
        because their file already existed.
    """
    n_state = len(specs[0])
    packed, skipped = [], []
    for split, split_list in years.items():
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)
        offset = train_offset_samples * STEP_SECONDS if split == "train" else 0
        for year in split_list:
            out_path = os.path.join(args.output_root, split, f"{year}.h5")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"skip existing {out_path} (--overwrite to force)")
                with h5py.File(out_path, "r") as f:
                    shape = f["fields_state"].shape
                if shape[1] != n_state:
                    sys.exit(f"ERROR WRONG_CONTRACT: {out_path} has {shape[1]} state channels, "
                             f"expected {n_state}. This output root holds a different pack; "
                             f"use a new --output-root.")
                offset += shape[0] * STEP_SECONDS
                if split == "train":
                    skipped.append(year)
                continue
            attrs = {"source_root": args.e3sm_root, "year": year, "converter": converter,
                     **FILE_ATTRS, **(extra_attrs or {})}
            files = year_files(args.e3sm_root, year, args.max_samples_per_year)
            n = write_year(out_path, files, offset, specs, attrs,
                           on_train_sample if split == "train" else None)
            offset += n * STEP_SECONDS
            if split == "train":
                packed.append(year)
    return packed, skipped


def new_sums(n_target, n_forcing):
    """Return zeroed float64 running sums for the normalization statistics.

    Keys: n (values per channel), t_count (samples), and for targets (_t) and forcing
    (_f): sum, sum of squares, and the per-pixel sum over time.
    """
    return {
        "n": 0, "t_count": 0,
        "sum_t": np.zeros(n_target), "sumsq_t": np.zeros(n_target),
        "tsum_t": np.zeros((n_target, H, W)),
        "sum_f": np.zeros(n_forcing), "sumsq_f": np.zeros(n_forcing),
        "tsum_f": np.zeros((n_forcing, H, W)),
    }


def write_stats(stats_dir, sums, target_names, forcing_names):
    """Write the six statistics files from the running sums.

    Targets are the state channels followed by the diagnostic channel. Each gets a
    global mean and std (over time and space) and a per-pixel time mean.

    Returns:
        Names of channels whose std is at or below ZERO_VAR_STD.
    """
    os.makedirs(stats_dir, exist_ok=True)
    n, t_count = sums["n"], sums["t_count"]
    constant = []
    for prefix, names, total, total_sq, time_total in (
        ("", target_names, sums["sum_t"], sums["sumsq_t"], sums["tsum_t"]),
        ("forcing_", forcing_names, sums["sum_f"], sums["sumsq_f"], sums["tsum_f"]),
    ):
        c = len(names)
        mean = total / n
        var = np.maximum(total_sq / n - mean * mean, 0.0)
        std = np.maximum(np.sqrt(var), STD_FLOOR)
        constant += [names[i] for i in np.flatnonzero(std <= ZERO_VAR_STD)]
        np.save(os.path.join(stats_dir, f"{prefix}global_means.npy"),
                mean.astype(np.float32).reshape(1, c, 1, 1))
        np.save(os.path.join(stats_dir, f"{prefix}global_stds.npy"),
                std.astype(np.float32).reshape(1, c, 1, 1))
        np.save(os.path.join(stats_dir, f"{prefix}time_means.npy"),
                (time_total / t_count).astype(np.float32).reshape(1, c, H, W))
    print(f"wrote stats to {stats_dir} (n={n}, t_count={t_count})", flush=True)
    return constant


def write_metadata(output_root, dataset_name, description, years, e3sm_root, specs,
                   extra_attrs):
    """Write metadata/data.json, in the schema of plasim_makani_packager.metadata."""
    state, diag, forcing = (channel_names(group) for group in specs)
    meta = {
        "dataset_name": dataset_name,
        "h5_path": "fields_state",
        "diagnostic_h5_path": "fields_diagnostic",
        "forcing_h5_path": "forcing",
        "dims": ["time", "channel", "lat", "lon"],
        "dhours": 6,
        "coords": {
            "grid_type": "equiangular",
            "lat": LAT.tolist(),
            "lon": LON.tolist(),
            "channel": state + diag,
            "channel_state": state,
            "channel_diagnostic": diag,
            "channel_forcing": forcing,
        },
        "attrs": {
            "description": description,
            "source_root": e3sm_root,
            "train_years": years["train"],
            "valid_years": years["valid"],
            "test_years": years["test"],
            "requires_patched_makani": True,
            "sst_land_fill_degC": -1.8,
            **extra_attrs,
        },
    }
    meta_dir = os.path.join(output_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "data.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_dir}/data.json", flush=True)


def _require(ok, message):
    if not ok:
        sys.exit(f"ERROR VALIDATE_FAILED: {message}")


def validate(output_root, years, specs, max_samples=None):
    """Read the finished pack back and check it.

    Checks that every requested year file exists and holds a full year (or
    max_samples, for a small test pack); that the first train file has the expected
    shapes, north-to-south latitudes, 6-hour timestamps and no NaN in its first
    sample; and that all six statistics files exist with the right shapes and finite
    values. The coordinates are read through the dimension scales, the same way the
    trainer reads them.

    Args:
        output_root: The pack to check.
        years: Output of split_years.
        specs: (state, diagnostic, forcing) channel lists.
        max_samples: The --max-samples-per-year the pack was built with, or None.
    """
    n_state, n_diag, n_forcing = (len(group) for group in specs)
    n_target = n_state + n_diag
    want_t = min(max_samples, SAMPLES_PER_YEAR) if max_samples else SAMPLES_PER_YEAR
    n_files = 0
    for split, split_list in years.items():
        for year in split_list:
            path = os.path.join(output_root, split, f"{year}.h5")
            _require(os.path.exists(path), f"missing {path}")
            with h5py.File(path, "r") as f:
                t = f["fields_state"].shape[0]
            _require(t == want_t, f"{path} has {t} samples, expected {want_t}")
            n_files += 1

    train_files = sorted(glob.glob(os.path.join(output_root, "train", "*.h5")))
    _require(train_files, "no train files")
    with h5py.File(train_files[0], "r") as f:
        state = f["fields_state"]
        _require(state.shape[1:] == (n_state, H, W) and state.dtype == np.float32,
                 f"fields_state shape {state.shape} dtype {state.dtype}")
        _require(f["fields_diagnostic"].shape[1:] == (n_diag, H, W), "fields_diagnostic shape")
        _require(f["forcing"].shape[1:] == (n_forcing, H, W), "forcing shape")
        lat = state.dims[2]["lat"][...]
        lon = state.dims[3]["lon"][...]
        steps = np.diff(state.dims[0]["timestamp"][...])
        _require(lat[0] > lat[-1], "lat must run north to south")
        _require(lon.shape == (W,), "lon shape")
        _require(steps.size and np.all(steps == STEP_SECONDS), "timestamps must step 21600 s")
        for name in ("fields_state", "fields_diagnostic", "forcing"):
            _require(np.isfinite(f[name][0]).all(), f"NaN or inf in {name} sample 0")

    for prefix, c in (("", n_target), ("forcing_", n_forcing)):
        for name, shape in ((f"{prefix}global_means.npy", (1, c, 1, 1)),
                            (f"{prefix}global_stds.npy", (1, c, 1, 1)),
                            (f"{prefix}time_means.npy", (1, c, H, W))):
            arr = np.load(os.path.join(output_root, "stats", name))
            _require(arr.shape == shape and np.isfinite(arr).all(), f"{name} shape {arr.shape}")
    stds = np.load(os.path.join(output_root, "stats", "global_stds.npy")).ravel()
    n_constant = int((stds <= ZERO_VAR_STD).sum())
    print(f"validate: {n_files} year files of {want_t} samples; min target std "
          f"{stds.min():.3e}; {n_constant} constant channel(s)", flush=True)
