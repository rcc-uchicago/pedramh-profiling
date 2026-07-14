#!/usr/bin/env python3
"""convert_e3sm_for_si.py — one-time staging so SI can train on the Polaris E3SM data.

SI's GetDataset (data/amip_new.py) is structurally incompatible with the raw E3SM
per-sample h5 in two ways; this converter fixes both:

  (A) npz -> normalize_{mean,std}.nc — SI's `_load_mean_std` only reads NetCDF
      (h5netcdf engine) and needs a 'level' dim for the upper-air stats. E3SM ships
      normalize_{mean,std}.npz (162 scalar keys). GUARD: normalize_std.npz has
      **16 zero-std keys** (CLDLIQ x8 at 4.71-145.04 hPa, CLDICE x4, CLOUD x4 — the
      condensate fields are identically zero in the upper stratosphere). All are set
      to 1.0 so the (x-mean)/std normalization yields 0 (constant field) not NaN/inf.
      The guard is a vectorized `where(std==0, 1.0, std)`, so it covers all 16.

  (B) h5 repack with an upper-air key rename — SI builds keys as
      f'{var}_{int(level)}.0' (e.g. 'T_850.0'); E3SM keys carry the full float
      pressure ('T_849.6612491105952'). Each sample is repacked with the renamed
      keys (int(round(level)).0), keeping the surface/diagnostic/boundary 2D fields
      and 'time'.

Coverage: all 1460 files of 2015 + 2016_0000..0003 (end-of-2015 training samples
need their t+24h targets, which fall in early 2016). ~60 GB, pure I/O.

RUN AS A PBS JOB (compute node) — it is ~20-40 min of I/O; do not run heavy
conversions on a login node. si/bench_polaris.pbs invokes this automatically if the
stage is missing. Prints CONVERT_OK on success.
"""
import argparse
import os

import h5py
import numpy as np
import xarray as xr

# Default source archive. $E3SM_ROOT (exported by polaris_env.sh, overridable with
# POLARIS_E3SM_ROOT) wins, so the advertised knob actually works; the literal is the
# fallback for a bare run outside a PBS job.
DEFAULT_ROOT = os.environ.get("E3SM_ROOT") or (
    "/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/"
    "E3SMv3_SSP245AMIP_CTL_SST0051_REST0101")
DEFAULT_STAGE = "/eagle/projects/lighthouse-uchicago/members/mehta5/si_e3sm_stage"

UA_VARS = ["T", "U", "V", "Z3", "RELHUM", "CLDICE", "CLDLIQ", "CLOUD"]
SCALAR_VARS = ["TREFHT", "PS", "PSL", "TMQ", "U10", "RHREFHT",   # surface (6)
               "FSNTOA", "FSNT", "PRECT",                        # diagnostic (3)
               "sol_in", "SST", "ICE"]                           # varying boundary (3)
KEEP_2D = SCALAR_VARS + ["TOPO", "PFTDATA_MASK"]                 # + constant boundary (2)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--stage", default=DEFAULT_STAGE)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    root, stage = args.root, args.stage

    # Discover the exact level suffixes from one file (keep exact strings for npz lookup).
    with h5py.File(f"{root}/h5/plev_data/2015_0000.h5") as f:
        suffixes = sorted((k.rpartition("_")[2] for k in f["input"] if k.startswith("T_")),
                          key=float)
    int_levels = [round(float(s)) for s in suffixes]
    assert len(set(int_levels)) == len(int_levels) == 18, int_levels
    print("config data.levels must be:", int_levels, flush=True)

    os.makedirs(f"{stage}/h5", exist_ok=True)

    # -- (A) npz -> nc (with std==0 guard) --------------------------------------
    for stat in ("mean", "std"):
        npz = np.load(f"{root}/normalize_{stat}.npz")
        ds = xr.Dataset(coords={"level": np.array(int_levels, dtype=np.int64)})
        for v in UA_VARS:
            vals = np.asarray([float(npz[f"{v}_{s}"][0]) for s in suffixes], dtype=np.float32)
            if stat == "std":
                vals = np.where(vals == 0.0, np.float32(1.0), vals)   # CLDICE strat guard
            ds[v] = xr.DataArray(vals, dims=["level"])
        for v in SCALAR_VARS:
            val = np.float32(npz[v][0])
            if stat == "std" and val == 0.0:
                val = np.float32(1.0)
            ds[v] = xr.DataArray(np.asarray([val], dtype=np.float32), dims=["scalar"])
        out = f"{stage}/normalize_{stat}.nc"
        ds.to_netcdf(out, engine="h5netcdf")
        print("wrote", out, flush=True)

    # -- (B) h5 repack with upper-air key rename --------------------------------
    def convert_file(fname):
        src, dst = f"{root}/h5/plev_data/{fname}", f"{stage}/h5/{fname}"
        if os.path.exists(dst):
            return
        with h5py.File(src, "r") as fi, h5py.File(dst + ".tmp", "w") as fo:
            g = fo.create_group("input")
            for k in fi["input"]:
                var, _, suf = k.rpartition("_")
                if var in UA_VARS:
                    g[f"{var}_{round(float(suf))}.0"] = fi["input"][k][()]
                elif k in KEEP_2D or k == "time":
                    g[k] = fi["input"][k][()]
                # else dropped (SOILWATER_10CM, TSOI_10CM, PCT_*, etc.)
        os.replace(dst + ".tmp", dst)

    files = [f"2015_{i:04d}.h5" for i in range(1460)] + [f"2016_{i:04d}.h5" for i in range(4)]
    for i, fn in enumerate(files):
        convert_file(fn)
        if i % 100 == 0:
            print(f"{i}/{len(files)}", flush=True)
    print("CONVERT_DONE", flush=True)

    if args.validate:
        with h5py.File(f"{stage}/h5/2015_0000.h5") as f:
            keys = set(f["input"].keys())
            need = {f"T_{int_levels[0]}.0", "PS", "TREFHT", "SST", "TOPO", "PFTDATA_MASK"}
            missing = need - keys
            assert not missing, f"missing keys after repack: {missing}"
        for stat in ("mean", "std"):
            with xr.open_dataset(f"{stage}/normalize_{stat}.nc", engine="h5netcdf") as d:
                assert "level" in d.dims and d.sizes["level"] == 18, d.sizes
                assert bool(np.isfinite(d["T"].values).all())
        print("validate: keys + nc stats OK", flush=True)
    print("CONVERT_OK", flush=True)


if __name__ == "__main__":
    main()
