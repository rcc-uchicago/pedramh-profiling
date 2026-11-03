"""File-reading helpers for the map extraction jobs (need netCDF4 / h5py: compute node, sfno env)."""
import json

import numpy as np

PACK = "/eagle/projects/lighthouse-uchicago/members/mehta5/data/e3sm_makani_alldata_production"
# One yearly file of the source the pack was built from (pack metadata attrs.source_root); it carries hyam/hybm/P0
# for exactly the levels the pack kept (an `ncks -d lev,...` subset).
SIGMA_NC = ("/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/"
            "E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/sigma_data/2045_Combined_EAM_ELM.nc")


def hybrid_coeffs():
    """hyam, hybm (18,) ordered l00 (top) .. l17 (bottom) as the pack's Z3_lNN channels, and P0 (Pa).

    Levels are matched by value: the pack's level_table gives each channel's `lev` (hPa, 'exact_dataset_suffix').
    Gate: E3SM's lev = 1000 * (hyam + hybm) must hold to 1e-6 hPa for every matched level, otherwise the
    coefficients belong to other levels and Z500 would be silently wrong.
    """
    import netCDF4
    table = json.load(open(PACK + "/metadata/data.json"))["attrs"]["level_table"]
    want = np.array([float(r["exact_dataset_suffix"]) for r in sorted(table, key=lambda r: r["index"])])
    with netCDF4.Dataset(SIGMA_NC) as ds:
        lev = np.asarray(ds["lev"][:], dtype=np.float64)
        hyam = np.asarray(ds["hyam"][:], dtype=np.float64)
        hybm = np.asarray(ds["hybm"][:], dtype=np.float64)
        p0 = float(np.asarray(ds["P0"][...]))
    idx = [int(np.argmin(np.abs(lev - x))) for x in want]
    err_lev = float(np.max(np.abs(lev[idx] - want)))
    err_def = float(np.max(np.abs(1000.0 * (hyam[idx] + hybm[idx]) - want)))
    if err_lev > 1e-6 or err_def > 1e-6 or abs(p0 - 1.0e5) > 1e-6 or len(set(idx)) != len(idx):
        raise SystemExit("ERROR HYBRID_COEFFS_MISMATCH lev %.3g def %.3g P0 %s idx %s" % (err_lev, err_def, p0, idx))
    print("hybrid coeffs: %d levels matched (max |lev - table| %.1e hPa, |1000(hyam+hybm) - lev| %.1e), P0 %.0f Pa"
          % (len(idx), err_lev, err_def, p0))
    return hyam[idx], hybm[idx], p0


def land_sea_mask():
    """The pack's static land fraction (forcing channel 'lsm', first time step of valid/2045.h5): (180, 360)."""
    import h5py
    meta = json.load(open(PACK + "/metadata/data.json"))
    k = meta["coords"]["channel_forcing"].index("lsm")
    with h5py.File(PACK + "/valid/2045.h5", "r") as f:
        lsm = np.asarray(f[meta["forcing_h5_path"]][0, k], dtype=np.float64)
    lat = np.asarray(meta["coords"]["lat"], dtype=np.float64)
    lon = np.asarray(meta["coords"]["lon"], dtype=np.float64)
    return lsm, lat, lon
