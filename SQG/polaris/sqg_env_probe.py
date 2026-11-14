"""GPU probe of the SQG Polaris env and of the patched solver import (PROFILING_PROMPT.md §2c, §3.1b). Run by
sqg_env_probe.pbs on one compute-node GPU, before any profiling job.

  1. env: jax.devices(), torch.cuda.get_device_name(0), every harness import (pyqg_jax, timm, netCDF4, h5py), a
     4096^2 complex128 jnp.fft.fft2 checked against numpy, and the CUDA libraries actually mapped -> SQG_ENV_OK
  2. import: profile_solver.import_asqg() with SQG_TRAINING_ROOT set -- GD.TRAINING_ROOT is the override,
     GD.CONFIG["solver_sha256"] is the sha256 of the file on disk, and GD.advance(template, 2) is at AB3 level 2
     (the SQG_INIT=random start state) -> SQG_IMPORT_OK
Writes <out>/sqg_env_probe.json.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

SQG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_harness(name):
    """SQG/analysis/<name>.py as a module (top level is stdlib only; its SQG_* overrides are read from the env)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(SQG, "analysis", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ps = load_harness("profile_solver")
    res = {}

    # ---- 1. env ---------------------------------------------------------------------------------------------------
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_enable_x64", True)            # as A-SQG_data.py
    import pyqg_jax
    import torch
    import timm
    import netCDF4
    import h5py
    res["versions"] = {"jax": jax.__version__, "pyqg_jax": getattr(pyqg_jax, "__version__", "?"),
                       "torch": torch.__version__,
                       "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                       "timm": timm.__version__, "netCDF4": netCDF4.__version__, "h5py": h5py.__version__,
                       "numpy": np.__version__}
    res["jax_backend"], res["jax_devices"] = jax.default_backend(), [str(d) for d in jax.devices()]
    res["torch_device"] = torch.cuda.get_device_name(0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4096, 4096)) + 1j * rng.standard_normal((4096, 4096))
    t0 = time.perf_counter()
    y = jax.block_until_ready(jnp.fft.fft2(jnp.asarray(x)))
    first_s = time.perf_counter() - t0
    ref = np.fft.fft2(x)
    res["fft2_4096_c128"] = {"dtype": str(y.dtype), "first_call_s": round(first_s, 3),
                             "max_rel_err_vs_numpy": float(np.max(np.abs(np.asarray(y) - ref)) / np.max(np.abs(ref)))}
    res["loaded_cuda_libs"] = ps.loaded_libs()
    print(f"jax {jax.__version__} {res['jax_backend']} {res['jax_devices']}; torch {torch.__version__} "
          f"(cuda {torch.version.cuda}, cudnn {res['versions']['cudnn']}) on {res['torch_device']}")
    print(f"fft2 4096^2 {y.dtype}: max rel err vs numpy {res['fft2_4096_c128']['max_rel_err_vs_numpy']:.2e}")
    print(f"cuda libs: {[os.path.basename(p) for p in res['loaded_cuda_libs']]}")
    ok_env = (res["jax_backend"] == "gpu" and y.dtype == jnp.complex128
              and res["fft2_4096_c128"]["max_rel_err_vs_numpy"] < 1e-12)
    print("SQG_ENV_OK" if ok_env else "ERROR SQG_ENV_BAD: see sqg_env_probe.json", flush=True)

    # ---- 2. patched solver import -----------------------------------------------------------------------------------
    t0 = time.perf_counter()
    GD = ps.import_asqg()
    import_s = time.perf_counter() - t0
    with open(ps.ASQG_PY, "rb") as fh:
        disk_sha = hashlib.sha256(fh.read()).hexdigest()
    template = GD.sqg.init(GD.p, jax.random.key(GD.SEED_INITIAL_CONDITION))
    st = jax.block_until_ready(GD.advance(template, 2))
    want_root = Path(os.environ["SQG_TRAINING_ROOT"])
    res["import"] = {"import_s": round(import_s, 2), "asqg_py": ps.ASQG_PY, "training_root": str(GD.TRAINING_ROOT),
                     "run_dir_exists": GD.CHECKPOINT_DIR.is_dir(), "solver_sha256": GD.CONFIG["solver_sha256"],
                     "disk_sha256": disk_sha, "ablevel_after_advance_2": int(st._ablevel),
                     "qh_finite": bool(np.isfinite(np.asarray(st.state.qh)).all())}
    checks = {"training_root": GD.TRAINING_ROOT == want_root, "self_hash": GD.CONFIG["solver_sha256"] == disk_sha,
              "ablevel_2": int(st._ablevel) == 2, "run_dir": res["import"]["run_dir_exists"],
              "finite": res["import"]["qh_finite"]}
    res["import"]["checks"] = checks
    with open(os.path.join(args.out, "sqg_env_probe.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"solver import {import_s:.1f} s from {ps.ASQG_PY}; TRAINING_ROOT {GD.TRAINING_ROOT}; checks {checks}")
    print("SQG_IMPORT_OK" if all(checks.values()) else
          "ERROR SQG_IMPORT_BAD: " + ",".join(k for k, v in checks.items() if not v), flush=True)


if __name__ == "__main__":
    main()
