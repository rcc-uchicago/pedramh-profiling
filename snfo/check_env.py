"""Quick environment check — runs on a login node, no GPU required.

Usage:
    python check_env.py

Prints the version of every required package and attempts to import all
project modules. Exits with code 1 if anything is missing.
"""

import sys

errors = []

def check(label, fn):
    try:
        result = fn()
        print(f"  OK  {label:<30} {result}")
    except Exception as e:
        print(f"  FAIL {label:<30} {e}")
        errors.append(label)

print("\n=== Third-party packages ===")
check("torch",        lambda: __import__("torch").__version__)
check("torchvision",  lambda: __import__("torchvision").__version__)
check("lightning",    lambda: __import__("lightning").__version__)
check("wandb",        lambda: __import__("wandb").__version__)
check("numpy",        lambda: __import__("numpy").__version__)
check("xarray",       lambda: __import__("xarray").__version__)
check("cftime",       lambda: __import__("cftime").__version__)
check("h5py",         lambda: __import__("h5py").__version__)
check("h5netcdf",     lambda: __import__("h5netcdf").__version__)
check("h5pickle",     lambda: getattr(__import__("h5pickle"), "__version__", "installed"))
check("einops",       lambda: __import__("einops").__version__)
check("timm",         lambda: __import__("timm").__version__)
check("yaml (pyyaml)",lambda: __import__("yaml").__version__)
check("matplotlib",   lambda: __import__("matplotlib").__version__)
check("tqdm",         lambda: __import__("tqdm").__version__)
check("muon",         lambda: (
    __import__("muon"),
    getattr(__import__("muon"), "__version__", "installed")
)[-1])
check("torch_harmonics", lambda: __import__("torch_harmonics").__version__)
check("muon.MuonWithAuxAdam", lambda: (
    __import__("muon", fromlist=["MuonWithAuxAdam"]).MuonWithAuxAdam, "ok")[-1])

print("\n=== CUDA availability ===")
import torch
print(f"  cuda available : {torch.cuda.is_available()}")
print(f"  cuda version   : {torch.version.cuda}")
print(f"  device count   : {torch.cuda.device_count()}")

print("\n=== Project modules ===")
check("common.utils",        lambda: "ok")
check("common.loss",         lambda: (
    __import__("common.loss", fromlist=["latitude_weighted_rmse"]), "ok")[-1])
check("common.bench_callback", lambda: (
    __import__("common.bench_callback", fromlist=["BenchCallback"]), "ok")[-1])
check("data.amip_new",       lambda: (
    __import__("data.amip_new", fromlist=["GetDataset"]), "ok")[-1])
check("data.datamodule",     lambda: (
    __import__("data.datamodule", fromlist=["ClimateDataModule"]), "ok")[-1])
check("modules.train_module",lambda: (
    __import__("modules.train_module", fromlist=["TrainModule"]), "ok")[-1])
check("modules.ae_module",   lambda: (
    __import__("modules.ae_module", fromlist=["AutoencoderModule"]), "ok")[-1])
check("modules.models.DiT",  lambda: (
    __import__("modules.models.DiT", fromlist=["DiT"]), "ok")[-1])
check("modules.models.Unet", lambda: (
    __import__("modules.models.Unet", fromlist=["UNet"]), "ok")[-1])

print()
if errors:
    print(f"FAILED ({len(errors)} errors): {', '.join(errors)}")
    sys.exit(1)
else:
    print("All checks passed.")
