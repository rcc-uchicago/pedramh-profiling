#!/usr/bin/env python
"""Polaris toolchain probe (bring-up gate 1).

Confirms, on ONE Polaris compute node, that:
  * torch sees the 4x A100 GPUs,
  * the core scientific-python stack imports,
  * each model package imports under its own Polaris PYTHONPATH.

Each model repo ships colliding top-level package names (utils / networks /
data / modules), so every repo import is checked in a SEPARATE subprocess with
its own cwd + PYTHONPATH -- never in-process.

Prints one greppable final line:  PROBE_OK   or   PROBE_FAIL: <labels>
Hard-gated checks fail the probe; informational ones (SFNO frameworks that need
extra setup) only warn.
"""
import os
import sys
import socket
import platform
import subprocess

REPO = os.environ.get("REPO_ROOT", os.getcwd())
PY = sys.executable
_hard_fail = []
_warn_fail = []


def _last_line(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def run_import(label, cwd, pythonpath, code, hard=True):
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run([PY, "-c", code], cwd=cwd, env=env,
                           capture_output=True, text=True, timeout=420)
        ok = p.returncode == 0
        msg = _last_line(p.stdout) if ok else _last_line(p.stderr)
    except Exception as e:  # noqa: BLE001
        ok, msg = False, "%s: %s" % (type(e).__name__, e)
    if not ok:
        (_hard_fail if hard else _warn_fail).append(label)
    tag = "OK  " if ok else ("FAIL" if hard else "warn")
    print("  [%s] %-42s %s" % (tag, label, msg[:150]))
    return ok


print("=" * 74)
print("host=%s  python=%s" % (socket.gethostname(), platform.python_version()))
print("PBS_JOBID=%s  REPO_ROOT=%s" % (os.environ.get("PBS_JOBID", "?"), REPO))
print("=" * 74)

# --- torch + GPU visibility (in-process is safe; torch has no name collision) ---
try:
    import torch
    print("torch=%s  cuda=%s  is_available=%s  device_count=%d"
          % (torch.__version__, torch.version.cuda, torch.cuda.is_available(),
             torch.cuda.device_count()))
    n = torch.cuda.device_count()
    for i in range(n):
        pr = torch.cuda.get_device_properties(i)
        print("  gpu%d: %s  %.1f GB  sm%d%d" %
              (i, pr.name, pr.total_memory / 1e9, pr.major, pr.minor))
    if not torch.cuda.is_available() or n < 1:
        _hard_fail.append("torch.cuda")
        print("  [FAIL] torch.cuda: no CUDA device visible")
    else:
        print("  [OK  ] torch.cuda: %d GPU(s) visible" % n)
except Exception as e:  # noqa: BLE001
    _hard_fail.append("torch")
    print("  [FAIL] torch import: %s: %s" % (type(e).__name__, e))

print("-" * 74)
print("Core scientific-python stack:")
for m in ["numpy", "h5py", "netCDF4", "zarr", "xarray", "einops", "timm",
          "pytorch_lightning", "wandb"]:
    run_import("import %s" % m, REPO, "",
               "import %s as _m; print(getattr(_m,'__version__','ok'))" % m, hard=True)
# torch_harmonics: needed only by the SFNO frameworks; ABI-sensitive -> informational.
run_import("import torch_harmonics", REPO, "",
           "import torch_harmonics as _m; print(getattr(_m,'__version__','ok'))", hard=False)

print("-" * 74)
print("Model-package imports (each in its own subprocess / PYTHONPATH):")

run_import("S2S  networks.pangu + utils.losses",
           os.path.join(REPO, "s2s", "v2.0"),
           os.path.join(REPO, "s2s", "v2.0"),
           "import networks.pangu, utils.losses, utils.data_loader_multifiles; "
           "print('PanguModel_Plasim=%s' % hasattr(networks.pangu,'PanguModel_Plasim'))",
           hard=True)

# Import the REAL entrypoint module, not the top-level packages. `import common, data,
# modules` is a HOLLOW check: those dirs have no __init__.py, so they are namespace packages
# and the import succeeds without executing a single line of the smoke's code. That is how
# the port's missing `cf_xarray` (train_module.py:52, a bare import reached from both
# entrypoints) survived a green PROBE_OK — and the docs then told the next person the port's
# env was "proven by the probe" and only the ERA5 data was missing.
run_import("S2S-Lightning  modules.train_module",
           os.path.join(REPO, "s2s-lightning"),
           os.pathsep.join([os.path.join(REPO, "s2s", "v2.0"),
                            os.path.join(REPO, "s2s-lightning"),
                            os.environ.get("POLARIS_TOPUPS", "")]),
           "import modules.train_module; print('port train_module ok')",
           hard=True)

run_import("SI  common+data+modules",
           os.path.join(REPO, "si"),
           os.path.join(REPO, "si"),
           "import common, data, modules; print('si packages ok')",
           hard=True)

run_import("PanguWeather  networks.pangu",
           os.path.join(REPO, "PanguWeather", "v2.0"),
           os.path.join(REPO, "PanguWeather", "v2.0"),
           "import networks.pangu; print('pangu ok')",
           hard=True)

# SFNO frameworks: need their own install (pip -e / requirements) -> informational.
run_import("makani (import makani)",
           os.path.join(REPO, "makani_sfno"), os.path.join(REPO, "makani_sfno", "src"),
           "import makani; print('makani', getattr(makani,'__version__','ok'))", hard=False)
run_import("physicsnemo (import physicsnemo)",
           os.path.join(REPO, "physicsnemo_sfno"), os.path.join(REPO, "physicsnemo_sfno"),
           "import physicsnemo; print('physicsnemo', getattr(physicsnemo,'__version__','ok'))", hard=False)

print("=" * 74)
if _warn_fail:
    print("INFO (non-blocking) not yet importable: %s" % ", ".join(_warn_fail))
if _hard_fail:
    print("PROBE_FAIL: %s" % ", ".join(_hard_fail))
    sys.exit(1)
print("PROBE_OK")
