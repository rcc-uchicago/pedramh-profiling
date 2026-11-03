#!/usr/bin/env python3
"""
A-SQG_data (scheme C): long training trajectory, one member per GPU.

Model, numerics and forcing are those of the first version (500 T saved every
0.02, now A-SQG_data_v1_500T_save0.02.py.bak). Only the output changes:

  * spin-up t = 0 -> T_SPIN = 20 from the member's own small random IC
    (seed = --member; seed 0 is 1c/1.1); integrated, NOT saved;
  * production t = 20 -> 20 + T_PROD = 1120:
      theta  every DT_SAVE = 0.0025 (500 steps), float32;
      dtheta every BLOCK_T = 0.02 (4000 steps), at the same times as
      theta[8j]: dtheta = theta(t+DT) - theta(t), the one-step map of the
      model (AB3 + filter + damping), differenced in float64 on the GPU and
      then rounded to float32 (7 significant digits of the increment; the
      difference of two float32 fields keeps only ~4, because one step changes
      theta by ~1.6e-4 of its rms). theta(t+DT) = theta[8j] + dtheta[j],
      summed in float64;
      diagnostics (SPE, VIE, eps_nu, eps_mu, P, umax, E(k)) at every theta;
  * one NetCDF file per T_FILE = 1 time unit (400 theta + 50 dtheta, ~23 GB,
    zlib level 1 + shuffle, chunks (1, 256, 256), as the first version);
    records cover [t0, t0 + 1): t0, t0 + 0.0025, ..., t0 + 0.9975;
  * full float64 AB3 checkpoint archived at every file boundary
    (ckpt_t<time>.npz, never overwritten, ~0.4 GB) so the trajectory can be
    restarted bit for bit from any of them; latest.npz is used for resuming.

The GPU never waits for the disk: block j+1 is dispatched before block j is
copied to the host, and a writer thread compresses the chunks with a thread
pool (zlib releases the GIL) and writes them with h5py direct chunk writes,
byte-identical to what netCDF4 itself would write. A checkpoint is written by
the writer thread after the data it covers, so it is never ahead of the files.

    python A-SQG_data.py --member 1
    python A-SQG_data.py --member 1 --verify 30   # rerun one block from ckpt_t0030.00
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import queue
import signal
import sys
import threading
import time
import zlib
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Avoid HDF5 file-lock stalls on GPFS.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

try:
    from netCDF4 import Dataset
    import h5py
except ImportError as exc:
    raise ImportError(
        "netCDF4 and h5py are required in the environment."
    ) from exc


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Keep this script and sqg.py in the same directory.
EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = Path("/project/pedramh/weidong/SQG/SQG_deterForcing/AllTrainData")

if not (EXPERIMENT_ROOT / "sqg.py").is_file():
    raise FileNotFoundError(f"Cannot find {EXPERIMENT_ROOT / 'sqg.py'}")

sys.path.insert(0, str(EXPERIMENT_ROOT))
import sqg

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

N = 4096
L = 2.0 * np.pi

# NU=0.003958, B=1 gave Re_measured ~ 1000 at N=1024.
# Calibration run 1c (N=4096, NU=2e-4, B=1) gave Re_measured = 24596,
# kmax*l_nu = 1.76 (t = 10-20).
NU = 2.0e-4
B = 1.0

KF = 3.5
DKF = 0.5
MU = 1.0

# Inclusive forcing band: KF-DKF <= |k| <= KF+DKF.
# For the present band, M=24 signed modes and rms(f0)=sqrt(24)*B.
# B is the amplitude of each NORMALIZED Fourier coefficient of the RHS.


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="High-Re SQG training trajectory (scheme C), one member")
parser.add_argument("--member", type=int, required=True, help="member = random IC seed, >= 1 (seed 0 is 1c/1.1)")
parser.add_argument(
    "--verify", type=float, default=None, metavar="T",
    help="rerun one block (0.02) from the archived checkpoint at time T and "
         "compare theta and dtheta bitwise with the stored file (writes nothing)",
)
args = parser.parse_args()

MEMBER = args.member
VERIFY_T = args.verify

if MEMBER < 1:
    raise ValueError(f"--member must be >= 1 (IC seed 0 is the 1c/1.1 trajectory); got {MEMBER}")


# ---------------------------------------------------------------------------
# Integration and output
# ---------------------------------------------------------------------------

# DT=1e-5 reached a sampled CFL of 0.71 (abort at 0.72); DT=5e-6 stayed <= 0.42.
DT = 5e-6

# Spin-up from a small random IC, as 1c: statistically steady for t >= ~10.
# Integrated in chunks of SPIN_CHUNK for the CFL check, not saved.
T_START = 0.0
T_SPIN = 20.0
SPIN_CHUNK = 0.05
# 1100 time units = 2.2e8 steps = 440000 theta + 55000 dtheta (~287 h on an H100).
T_PROD = 1100.0
DT_SAVE = 0.0025
# One block = 8 theta = 0.02 = 4000 steps (~19 s); dtheta once per block;
# SIGUSR1/SIGTERM is honoured at a block boundary.
SAVES_PER_BLOCK = 8
# One NetCDF file and one archived checkpoint per time unit (50 blocks, ~16 min).
T_FILE = 1.0

# Host side: blocks waiting for the writer, and zlib threads of the writer.
QUEUE_BLOCKS = 4
WRITER_THREADS = 4
CHUNK = 256

SEED_INITIAL_CONDITION = MEMBER
SEED_FORCING = 1

CFL_WARNING = 0.60
CFL_ABORT = 0.72


# ---------------------------------------------------------------------------
# Derived configuration
# ---------------------------------------------------------------------------

if min(DT, NU, B, DT_SAVE, SPIN_CHUNK, T_PROD, T_FILE) <= 0 or T_SPIN < 0:
    raise ValueError("NU, B, timesteps and simulation durations must be positive.")

if not np.isclose(L, 2.0 * np.pi):
    raise ValueError("This configuration assumes the domain L=2*pi.")

L_F = 2.0 * np.pi / KF
KMAX = N / 3.0


def exact_integer(name, value):
    rounded = int(round(value))
    if not np.isclose(value, rounded, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} must be an integer; got {value}")
    return rounded


BLOCK_T = SAVES_PER_BLOCK * DT_SAVE
N_INNER = exact_integer("DT_SAVE / DT", DT_SAVE / DT)
N_INNER_SPIN = exact_integer("SPIN_CHUNK / DT", SPIN_CHUNK / DT)
N_SPIN_CHUNKS = exact_integer("T_SPIN / SPIN_CHUNK", T_SPIN / SPIN_CHUNK)
N_BLOCKS = exact_integer("T_PROD / BLOCK_T", T_PROD / BLOCK_T)
BLOCKS_PER_FILE = exact_integer("T_FILE / BLOCK_T", T_FILE / BLOCK_T)
N_FILES = exact_integer("T_PROD / T_FILE", T_PROD / T_FILE)
RECORDS_PER_FILE = BLOCKS_PER_FILE * SAVES_PER_BLOCK
T_FIRST = T_START + T_SPIN

if N % CHUNK:
    raise ValueError("N must be a multiple of CHUNK.")

SCHEME_VERSION = "frozen_ring_NU_B_v2"

RUN_LABEL = (
    f"trainingdata_m{MEMBER:02d}_{SCHEME_VERSION}"
    f"_nu{NU:.9g}_B{B:g}_mu{MU:g}"
    f"_kf{KF:g}_dkf{DKF:g}_seed{SEED_FORCING}"
    f"_N{N}_dt{DT:g}_Tstart{T_START:g}_Tspin{T_SPIN:g}"
    f"_Tprod{T_PROD:g}_save{DT_SAVE:g}_pair{BLOCK_T:g}"
)

RUN_DIR = TRAINING_ROOT / RUN_LABEL
SIMULATION_DIR = RUN_DIR / "simulation"
CHECKPOINT_DIR = RUN_DIR / "checkpoint"
CHECKPOINT_FILE = CHECKPOINT_DIR / "latest.npz"


def block_time(completed_blocks):
    """Model time after `completed_blocks` production blocks."""
    return T_FIRST + completed_blocks * BLOCK_T


def file_path(index):
    t0 = T_FIRST + index * T_FILE
    return SIMULATION_DIR / f"t{t0:07.2f}-{t0 + T_FILE:07.2f}.nc"


def archive_file(completed_blocks):
    """Archived checkpoint after `completed_blocks` blocks (a file boundary)."""
    return CHECKPOINT_DIR / f"ckpt_t{block_time(completed_blocks):07.2f}.npz"


CONFIG = {
    "scheme_version": SCHEME_VERSION,
    "output_scheme": "C: theta every dt_save, dtheta=theta(t+dt)-theta(t) every block_t, one file per t_file",
    "solver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "sqg_sha256": hashlib.sha256(Path(sqg.__file__).read_bytes()).hexdigest(),
    "N": N,
    "L": L,
    "nu": NU,
    "B": B,
    "mu": MU,
    "kf": KF,
    "dkf": DKF,
    "dt": DT,
    "T_start": T_START,
    "member": MEMBER,
    "initial_condition": "sqg.init small random field",
    "perturbation_amp": 0.0,
    "T_spin": T_SPIN,
    "spin_chunk": SPIN_CHUNK,
    "T_prod": T_PROD,
    "dt_save": DT_SAVE,
    "saves_per_block": SAVES_PER_BLOCK,
    "block_t": BLOCK_T,
    "t_file": T_FILE,
    "seed_initial_condition": SEED_INITIAL_CONDITION,
    "seed_forcing": SEED_FORCING,
    "forcing_method": "fixed Hermitian ring source inside AB3 tendency",
}

CONFIG_JSON = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"))
CONFIG_HASH = hashlib.sha256(CONFIG_JSON.encode()).hexdigest()

FIELD_BYTES = N * N * np.dtype(np.float32).itemsize


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

print("Python:", sys.executable, flush=True)
print("JAX version:", jax.__version__, flush=True)
print("JAX backend:", jax.default_backend(), flush=True)
print("JAX devices:", jax.devices(), flush=True)
print("JAX x64 enabled:", jax.config.jax_enable_x64, flush=True)

if jax.default_backend() != "gpu":
    raise RuntimeError("JAX is not using the allocated GPU")

SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Run label: {RUN_LABEL}", flush=True)
print(f"Files: {SIMULATION_DIR}", flush=True)
print(f"Checkpoints: {CHECKPOINT_DIR}", flush=True)
print(f"NU={NU:.9g}, B={B:g}, N={N}, DT={DT:g}", flush=True)
print(f"Member {MEMBER}: random IC seed={SEED_INITIAL_CONDITION}", flush=True)
print(
    f"spin-up={T_SPIN:g} (not saved), production={T_PROD:g} model-time units, "
    f"t = {T_FIRST:g} -> {T_FIRST + T_PROD:g}",
    flush=True,
)
print(
    f"theta every {DT_SAVE:g} ({N_BLOCKS * SAVES_PER_BLOCK} fields), "
    f"dtheta every {BLOCK_T:g} ({N_BLOCKS} fields), "
    f"{N_FILES} files of {T_FILE:g}; raw "
    f"{N_BLOCKS * (SAVES_PER_BLOCK + 1) * FIELD_BYTES / 1e12:.1f} TB before compression",
    flush=True,
)
print(
    f"physical steps={int(round((T_SPIN + T_PROD) / DT)):,}",
    flush=True,
)




# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

p = sqg.P(
    nx=N,
    L=L,
    kf=KF,
    dkf=DKF,
    eps=0.0,       # The helper's stochastic forcing is not used.
    r=0.0,         # Damping is applied explicitly below.
    dt=DT,
    filterfac=0.0,
    f64=True,
)

sm, ring, _, w, kap = sqg.aux(p)

kapi = jnp.where(kap > 0, kap, 1.0)
pos = (kap > 0).astype(jnp.float64)

# Ordinary viscosity NU*k^2 and hypofriction MU/k^2.
# No hyperdiffusion.
decay = (
    (kap <= KMAX).astype(jnp.float64)
    * jnp.exp(-(NU * kap**2 + MU * pos / kapi**2) * DT)
)

ibj = jnp.round(kap).astype(int)
nb = N // 2 + 1

kx = jnp.fft.rfftfreq(N, 1.0 / N)[None, :]
ky = jnp.fft.fftfreq(N, 1.0 / N)[:, None]


# ---------------------------------------------------------------------------
# Fixed forcing: draw phases once
# ---------------------------------------------------------------------------

if N % 2 or not (0 < KF - DKF <= KF + DKF <= KMAX):
    raise ValueError(
        "Use even N and a nonzero forcing band inside k <= N/3."
    )

n_forced_modes = int(round(float(jnp.sum(w * ring))))

if n_forced_modes == 0:
    raise ValueError("The forcing band contains no Fourier modes.")

# One representative per pair {k, -k}.
# Sorting by physical wavevector preserves the pattern when N changes.
forced_pairs = sorted(
    (int(ix), int(round(float(ky[iy, 0]))), int(iy))
    for iy, ix in np.argwhere(np.asarray(ring) != 0)
    if ix > 0 or ky[iy, 0] > 0
)

forcing_kx = np.array([q[0] for q in forced_pairs], dtype=np.int32)
forcing_ky = np.array([q[1] for q in forced_pairs], dtype=np.int32)
forcing_iy = np.array([q[2] for q in forced_pairs], dtype=np.int32)

phase = 2.0 * np.pi * jax.random.uniform(
    jax.random.key(SEED_FORCING),
    (len(forced_pairs),),
    dtype=jnp.float64,
)

fh = jnp.zeros(ring.shape, dtype=jnp.complex128)
fh = fh.at[forcing_iy, forcing_kx].set(jnp.exp(1j * phase))

# Enforce conjugate symmetry on the kx=0 column.
positive_y = jnp.arange(1, N // 2)
fh = fh.at[-positive_y, 0].set(jnp.conj(fh[positive_y, 0]))

# THE FORCING:
F0_hat = (N**2 * B * fh)[None]

f0 = np.asarray(jnp.fft.irfft2(F0_hat[0], s=(N, N)))
FORCING_RMS = float(np.sqrt(np.mean(f0**2)))
FORCING_HASH = hashlib.sha256(np.asarray(F0_hat).tobytes()).hexdigest()

print(
    f"Fixed forcing: {n_forced_modes} signed modes, "
    f"seed={SEED_FORCING}, B={B:g}, RMS={FORCING_RMS:.6g}",
    flush=True,
)


# ---------------------------------------------------------------------------
# Integration and diagnostics
# ---------------------------------------------------------------------------

def inner(state, _):
    """AB3 on advection + fixed forcing, then exact linear damping."""
    updates = sm.model.get_updates(state.state)
    updates = updates.update(qh=updates.qh + F0_hat)

    state = sm.stepper.apply_updates(state, updates)
    qh = sm.model.postprocess_state(state.state).qh * decay

    return state.update(state=state.state.update(qh=qh)), None


def diagnostics(state):
    qh = state.state.qh
    a = w * jnp.abs(qh[0] / N**2) ** 2

    spectrum = jnp.zeros(nb).at[ibj].add(0.5 * a, mode="drop")

    values = jnp.array(
        [
            0.5 * jnp.sum(a),                         # SPE
            0.5 * jnp.sum(pos * a / kapi),            # VIE
            NU * jnp.sum(kap**2 * a),                 # eps_nu
            MU * jnp.sum(pos * a / kapi**2),          # eps_mu
            # P = spatial mean(theta*f0), evaluated through Parseval.
            jnp.sum(
                w * jnp.real(jnp.conj(qh[0]) * F0_hat[0])
            ) / N**4,
        ]
    )

    u = jnp.fft.irfft2(-1j * ky * qh[0] / kapi, s=(N, N))
    v = jnp.fft.irfft2(1j * kx * qh[0] / kapi, s=(N, N))
    umax = jnp.max(jnp.sqrt(u**2 + v**2))

    return values, spectrum, umax



def advance(state, n_steps):
    state, _ = lax.scan(inner, state, None, length=n_steps)
    return state


def theta64(state):
    return jnp.fft.irfft2(state.state.qh[0], s=(N, N))


@jax.jit
def run_spin_chunk(state):
    state = advance(state, N_INNER_SPIN)
    return state, diagnostics(state)


@jax.jit
def run_production_block(state):
    """One block (BLOCK_T): 8 theta at t, t+DT_SAVE, ... and dtheta at t.

    dtheta = theta(t+DT) - theta(t) is differenced in float64 before the
    float32 rounding. The trajectory is the same 4000 AB3 steps as without
    the extra output."""
    theta_first = theta64(state)
    diag_first = diagnostics(state)
    state = advance(state, 1)
    dtheta = (theta64(state) - theta_first).astype(jnp.float32)
    state = advance(state, N_INNER - 1)

    def save(s, _):
        out = (*diagnostics(s), theta64(s).astype(jnp.float32))
        return advance(s, N_INNER), out

    state, rest = lax.scan(save, state, None, length=SAVES_PER_BLOCK - 1)
    first = (*diag_first, theta_first.astype(jnp.float32))
    values, spectra, umax, fields = (
        jnp.concatenate([a[None], b]) for a, b in zip(first, rest)
    )
    return state, (values, spectra, umax, fields, dtheta)


def check_finite_outputs(*arrays):
    if not all(np.isfinite(a).all() for a in arrays):
        raise FloatingPointError(
            "Non-finite output; the invalid block was not saved."
        )


# ---------------------------------------------------------------------------
# Full AB3 checkpoint
# ---------------------------------------------------------------------------

def qh_sha256(host_state):
    return hashlib.sha256(np.asarray(host_state.state.qh).tobytes()).hexdigest()


def save_checkpoint(host_state, completed_blocks):
    """Write latest.npz; on a file boundary it is a hard link to the archive."""
    leaves, _ = jax.tree_util.tree_flatten(host_state)

    payload = {
        "config_hash": np.asarray(CONFIG_HASH),
        "forcing_hash": np.asarray(FORCING_HASH),
        "completed_blocks": np.asarray(completed_blocks, dtype=np.int64),
        "t": np.asarray(block_time(completed_blocks), dtype=np.float64),
        "qh_sha256": np.asarray(qh_sha256(host_state)),
        "n_leaves": np.asarray(len(leaves), dtype=np.int64),
    }

    for index, leaf in enumerate(leaves):
        payload[f"leaf_{index}"] = np.asarray(leaf)

    temporary = CHECKPOINT_DIR / f".ckpt.{os.getpid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())

    if completed_blocks % BLOCKS_PER_FILE == 0:
        archive = archive_file(completed_blocks)
        temporary.replace(archive)
        os.link(archive, temporary)

    temporary.replace(CHECKPOINT_FILE)


def load_checkpoint(template_state, path=CHECKPOINT_FILE):
    template_leaves, tree = jax.tree_util.tree_flatten(template_state)

    with np.load(path, allow_pickle=False) as saved:
        if str(saved["config_hash"].item()) != CONFIG_HASH:
            raise RuntimeError("Checkpoint configuration does not match.")

        if str(saved["forcing_hash"].item()) != FORCING_HASH:
            raise RuntimeError("Checkpoint forcing does not match.")

        n_leaves = int(saved["n_leaves"])
        if n_leaves != len(template_leaves):
            raise RuntimeError("Checkpoint stepper structure is incompatible.")

        leaves = [np.asarray(saved[f"leaf_{i}"]) for i in range(n_leaves)]
        completed = int(saved["completed_blocks"])
        saved_sha = str(saved["qh_sha256"].item())

    host_state = jax.tree_util.tree_unflatten(tree, leaves)
    if qh_sha256(host_state) != saved_sha:
        raise RuntimeError(f"{path} is corrupt (qh checksum mismatch).")
    state = jax.tree_util.tree_map(jnp.asarray, host_state)
    return state, completed


# ---------------------------------------------------------------------------
# NetCDF files (one per T_FILE)
# ---------------------------------------------------------------------------

DIAGNOSTIC_NAMES = ("SPE", "VIE", "eps_nu", "eps_mu", "P")


def create_file(index):
    """Define one file with netCDF4 (all metadata, no field data)."""
    path = file_path(index)
    path.unlink(missing_ok=True)
    t0 = T_FIRST + index * T_FILE

    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("time", RECORDS_PER_FILE)
        nc.createDimension("pair_time", BLOCKS_PER_FILE)
        nc.createDimension("y", N)
        nc.createDimension("x", N)
        nc.createDimension("wavenumber", nb - 1)
        nc.createDimension("forcing_pair", len(forced_pairs))

        nc.createVariable("x", "f8", ("x",))[:] = np.arange(N) * L / N
        nc.createVariable("y", "f8", ("y",))[:] = np.arange(N) * L / N
        nc.createVariable(
            "wavenumber", "i4", ("wavenumber",)
        )[:] = np.arange(1, nb, dtype=np.int32)

        tvar = nc.createVariable("time", "f8", ("time",))
        tvar[:] = t0 + np.arange(RECORDS_PER_FILE) * DT_SAVE
        tvar.long_name = "model time of theta (absolute; the random IC is at T_start)"

        pvar = nc.createVariable("pair_time", "f8", ("pair_time",))
        pvar[:] = t0 + np.arange(BLOCKS_PER_FILE) * BLOCK_T
        pvar.long_name = "model time t of dtheta; theta(t) is theta[time == t]"

        nc.createVariable(
            "theta", "f4", ("time", "y", "x"),
            zlib=True, complevel=1, shuffle=True,
            chunksizes=(1, CHUNK, CHUNK), fill_value=False,
        ).long_name = "surface buoyancy"

        dvar = nc.createVariable(
            "dtheta", "f4", ("pair_time", "y", "x"),
            zlib=True, complevel=1, shuffle=True,
            chunksizes=(1, CHUNK, CHUNK), fill_value=False,
        )
        dvar.long_name = "one-step increment theta(t+dt) - theta(t)"
        dvar.comment = (
            "Differenced in float64 on the GPU, then rounded to float32. "
            "theta(t+dt) = theta[time == t] + dtheta, summed in float64. "
            "One model step: AB3(advection + f0), pyqg filter, exact linear damping."
        )

        fvar = nc.createVariable(
            "f0", "f8", ("y", "x"), zlib=True, complevel=1
        )
        fvar[:] = f0
        fvar.long_name = "fixed deterministic forcing in the theta tendency"

        for name, data, dtype in (
            ("forcing_kx", forcing_kx, "i4"),
            ("forcing_ky", forcing_ky, "i4"),
            ("forcing_phase", np.asarray(phase), "f8"),
        ):
            nc.createVariable(name, dtype, ("forcing_pair",))[:] = data

        for name, description in (
            ("SPE", "surface potential energy"),
            ("VIE", "vertically integrated energy"),
            ("eps_nu", "viscous SPE dissipation rate"),
            ("eps_mu", "hypofriction SPE dissipation rate"),
            ("P", "instantaneous spatial mean(theta*f0)"),
            ("umax", "maximum horizontal speed"),
        ):
            nc.createVariable(
                name, "f8", ("time",), fill_value=False
            ).long_name = description

        nc.createVariable(
            "E", "f8", ("time", "wavenumber"),
            zlib=True, complevel=1, shuffle=True, fill_value=False,
        ).long_name = "instantaneous isotropic SPE spectrum"

        nc.title = "SQG with deterministic frozen-phase ring forcing (training data, A-SQG_data scheme C)"
        nc.forcing_type = "deterministic frozen-phase ring"
        nc.scheme_version = SCHEME_VERSION
        nc.jax_version = jax.__version__
        nc.numpy_version = np.__version__
        nc.run_label = RUN_LABEL
        nc.config_json = CONFIG_JSON
        nc.config_hash = CONFIG_HASH

        nc.N = N
        nc.L = L
        nc.nu = NU
        nc.B = B
        nc.mu = MU
        nc.kf = KF
        nc.dkf = DKF
        nc.l_f = L_F

        nc.dt = DT
        nc.dt_save = DT_SAVE
        nc.dt_pair = BLOCK_T
        nc.T_start = T_START
        nc.T_spin = T_SPIN
        nc.T_prod = T_PROD
        nc.t_file_start = t0
        nc.t_file_end = t0 + T_FILE
        nc.file_index = index
        nc.member = MEMBER
        nc.initial_condition = f"small random field, sqg.init seed {SEED_INITIAL_CONDITION}, at T_start"
        nc.seed_initial_condition = SEED_INITIAL_CONDITION
        nc.checkpoint_dir = str(CHECKPOINT_DIR)
        nc.checkpoint_note = (
            "ckpt_t<t_file_start>.npz holds the full float64 AB3 state at the "
            "start of this file; restarting from it reproduces it bit for bit."
        )

        nc.forcing_seed = SEED_FORCING
        nc.forcing_mode_count = n_forced_modes
        nc.forcing_rms = FORCING_RMS
        nc.forcing_hash = FORCING_HASH

        nc.dealiasing = "circular 2/3 truncation"
        nc.integration = (
            "AB3(advection+f0), then exact linear damping; "
            "first-order split overall"
        )
        nc.precision_model = "float64"
        nc.precision_output = "float32"
        nc.usage = "Read theta[:valid_records] and dtheta[:valid_pair_records]; complete=1 when the file is full."

        nc.valid_records = 0
        nc.valid_pair_records = 0
        nc.complete = 0


def file_is_complete(index):
    path = file_path(index)
    if not path.exists():
        return False
    with h5py.File(path, "r") as h5:
        return (
            int(h5.attrs.get("complete", 0)) == 1
            and _attr_str(h5.attrs.get("config_hash")) == CONFIG_HASH
        )


def _attr_str(value):
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _shuffle_compress(block):
    """HDF5 shuffle (4-byte elements) + deflate level 1, as netCDF4 writes it."""
    raw = np.ascontiguousarray(block).view(np.uint8).reshape(-1, 4).T.tobytes()
    return zlib.compress(raw, 1)


CHUNK_OFFSETS = [(y, x) for y in range(0, N, CHUNK) for x in range(0, N, CHUNK)]


class Writer(threading.Thread):
    """Writes blocks and checkpoints in queue order; never blocks the GPU loop
    unless QUEUE_BLOCKS blocks are waiting."""

    def __init__(self):
        super().__init__(name="writer", daemon=True)
        self.queue = queue.Queue(maxsize=QUEUE_BLOCKS)
        self.pool = ThreadPoolExecutor(WRITER_THREADS, thread_name_prefix="zlib")
        self.error = None
        self.h5 = None
        self.index = None
        self.busy = 0.0          # seconds spent writing blocks
        self.blocks = 0
        self.bytes = 0           # compressed field bytes written

    # -- main-thread side -------------------------------------------------
    def put(self, item):
        """Returns the seconds the main thread waited for queue space."""
        start = time.time()
        while True:
            self.raise_error()
            try:
                self.queue.put(item, timeout=1.0)
                return time.time() - start
            except queue.Full:
                continue

    def raise_error(self):
        if self.error is not None:
            raise RuntimeError("Writer thread failed") from self.error

    def close(self):
        if self.is_alive():
            self.queue.put(None)
            self.join()
        self.pool.shutdown()
        self.raise_error()

    # -- writer-thread side -----------------------------------------------
    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                self._close_file()
                return
            if self.error is not None:
                continue           # drain; main thread raises
            try:
                kind = item[0]
                if kind == "block":
                    self._write_block(*item[1:])
                elif kind == "checkpoint":
                    self._checkpoint(*item[1:])
            except BaseException as exc:
                self.error = exc
                self._close_file()

    def _open(self, index, offset_blocks):
        if self.index == index:
            return
        self._close_file()
        if offset_blocks == 0:
            create_file(index)
        elif not file_path(index).exists():
            raise FileNotFoundError(file_path(index))
        self.h5 = h5py.File(file_path(index), "r+")
        if _attr_str(self.h5.attrs.get("config_hash")) != CONFIG_HASH:
            raise RuntimeError(f"{file_path(index)} does not match this configuration.")
        self.index = index

    def _close_file(self):
        if self.h5 is not None:
            self.h5.close()
        self.h5, self.index = None, None

    def _write_field(self, dset, record, field):
        blocks = [field[y:y + CHUNK, x:x + CHUNK] for y, x in CHUNK_OFFSETS]
        for (y, x), data in zip(CHUNK_OFFSETS, self.pool.map(_shuffle_compress, blocks)):
            dset.id.write_direct_chunk((record, y, x), data)
            self.bytes += len(data)

    def _write_block(self, block, values, spectra, umax, fields, dtheta):
        start = time.time()
        index, offset = divmod(block, BLOCKS_PER_FILE)
        self._open(index, offset)
        h5 = self.h5
        first = offset * SAVES_PER_BLOCK
        stop = first + SAVES_PER_BLOCK

        for i in range(SAVES_PER_BLOCK):
            self._write_field(h5["theta"], first + i, fields[i])
        self._write_field(h5["dtheta"], offset, dtheta)

        for column, name in enumerate(DIAGNOSTIC_NAMES):
            h5[name][first:stop] = values[:, column]
        h5["umax"][first:stop] = umax
        h5["E"][first:stop] = spectra[:, 1:]

        if offset == BLOCKS_PER_FILE - 1:
            h5.attrs["valid_records"] = np.int64(RECORDS_PER_FILE)
            h5.attrs["valid_pair_records"] = np.int64(BLOCKS_PER_FILE)
            h5.attrs["complete"] = np.int64(1)
            self._close_file()

        self.busy += time.time() - start
        self.blocks += 1

    def _checkpoint(self, completed_blocks, host_state):
        # The data it covers is already in the file (queue order); make it durable.
        if self.h5 is not None:
            offset = completed_blocks - self.index * BLOCKS_PER_FILE
            self.h5.attrs["valid_records"] = np.int64(offset * SAVES_PER_BLOCK)
            self.h5.attrs["valid_pair_records"] = np.int64(offset)
            self.h5.flush()
        save_checkpoint(host_state, completed_blocks)


# ---------------------------------------------------------------------------
# Interruption handling
# ---------------------------------------------------------------------------

stop_requested = False


def request_stop(signum, _frame):
    global stop_requested
    stop_requested = True
    print(
        f"Received signal {signum}; checkpointing after the current block. "
        "During spin-up, the checkpoint is written after spin-up completes.",
        flush=True,
    )


signal.signal(signal.SIGUSR1, request_stop)
signal.signal(signal.SIGTERM, request_stop)


# ---------------------------------------------------------------------------
# Verification: rerun one block from an archived checkpoint
# ---------------------------------------------------------------------------

def verify(t_verify):
    steps = (t_verify - T_FIRST) / T_FILE
    index = int(round(steps))
    if abs(steps - index) > 1e-6 or not 0 <= index < N_FILES:
        raise ValueError(f"t={t_verify:g} is not an archived checkpoint time.")
    completed = index * BLOCKS_PER_FILE
    source = archive_file(completed)
    if not source.exists():
        raise FileNotFoundError(source)

    template_state = sqg.init(p, jax.random.key(SEED_INITIAL_CONDITION))
    state, saved_completed = load_checkpoint(template_state, source)
    if saved_completed != completed:
        raise RuntimeError(f"{source} holds block {saved_completed}, not {completed}.")

    print(f"Verify: {source.name} -> one block vs {file_path(index).name}", flush=True)
    _, result = run_production_block(state)
    values, spectra, umax, fields, dtheta = jax.device_get(result)

    with Dataset(file_path(index)) as nc:
        nc.set_auto_mask(False)
        if int(nc.valid_pair_records) < 1:
            raise RuntimeError("The file holds no committed block.")
        theta_equal = np.array_equal(fields, nc.variables["theta"][:SAVES_PER_BLOCK])
        dtheta_equal = np.array_equal(dtheta, nc.variables["dtheta"][0])
        spe_equal = np.array_equal(values[:, 0], nc.variables["SPE"][:SAVES_PER_BLOCK])

    print(
        f"theta bitwise equal: {theta_equal}; dtheta bitwise equal: {dtheta_equal}; "
        f"SPE equal: {spe_equal}",
        flush=True,
    )
    if not (theta_equal and dtheta_equal and spe_equal):
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def spin_up(state):
    wall_start = time.time()
    print(
        f"Spin-up from the random IC (seed {SEED_INITIAL_CONDITION}), "
        f"t={T_START:g} -> {T_FIRST:g}, not saved; first call includes JAX compilation.",
        flush=True,
    )
    max_cfl = 0.0
    report_every = max(1, N_SPIN_CHUNKS // 20)
    for chunk in range(N_SPIN_CHUNKS):
        state, (values, _, umax) = run_spin_chunk(state)
        values, umax = np.asarray(values), float(umax)
        check_finite_outputs(values, np.asarray(umax))
        cfl = umax * KMAX * DT
        max_cfl = max(max_cfl, cfl)
        if cfl >= CFL_ABORT:
            raise RuntimeError(
                f"Spin-up spectral CFL {cfl:.4f} >= {CFL_ABORT}; "
                "reduce DT while keeping NU and B fixed."
            )
        if (chunk + 1) % report_every and chunk + 1 != N_SPIN_CHUNKS:
            continue
        elapsed = time.time() - wall_start
        print(
            f"spin-up t={T_START + (chunk + 1) * SPIN_CHUNK:7.2f}  SPE={values[0]:9.4f}  "
            f"P={values[4]:9.4f}  CFL={cfl:.4f}  wall={elapsed / 3600:.2f} h",
            flush=True,
        )
    print(f"Spin-up complete (max sampled CFL {max_cfl:.4f}).", flush=True)
    return state


def try_checkpoint(template_state, path):
    """(state, completed) if `path` loads and the files before it are complete, else None."""
    try:
        state, completed = load_checkpoint(template_state, path)
    except Exception as exc:
        print(f"Skipping {path.name}: {exc}", flush=True)
        return None
    if not 0 <= completed <= N_BLOCKS:
        print(f"Skipping {path.name}: block {completed} outside 0..{N_BLOCKS}", flush=True)
        return None
    current, offset = divmod(completed, BLOCKS_PER_FILE)
    for index in range(current):
        if not file_is_complete(index):
            print(f"Skipping {path.name}: {file_path(index).name} is missing or incomplete", flush=True)
            return None
    if offset and not file_path(current).exists():
        print(f"Skipping {path.name}: it is inside {file_path(current).name}, which is missing", flush=True)
        return None
    return state, completed


def resume_or_start():
    """Resume from the newest usable checkpoint: latest.npz, else the archives
    ckpt_t*.npz from newest to oldest; start from the spin-up only if there is none."""
    template_state = sqg.init(p, jax.random.key(SEED_INITIAL_CONDITION))

    candidates = [CHECKPOINT_FILE] if CHECKPOINT_FILE.exists() else []
    candidates += sorted(CHECKPOINT_DIR.glob("ckpt_t*.npz"), reverse=True)

    for path in candidates:
        found = try_checkpoint(template_state, path)
        if found is None:
            continue
        state, completed = found
        current, offset = divmod(completed, BLOCKS_PER_FILE)
        # Anything after the checkpoint is uncommitted: rewritten from here.
        for index in range(current + (1 if offset else 0), N_FILES):
            file_path(index).unlink(missing_ok=True)
        print(
            f"Resuming from {path.name} at block {completed}/{N_BLOCKS}, "
            f"t={block_time(completed):.2f}",
            flush=True,
        )
        return state, completed

    if candidates:
        raise RuntimeError(
            f"No usable checkpoint in {CHECKPOINT_DIR} (see the messages above); "
            "move the run directory away to start again."
        )
    if any(SIMULATION_DIR.glob("*.nc")):
        raise FileExistsError(
            f"{RUN_DIR} holds files but no checkpoint; move it away to start again."
        )

    state = spin_up(template_state)
    save_checkpoint(jax.device_get(state), 0)
    return state, 0


def main():
    if VERIFY_T is not None:
        verify(VERIFY_T)
        return

    state, completed = resume_or_start()
    if completed == N_BLOCKS:
        print("Run is already complete.", flush=True)
        return
    if stop_requested:
        print("Stop requested; retaining the checkpoint.", flush=True)
        return

    writer = Writer()
    writer.start()

    wall_start = time.time()
    first_block = completed
    max_cfl_seen = 0.0
    waited = 0.0             # main thread blocked on a full writer queue
    window = (time.time(), 0.0, 0.0, 0)   # wall, waited, writer busy, writer blocks at last report

    # Block b+1 is queued on the GPU before block b is copied to the host.
    pending = run_production_block(state)

    try:
        for block in range(first_block, N_BLOCKS):
            next_state, result = pending
            last = block + 1 == N_BLOCKS
            if not last:
                pending = run_production_block(next_state)

            values, spectra, umax, fields, dtheta = jax.device_get(result)
            check_finite_outputs(values, spectra, umax, fields, dtheta)

            block_cfl = float(np.max(umax) * KMAX * DT)
            if block_cfl >= CFL_ABORT:
                raise RuntimeError(
                    f"Sampled spectral CFL {block_cfl:.4f} >= {CFL_ABORT}. "
                    "Reduce DT while keeping NU and B fixed. "
                    "Invalid block not saved; previous checkpoint retained."
                )
            max_cfl_seen = max(max_cfl_seen, block_cfl)

            waited += writer.put(("block", block, values, spectra, umax, fields, dtheta))
            completed = block + 1

            if completed % BLOCKS_PER_FILE == 0 or last or stop_requested:
                waited += writer.put(("checkpoint", completed, jax.device_get(next_state)))

            if completed % 10 == 0 or last or stop_requested:
                now = time.time()
                w_wall, w_wait, w_busy, w_blocks = window
                done = completed - first_block
                n_win = writer.blocks - w_blocks
                per_block = (now - w_wall) / 10 if completed % 10 == 0 else float("nan")
                eta = (now - wall_start) / done * (N_BLOCKS - completed)
                print(
                    f"block {completed:6d}/{N_BLOCKS}  t={block_time(completed):8.2f}  "
                    f"SPE={values[-1, 0]:9.4f}  P={values[-1, 4]:9.4f}  CFL={block_cfl:.4f}  "
                    f"wall/block={per_block:6.2f}s  main-waited={waited - w_wait:5.1f}s/10blk  "
                    f"writer={(writer.busy - w_busy) / max(n_win, 1):5.2f}s/blk  "
                    f"queue={writer.queue.qsize()}  "
                    f"MB/field={writer.bytes / max(writer.blocks * (SAVES_PER_BLOCK + 1), 1) / 1e6:5.1f}  "
                    f"ETA={eta / 3600:.1f} h",
                    flush=True,
                )
                window = (now, waited, writer.busy, writer.blocks)

            if block_cfl >= CFL_WARNING:
                print(f"WARNING: sampled spectral CFL={block_cfl:.4f}", flush=True)

            if stop_requested:
                print(
                    f"Stopping safely at block {completed} (t={block_time(completed):.2f}); "
                    "rerun A-SQG_data.sh to resume.",
                    flush=True,
                )
                return

        print(
            f"Run complete: t={block_time(N_BLOCKS):g}, max sampled CFL {max_cfl_seen:.4f}, "
            f"wall {(time.time() - wall_start) / 3600:.2f} h, "
            f"main thread waited for the writer {waited:.0f} s in total.",
            flush=True,
        )

    finally:
        writer.close()


if __name__ == "__main__":
    main()
