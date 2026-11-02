#!/usr/bin/env python3
"""
A-SQG_data: training trajectories with their own spin-up (2-GenData.py with
the spin-up of 1c / old2-GenerateData.py instead of a restart from 1.1).

Same model, numerics, production output and checkpointing as 2-GenData.py.
Only the start differs: each member starts at t=0 from its own small random
field (sqg.init, SEED_INITIAL_CONDITION = --member; the frozen ring forcing,
SEED_FORCING=1, is the same for all members and for 1c/1.1/2-GenData, whose
IC seed was 0), spins up for T_SPIN=20 (1c was statistically steady for
t >= ~10), then integrates T_PROD=500 time units, saving theta every
DT_SAVE=0.02 (t = 20 -> 520, 25000 fields).

Spin-up: theta and the diagnostics every DT_SAVE_SPIN=0.05 (as in 1c) go to
trainingdata/spinup/<RUN_LABEL>_spinup.nc. There is no checkpoint during the
spin-up: SIGUSR1/SIGTERM is honoured after it ends, and a run interrupted
during the spin-up starts it again.

Checkpoints: the full float64 AB3 state is archived every 1 time unit
(ckpt_t<time>.npz, never overwritten, including t=T_START+T_SPIN=20, the end
of the spin-up), so the trajectory can be restarted bit for bit from any of
them. latest.npz (a hard link to the newest archive, or a SIGUSR1/SIGTERM
checkpoint) is used for resuming.

    python A-SQG_data.py --member 1
    python A-SQG_data.py --member 1 --verify 30   # rerun 1 time unit from ckpt_t030.00
"""

from __future__ import annotations

import argparse
import hashlib
from functools import partial
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# The output NetCDFs may be read (e.g. from a notebook) while they are written.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from tqdm import tqdm

try:
    from netCDF4 import Dataset
except ImportError as exc:
    raise ImportError(
        "netCDF4 is required in the container environment."
    ) from exc


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Keep this script and sqg.py in the same directory.
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/weidong/SQG_data")
GENDATA_ROOT = DATA_ROOT / "GenData_deterForcing"
TRAINING_ROOT = GENDATA_ROOT / "trainingdata"
SPINUP_DIR = TRAINING_ROOT / "spinup"
SIMULATION_DIR = TRAINING_ROOT / "simulation"
CHECKPOINT_DIR = TRAINING_ROOT / "checkpoint"

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

parser = argparse.ArgumentParser(description="High-Re SQG training trajectory with spin-up, one member")
parser.add_argument("--member", type=int, required=True, help="member = random IC seed, >= 1 (seed 0 is 1c/1.1)")
parser.add_argument(
    "--verify", type=float, default=None, metavar="T",
    help="rerun one checkpoint interval from the archived checkpoint at time T "
         "and compare it bitwise with the stored trajectory (writes nothing)",
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

# Spin-up from a small random IC, as 1c (0-run_sqg.py): 1c was statistically
# steady for t >= ~10, and 2-GenData only used 1.1 records after t=20.
T_START = 0.0
T_SPIN = 20.0
DT_SAVE_SPIN = 0.05
# 500 time units = 1e8 steps = 25000 saved fields (~138 h, ~1.05 TB) per run.
T_PROD = 500.0
DT_SAVE = 0.02

# 0.02 / DT = 4000 integration steps per saved field (~19 s at 4.7 ms/step).
# Two outputs per block keep a SIGUSR1 checkpoint within ~1 minute;
# an archived checkpoint every 25 blocks = 1 model-time unit (~16 min, ~400 MB).
OUTPUTS_PER_BLOCK = 2
CHECKPOINT_EVERY_BLOCKS = 25

SEED_INITIAL_CONDITION = MEMBER
SEED_FORCING = 1

RESUME = True
OVERWRITE = False

CFL_WARNING = 0.60
CFL_ABORT = 0.72


# ---------------------------------------------------------------------------
# Derived configuration
# ---------------------------------------------------------------------------

if min(DT, NU, B, DT_SAVE, DT_SAVE_SPIN, T_PROD) <= 0 or T_SPIN < 0:
    raise ValueError("NU, B, timesteps and simulation durations must be positive.")

if not np.isclose(L, 2.0 * np.pi):
    raise ValueError("This configuration assumes the domain L=2*pi.")

L_F = 2.0 * np.pi / KF
KMAX = N / 3.0


def exact_integer(name, value):
    rounded = int(round(value))
    if not np.isclose(value, rounded, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be an integer; got {value}")
    return rounded


N_INNER = exact_integer("DT_SAVE / DT", DT_SAVE / DT)
N_INNER_SPIN = exact_integer("DT_SAVE_SPIN / DT", DT_SAVE_SPIN / DT)
N_SPIN_OUTPUTS = exact_integer("T_SPIN / DT_SAVE_SPIN", T_SPIN / DT_SAVE_SPIN)
N_PROD_OUTPUTS = exact_integer("T_PROD / DT_SAVE", T_PROD / DT_SAVE)
N_BLOCKS = exact_integer(
    "N_PROD_OUTPUTS / OUTPUTS_PER_BLOCK",
    N_PROD_OUTPUTS / OUTPUTS_PER_BLOCK,
)
OUTPUTS_PER_CHECKPOINT = OUTPUTS_PER_BLOCK * CHECKPOINT_EVERY_BLOCKS
exact_integer(
    "N_PROD_OUTPUTS / OUTPUTS_PER_CHECKPOINT",
    N_PROD_OUTPUTS / OUTPUTS_PER_CHECKPOINT,
)

SCHEME_VERSION = "frozen_ring_NU_B_v2"

RUN_LABEL = (
    f"trainingdata_m{MEMBER:02d}_{SCHEME_VERSION}"
    f"_nu{NU:.9g}_B{B:g}_mu{MU:g}"
    f"_kf{KF:g}_dkf{DKF:g}_seed{SEED_FORCING}"
    f"_N{N}_dt{DT:g}_Tstart{T_START:g}_Tspin{T_SPIN:g}"
    f"_Tprod{T_PROD:g}_save{DT_SAVE:g}"
)

NETCDF_FILE = SIMULATION_DIR / f"{RUN_LABEL}.nc"
RUN_CHECKPOINT_DIR = CHECKPOINT_DIR / RUN_LABEL
CHECKPOINT_FILE = RUN_CHECKPOINT_DIR / "latest.npz"
SPIN_NETCDF_FILE = SPINUP_DIR / f"{RUN_LABEL}_spinup.nc"


def output_time(completed):
    return T_START + T_SPIN + completed * DT_SAVE


def archive_file(completed):
    """Archived checkpoint after `completed` production outputs."""
    return RUN_CHECKPOINT_DIR / f"ckpt_t{output_time(completed):06.2f}.npz"

CONFIG = {
    "scheme_version": SCHEME_VERSION,
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
    "dt_save_spin": DT_SAVE_SPIN,
    "T_prod": T_PROD,
    "dt_save": DT_SAVE,
    "outputs_per_block": OUTPUTS_PER_BLOCK,
    "outputs_per_checkpoint": OUTPUTS_PER_CHECKPOINT,
    "seed_initial_condition": SEED_INITIAL_CONDITION,
    "seed_forcing": SEED_FORCING,
    "forcing_method": "fixed Hermitian ring source inside AB3 tendency",
}

CONFIG_JSON = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"))
CONFIG_HASH = hashlib.sha256(CONFIG_JSON.encode()).hexdigest()

RAW_FIELD_GIB = (
    N_PROD_OUTPUTS * N * N * np.dtype(np.float32).itemsize / 1024**3
)


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

SPINUP_DIR.mkdir(parents=True, exist_ok=True)
SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
RUN_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Run label: {RUN_LABEL}", flush=True)
print(f"NetCDF: {NETCDF_FILE}", flush=True)
print(f"Checkpoints: {RUN_CHECKPOINT_DIR}", flush=True)
print(f"NU={NU:.9g}, B={B:g}, N={N}, DT={DT:g}", flush=True)
print(
    f"Member {MEMBER}: random IC seed={SEED_INITIAL_CONDITION}, "
    f"spin-up NetCDF: {SPIN_NETCDF_FILE}",
    flush=True,
)
print(
    f"spin={T_SPIN:g}, production={T_PROD:g} model-time units",
    flush=True,
)
print(
    f"fields={N_PROD_OUTPUTS}, interval={DT_SAVE:g}, "
    f"raw theta={RAW_FIELD_GIB:.2f} GiB before compression",
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


def spin_output(state, _):
    state, _ = lax.scan(inner, state, None, length=N_INNER)
    values, spectrum, umax = diagnostics(state)
    return state, (values, spectrum, umax)


def spinup_output(state, _):
    """One spin-up save interval (DT_SAVE_SPIN): diagnostics and the field."""
    state, _ = lax.scan(inner, state, None, length=N_INNER_SPIN)
    values, spectrum, umax = diagnostics(state)
    field = jnp.fft.irfft2(
        state.state.qh[0], s=(N, N)
    ).astype(jnp.float32)
    return state, (values, spectrum, umax, field)


def production_output(state, _):
    state, result = spin_output(state, None)
    field = jnp.fft.irfft2(
        state.state.qh[0], s=(N, N)
    ).astype(jnp.float32)
    return state, (*result, field)


@partial(jax.jit, static_argnames=("n_outputs",))
def run_spin(state, n_outputs=OUTPUTS_PER_BLOCK):
    return lax.scan(spinup_output, state, None, length=n_outputs)


@jax.jit
def run_production_block(state):
    return lax.scan(
        production_output, state, None, length=OUTPUTS_PER_BLOCK
    )


# ---------------------------------------------------------------------------
# NetCDF
# ---------------------------------------------------------------------------

def create_netcdf(spin_values, spin_umax):
    nc = Dataset(NETCDF_FILE, "w", format="NETCDF4")

    nc.createDimension("time", None)
    if N_SPIN_OUTPUTS:
        nc.createDimension("spin_time", N_SPIN_OUTPUTS)
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
    tvar.long_name = "model time (absolute; the initial condition is at T_start)"
    tvar.comment = "Production only; first record is T_start + T_spin + dt_save."

    if N_SPIN_OUTPUTS:
        nc.createVariable(
            "spin_time", "f8", ("spin_time",)
        )[:] = T_START + (np.arange(N_SPIN_OUTPUTS) + 1) * DT_SAVE_SPIN

    nc.createVariable(
        "theta",
        "f4",
        ("time", "y", "x"),
        zlib=True,
        complevel=1,
        shuffle=True,
        chunksizes=(1, min(256, N), min(256, N)),
        fill_value=False,
    ).long_name = "surface buoyancy"

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

    nc.variables["forcing_phase"].comment = (
        "Radians; opposite wavevector has the negative phase."
    )

    for name, description in (
        ("SPE", "surface potential energy"),
        ("VIE", "vertically integrated energy"),
        ("eps_nu", "viscous SPE dissipation rate"),
        ("eps_mu", "hypofriction SPE dissipation rate"),
        ("umax", "maximum horizontal speed"),
        ("P", "instantaneous spatial mean(theta*f0)"),
    ):
        var = nc.createVariable(
            name, "f8", ("time",), fill_value=False
        )
        var.long_name = description

    for name, description in (
        ("P_mean", "mean of saved production P; spin-up excluded"),
        ("eps_nu_mean", "mean of saved production viscous SPE dissipation"),
        ("Re_measured", "P_mean**(1/3) * l_f**(4/3) / nu"),
        ("Re_dissipation", "eps_nu_mean**(1/3) * l_f**(4/3) / nu"),
        ("tau_f_measured", "P_mean**(-1/3) * l_f**(2/3)"),
    ):
        var = nc.createVariable(name, "f8", ())
        var.long_name = description
        var.assignValue(np.nan)
        nc.setncattr(name, np.nan)

    nc.createVariable(
        "E",
        "f8",
        ("time", "wavenumber"),
        zlib=True,
        complevel=1,
        shuffle=True,
        fill_value=False,
    ).long_name = "instantaneous isotropic SPE spectrum"

    if N_SPIN_OUTPUTS:
        for column, name in enumerate(
            ("SPE_spin", "VIE_spin", "eps_nu_spin", "eps_mu_spin", "P_spin")
        ):
            nc.createVariable(
                name, "f8", ("spin_time",)
            )[:] = spin_values[:, column]

        nc.createVariable(
            "umax_spin", "f8", ("spin_time",)
        )[:] = spin_umax

    nc.title = "SQG with deterministic frozen-phase ring forcing (training data with spin-up, A-SQG_data)"
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
    nc.T_start = T_START
    nc.member = MEMBER
    nc.initial_condition = f"small random field, sqg.init seed {SEED_INITIAL_CONDITION}, at T_start"
    nc.seed_initial_condition = SEED_INITIAL_CONDITION
    nc.dt_save_spin = DT_SAVE_SPIN
    nc.spinup_netcdf = str(SPIN_NETCDF_FILE)
    nc.perturbation_amp = 0.0
    nc.checkpoint_dir = str(RUN_CHECKPOINT_DIR)
    nc.checkpoint_interval = OUTPUTS_PER_CHECKPOINT * DT_SAVE
    nc.checkpoint_note = (
        "ckpt_t<time>.npz holds the full float64 AB3 state at that time; "
        "restarting from it reproduces the later records bit for bit."
    )
    nc.T_spin = T_SPIN
    nc.T_prod = T_PROD

    nc.forcing_seed = SEED_FORCING
    nc.forcing_mode_count = n_forced_modes
    nc.forcing_rms = FORCING_RMS
    nc.forcing_hash = FORCING_HASH

    nc.statistics = (
        "Equal-weight means of saved production diagnostics; "
        "spin-up excluded. Read only [:valid_records]."
    )
    nc.stationarity_note = (
        "T_spin is user chosen; assess stationarity from diagnostics."
    )
    nc.dealiasing = "circular 2/3 truncation"
    nc.forcing = (
        "f0_hat/N**2 = B*exp(i*phase) on the ring. "
        "Phases drawn once; the AB3 timestepper supplies DT."
    )
    nc.integration = (
        "AB3(advection+f0), then exact linear damping; "
        "first-order split overall"
    )
    nc.precision_model = "float64"
    nc.precision_theta_output = "float32"

    nc.summary_records = 0
    nc.valid_records = 0
    nc.written_records = 0
    nc.complete = 0

    nc.sync()
    return nc


def open_existing_netcdf():
    nc = Dataset(NETCDF_FILE, "r+")

    if getattr(nc, "config_hash", None) != CONFIG_HASH:
        nc.close()
        raise RuntimeError("Existing NetCDF does not match this configuration.")

    if getattr(nc, "forcing_hash", None) != FORCING_HASH:
        nc.close()
        raise RuntimeError("Fixed forcing differs from the saved forcing.")

    if int(getattr(nc, "aborted_for_cfl", 0)) == 1:
        nc.close()
        raise RuntimeError(
            "The previous run stopped for excessive CFL. "
            "Reduce DT while keeping NU and B fixed."
        )

    return nc


def write_block(nc, start, values, spectra, umax, fields):
    stop = start + fields.shape[0]
    indices = np.arange(start, stop)

    nc.variables["time"][start:stop] = (
        T_START + T_SPIN + (indices + 1) * DT_SAVE
    )
    nc.variables["theta"][start:stop] = fields

    for column, name in enumerate(
        ("SPE", "VIE", "eps_nu", "eps_mu", "P")
    ):
        nc.variables[name][start:stop] = values[:, column]

    nc.variables["umax"][start:stop] = umax
    nc.variables["E"][start:stop] = spectra[:, 1:]

    nc.written_records = stop
    nc.sync()
    return stop


def update_measured_statistics(nc, completed):
    """Compute both Reynolds numbers from committed production records."""

    def mean(name):
        if completed == 0:
            return np.nan
        return float(np.mean(nc.variables[name][:completed]))

    power = mean("P")
    eps_nu_mean = mean("eps_nu")
    eps_mu_mean = mean("eps_mu")

    factor = L_F**(4.0 / 3.0) / NU

    re_input = (
        factor * np.cbrt(power) if power > 0 else np.nan
    )
    re_dissipation = (
        factor * np.cbrt(eps_nu_mean)
        if eps_nu_mean >= 0 else np.nan
    )
    turnover = (
        L_F**(2.0 / 3.0) / np.cbrt(power)
        if power > 0 else np.nan
    )

    for name, value in (
        ("P_mean", power),
        ("eps_nu_mean", eps_nu_mean),
        ("Re_measured", re_input),
        ("Re_dissipation", re_dissipation),
        ("tau_f_measured", turnover),
    ):
        nc.variables[name].assignValue(value)
        nc.setncattr(name, value)

    nc.summary_records = completed
    nc.mean_input_minus_dissipation = (
        power - eps_nu_mean - eps_mu_mean
    )
    nc.kmax_lnu_measured = (
        KMAX * (NU**3 / eps_nu_mean)**0.25
        if eps_nu_mean > 0 else np.nan
    )

    # Compatibility aliases: these are measured values.
    nc.Re = re_input
    nc.epsilon_I = power
    nc.tau_f = turnover


# ---------------------------------------------------------------------------
# Full AB3 checkpoint
# ---------------------------------------------------------------------------

def qh_sha256(state):
    return hashlib.sha256(np.asarray(jax.device_get(state.state.qh)).tobytes()).hexdigest()


def save_checkpoint(state, completed_outputs):
    """Write latest.npz; on a 1-time-unit boundary it is a hard link to the archive."""
    host_state = jax.device_get(state)
    leaves, _ = jax.tree_util.tree_flatten(host_state)

    payload = {
        "config_hash": np.asarray(CONFIG_HASH),
        "forcing_hash": np.asarray(FORCING_HASH),
        "completed_outputs": np.asarray(
            completed_outputs, dtype=np.int64
        ),
        "t": np.asarray(output_time(completed_outputs), dtype=np.float64),
        "qh_sha256": np.asarray(qh_sha256(host_state)),
        "n_leaves": np.asarray(len(leaves), dtype=np.int64),
    }

    for index, leaf in enumerate(leaves):
        payload[f"leaf_{index}"] = np.asarray(leaf)

    temporary = RUN_CHECKPOINT_DIR / f".ckpt.{os.getpid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)

    if completed_outputs % OUTPUTS_PER_CHECKPOINT == 0:
        archive = archive_file(completed_outputs)
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

        leaves = [
            jnp.asarray(saved[f"leaf_{i}"])
            for i in range(n_leaves)
        ]
        completed = int(saved["completed_outputs"])
        saved_sha = str(saved["qh_sha256"].item())

    state = jax.tree_util.tree_unflatten(tree, leaves)
    if qh_sha256(state) != saved_sha:
        raise RuntimeError(f"{path} is corrupt (qh checksum mismatch).")
    return state, completed


def commit_progress(nc, state, completed):
    nc.sync()
    save_checkpoint(state, completed)
    nc.valid_records = completed
    update_measured_statistics(nc, completed)
    nc.sync()


# ---------------------------------------------------------------------------
# Interruption handling
# ---------------------------------------------------------------------------

stop_requested = False


def request_stop(signum, _frame):
    global stop_requested
    stop_requested = True
    print(
        f"Received signal {signum}; requesting a checkpoint. "
        "During spin-up, the checkpoint is written after spin-up completes.",
        flush=True,
    )


signal.signal(signal.SIGUSR1, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def check_finite_outputs(values, spectra, umax, fields=None):
    arrays = (
        (values, spectra, umax)
        if fields is None
        else (values, spectra, umax, fields)
    )
    if not all(np.isfinite(a).all() for a in arrays):
        raise FloatingPointError(
            "Non-finite output; the invalid block was not saved."
        )


# ---------------------------------------------------------------------------
# Spin-up NetCDF (theta and diagnostics every DT_SAVE_SPIN)
# ---------------------------------------------------------------------------

def create_spin_netcdf():
    nc = Dataset(SPIN_NETCDF_FILE, "w", format="NETCDF4")

    nc.createDimension("time", None)
    nc.createDimension("y", N)
    nc.createDimension("x", N)
    nc.createDimension("wavenumber", nb - 1)

    nc.createVariable("x", "f8", ("x",))[:] = np.arange(N) * L / N
    nc.createVariable("y", "f8", ("y",))[:] = np.arange(N) * L / N
    nc.createVariable(
        "wavenumber", "i4", ("wavenumber",)
    )[:] = np.arange(1, nb, dtype=np.int32)

    tvar = nc.createVariable("time", "f8", ("time",))
    tvar.long_name = "model time since the random initial condition"

    nc.createVariable(
        "theta",
        "f4",
        ("time", "y", "x"),
        zlib=True,
        complevel=1,
        shuffle=True,
        chunksizes=(1, min(256, N), min(256, N)),
        fill_value=False,
    ).long_name = "surface buoyancy"

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
        "E",
        "f8",
        ("time", "wavenumber"),
        zlib=True,
        complevel=1,
        shuffle=True,
        fill_value=False,
    ).long_name = "instantaneous isotropic SPE spectrum"

    nc.title = "SQG spin-up from a random initial condition (A-SQG_data)"
    nc.run_label = RUN_LABEL
    nc.config_json = CONFIG_JSON
    nc.config_hash = CONFIG_HASH
    nc.production_netcdf = str(NETCDF_FILE)
    nc.N = N
    nc.L = L
    nc.nu = NU
    nc.B = B
    nc.mu = MU
    nc.kf = KF
    nc.dkf = DKF
    nc.l_f = L_F
    nc.dt = DT
    nc.dt_save_spin = DT_SAVE_SPIN
    nc.T_start = T_START
    nc.T_spin = T_SPIN
    nc.member = MEMBER
    nc.seed_initial_condition = SEED_INITIAL_CONDITION
    nc.forcing_seed = SEED_FORCING
    nc.forcing_hash = FORCING_HASH
    nc.written_records = 0
    nc.complete = 0
    nc.sync()
    return nc


def write_spin_block(nc, start, values, spectra, umax, fields):
    stop = start + fields.shape[0]
    nc.variables["time"][start:stop] = (
        T_START + (np.arange(start, stop) + 1) * DT_SAVE_SPIN
    )
    nc.variables["theta"][start:stop] = fields
    for column, name in enumerate(("SPE", "VIE", "eps_nu", "eps_mu", "P")):
        nc.variables[name][start:stop] = values[:, column]
    nc.variables["umax"][start:stop] = umax
    nc.variables["E"][start:stop] = spectra[:, 1:]
    nc.written_records = stop
    nc.sync()
    return stop


# ---------------------------------------------------------------------------
# Verification: rerun one checkpoint interval from an archived checkpoint
# ---------------------------------------------------------------------------

def verify(t_verify):
    """Rerun OUTPUTS_PER_CHECKPOINT outputs from ckpt_t{t_verify} on this GPU and
    compare the end state with the next archive and the fields with the NetCDF."""
    # t_verify is typed by hand: round to the nearest saved output.
    steps = (t_verify - T_START - T_SPIN) / DT_SAVE
    completed = int(round(steps))
    if abs(steps - completed) > 1e-6:
        raise ValueError(f"t={t_verify:g} is not a saved output time.")
    if completed % OUTPUTS_PER_CHECKPOINT:
        raise ValueError(f"t={t_verify:g} is not an archived checkpoint time.")
    stop = completed + OUTPUTS_PER_CHECKPOINT
    source, target = archive_file(completed), archive_file(stop)
    for path in (source, target):
        if not path.exists():
            raise FileNotFoundError(path)

    template_state = sqg.init(p, jax.random.key(SEED_INITIAL_CONDITION))
    state, saved_completed = load_checkpoint(template_state, source)
    if saved_completed != completed:
        raise RuntimeError(f"{source} holds record {saved_completed}, not {completed}.")
    reference, _ = load_checkpoint(template_state, target)

    print(f"Verify: {source.name} -> {target.name} ({OUTPUTS_PER_CHECKPOINT} outputs)", flush=True)
    fields_equal = True
    with Dataset(NETCDF_FILE) as nc:
        nc.set_auto_mask(False)
        if int(nc.valid_records) < stop:
            raise RuntimeError(f"NetCDF holds only {int(nc.valid_records)} valid records.")
        for first in range(completed, stop, OUTPUTS_PER_BLOCK):
            state, result = run_production_block(state)
            fields = np.asarray(result[3])
            saved = nc.variables["theta"][first:first + OUTPUTS_PER_BLOCK]
            fields_equal &= np.array_equal(fields, saved)

    new_leaves = jax.tree_util.tree_leaves(jax.device_get(state))
    ref_leaves = jax.tree_util.tree_leaves(jax.device_get(reference))
    state_equal = all(np.array_equal(a, b) for a, b in zip(new_leaves, ref_leaves))
    diff = np.max(np.abs(np.asarray(state.state.qh) - np.asarray(reference.state.qh)))
    print(
        f"AB3 state bitwise equal: {state_equal} (max |dqh| = {diff:.3e}); "
        f"theta records bitwise equal: {fields_equal}",
        flush=True,
    )
    if not (state_equal and fields_equal):
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if VERIFY_T is not None:
        verify(VERIFY_T)
        return

    if OVERWRITE:
        NETCDF_FILE.unlink(missing_ok=True)
        for path in RUN_CHECKPOINT_DIR.glob("*.npz"):
            path.unlink()

    template_state = sqg.init(
        p, jax.random.key(SEED_INITIAL_CONDITION)
    )

    if RESUME and CHECKPOINT_FILE.exists():
        if not NETCDF_FILE.exists():
            raise RuntimeError("Checkpoint exists but NetCDF is missing.")

        state, completed = load_checkpoint(template_state)
        nc = open_existing_netcdf()

        if (
            not 0 <= completed <= N_PROD_OUTPUTS
            or len(nc.dimensions["time"]) < completed
        ):
            nc.close()
            raise RuntimeError("Checkpoint and NetCDF record counts disagree.")

        # Extra rows from an interrupted, uncommitted block are overwritten.
        nc.valid_records = completed
        nc.written_records = completed
        update_measured_statistics(nc, completed)
        nc.sync()

        if (
            completed == N_PROD_OUTPUTS
            and int(getattr(nc, "complete", 0)) == 1
        ):
            print("Run is already complete.", flush=True)
            nc.close()
            return

        if completed % OUTPUTS_PER_BLOCK:
            nc.close()
            raise RuntimeError("Checkpoint is not aligned with an output block.")

        print(
            f"Resuming at production record {completed}/{N_PROD_OUTPUTS}, "
            f"t={T_START + T_SPIN + completed * DT_SAVE:.2f}",
            flush=True,
        )

    else:
        if NETCDF_FILE.exists() or CHECKPOINT_FILE.exists():
            raise FileExistsError(
                "Output already exists. Use RESUME=True or choose a new run."
            )

        wall_start = time.time()
        state = template_state
        print(
            f"Starting spin-up from the random IC (seed {SEED_INITIAL_CONDITION}) "
            f"at t={T_START:g}; first call includes JAX compilation.",
            flush=True,
        )

        spin_chunks = []
        # A spin-up interrupted earlier (no checkpoint) is started again.
        spin_nc = create_spin_netcdf()
        spin_written = 0
        spin_progress = tqdm(
            total=N_SPIN_OUTPUTS, unit="field", desc="spin-up", file=sys.stdout,
            mininterval=60, ncols=120,
        )

        for first in range(0, N_SPIN_OUTPUTS, OUTPUTS_PER_BLOCK):
            count = min(
                OUTPUTS_PER_BLOCK, N_SPIN_OUTPUTS - first
            )
            state, result = run_spin(state, n_outputs=count)
            chunk = jax.tree_util.tree_map(np.asarray, result)

            check_finite_outputs(*chunk)
            cfl = float(np.max(chunk[2]) * KMAX * DT)

            if cfl >= CFL_ABORT:
                raise RuntimeError(
                    f"Spin-up spectral CFL {cfl:.4f} >= {CFL_ABORT}; "
                    "reduce DT while keeping NU and B fixed."
                )

            spin_written = write_spin_block(spin_nc, spin_written, *chunk)
            spin_chunks.append(chunk[:3])
            spin_progress.set_postfix_str(
                f"t={T_START + spin_written * DT_SAVE_SPIN:.2f} SPE={chunk[0][-1, 0]:.3f} CFL={cfl:.3f}",
                refresh=False,
            )
            spin_progress.update(count)

        spin_progress.close()
        spin_nc.complete = int(spin_written == N_SPIN_OUTPUTS)
        spin_nc.close()

        if spin_chunks:
            spin_values, spin_spectra, spin_umax = (
                np.concatenate([chunk[i] for chunk in spin_chunks])
                for i in range(3)
            )

            print(
                f"Spin-up complete: t={T_START + T_SPIN:g}, "
                f"SPE={spin_values[-1, 0]:.6g}, "
                f"P={spin_values[-1, 4]:.6g}, "
                f"max sampled CFL={np.max(spin_umax) * KMAX * DT:.4f}, "
                f"wall={time.time() - wall_start:.1f}s",
                flush=True,
            )
        else:
            spin_values, spin_umax = np.empty((0, 5)), np.empty(0)

        nc = create_netcdf(spin_values, spin_umax)
        completed = 0
        commit_progress(nc, state, completed)

    wall_start = time.time()
    first_block = completed // OUTPUTS_PER_BLOCK
    max_cfl_seen = 0.0

    # Progress bar over saved fields, written to stdout (= the log): refreshed at most once a minute.
    pbar = tqdm(total=N_PROD_OUTPUTS, initial=completed, unit="field", file=sys.stdout,
                mininterval=60, ncols=120, desc=f"t={T_START + T_SPIN:g}->{T_START + T_SPIN + T_PROD:g}")

    try:
        if stop_requested:
            print("Stop requested; retaining the checkpoint.", flush=True)
            return

        for block_index in range(first_block, N_BLOCKS):
            state, result = run_production_block(state)
            values, spectra, umax, fields = (
                jax.tree_util.tree_map(np.asarray, result)
            )

            check_finite_outputs(values, spectra, umax, fields)

            block_cfl = float(np.max(umax) * KMAX * DT)

            if block_cfl >= CFL_ABORT:
                nc.aborted_for_cfl = 1
                nc.sync()
                raise RuntimeError(
                    f"Sampled spectral CFL {block_cfl:.4f} >= {CFL_ABORT}. "
                    "Reduce DT while keeping NU and B fixed. "
                    "Invalid block not saved; previous checkpoint retained."
                )

            completed = write_block(
                nc, completed, values, spectra, umax, fields
            )
            max_cfl_seen = max(max_cfl_seen, block_cfl)
            pbar.set_postfix_str(
                f"t={T_START + T_SPIN + completed * DT_SAVE:.2f} SPE={values[-1, 0]:.3f} CFL={block_cfl:.3f}",
                refresh=False,
            )
            pbar.update(fields.shape[0])

            should_checkpoint = (
                (block_index + 1) % CHECKPOINT_EVERY_BLOCKS == 0
                or completed == N_PROD_OUTPUTS
                or stop_requested
            )

            if should_checkpoint:
                commit_progress(nc, state, completed)

            if (
                (block_index + 1) % 10 == 0
                or completed == N_PROD_OUTPUTS
                or stop_requested
            ):
                elapsed = time.time() - wall_start
                done = block_index - first_block + 1
                remaining = N_BLOCKS - block_index - 1
                eta = elapsed / done * remaining

                pbar.write(
                    f"record {completed:5d}/{N_PROD_OUTPUTS}  "
                    f"t={T_START + T_SPIN + completed * DT_SAVE:8.2f}  "
                    f"SPE={values[-1, 0]:9.4f}  "
                    f"P={values[-1, 4]:10.4f}  "
                    f"CFL={block_cfl:.4f}  "
                    f"ETA={eta / 3600:.2f} h"
                )

            if block_cfl >= CFL_WARNING:
                pbar.write(f"WARNING: sampled spectral CFL={block_cfl:.4f}")

            if stop_requested:
                pbar.write(
                    f"Stopped safely at record {completed}. "
                    "Rerun A-SQG_data.sh to resume."
                )
                return

        nc.valid_records = N_PROD_OUTPUTS
        nc.complete = 1
        nc.max_spectral_cfl_last_job = max_cfl_seen
        nc.sync()

        print(
            f"Run complete.\n"
            f"P_mean          = {nc.P_mean:.9g}\n"
            f"Re_measured     = {nc.Re_measured:.9g}\n"
            f"Re_dissipation  = {nc.Re_dissipation:.9g}\n"
            f"tau_f_measured  = {nc.tau_f_measured:.9g}\n"
            f"kmax*l_nu       = {nc.kmax_lnu_measured:.6g}\n"
            f"NetCDF: {NETCDF_FILE}",
            flush=True,
        )

    finally:
        pbar.close()
        nc.close()


if __name__ == "__main__":
    main()
