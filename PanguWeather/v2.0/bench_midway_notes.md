# PanguWeather SFNO on Midway — bench notes

Companion to `polaris_bench_report.md` (the A100/Polaris profile). Same model,
different cluster. **Read the hardware caveat before comparing any percentage.**

## What exists

| Piece | Path |
|---|---|
| Config | `config/E3SM_SFNO_H5_MIDWAY.yaml` — port of `E3SM_SFNO_H5_POLARIS.yaml`, paths only |
| nsys bench | `HPC_scripts/midway_bench_nsys_e3sm_sfno.sh` → `PANGU_NSYS_OK` |
| Stats prep | `midway_prepare_e3sm_stats.py` |
| Env | `/project/pedramh/shared/conda/envs/py311_pip_sfno_cu129` (torch 2.9.1+cu129) |
| Data | `/project/pedramh/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/sigma_data` |

⚠ **The normalisation in the staged stats is NOT correct for this data** (per the
data owner). These runs are **profiling only** — ignore their loss values, and do
not use them for any equivalence or convergence claim. Shapes and channel counts
are right, which is all a performance profile needs.

## Result — job 53539872, 4x H100 NVL (midway3-0423)

`PANGU_NSYS_OK`, 44 MB report, bench CSV row written.

| | **Midway H100 NVL** | **Polaris A100** (7255410) |
|---|---|---|
| step_med | **1.100 s** | 0.652 s |
| samples_per_s | **3.64** | 6.13 |
| peak mem | 26.97 GB | 26.98 GB |
| loader_wait_frac | 8.8% | 0.7% |

**Pangu is ~1.7x SLOWER on the newer H100s.** Peak memory matches to 0.01 GB, so
it is the same model at the same shape — not a config difference.

### Where the time goes

| bucket | Midway H100 | Polaris A100 |
|---|---|---|
| **NCCL (comm + wait)** | **74.2%** | **10.5%** |
| elementwise / copy | 13.5% | 61.0% |
| GEMM | 5.7% | 15.1% |

NVTX phases (median over the captured steps):

| range | n | median | min | max |
|---|---|---|---|---|
| `data_prep` | 160 | 0.5 ms | 0.2 | 2.3 |
| `forward_loss` | 160 | 46.3 ms | 42.5 | 73.7 |
| **`backward`** | 160 | **539.8 ms** | **112.2** | **993.1** |
| `optimizer` | 160 | 4.5 ms | 3.4 | 6.3 |
| step total | 156 | 995.5 ms (std 278.3) | | |

### ⚠ Read this before quoting any of the above

**Three-quarters of GPU time is gradient exchange, because of this node's
interconnect.** `midway3-0423` is H100 NVL: NVLink joins GPU pairs only
(GPU0<->1, GPU2<->3) and the pairs are joined by PCIe across a NUMA boundary —
**measured 261 vs 18 GB/s** by `gpu_topology_check.py`. A 4-GPU ring all-reduce
crosses that boundary twice. Polaris A100 nodes do not.

Pangu carries **1.18 B parameters = 4.73 GB of gradients**, 2.6x ACE2's, so it
suffers more than ACE2 did on the identical node: **74.2% NCCL vs ACE2's 52.1%**.
The same effect was isolated directly by running one job on an NVLink pair vs
across pairs: **+28.9% per step** (`ACE2_retrain/bench_midway_notes.md`).

Consequences for reading this profile:

1. **The 1.7x slowdown is the interconnect, not the GPU.** H100 compute is
   faster; the node cannot feed the all-reduce.
2. **The backward's 9x swing (112–993 ms) is all-reduce wait**, not compute
   variance — DDP overlaps gradient exchange with the backward pass, so the wait
   lands inside that range.
3. **The Polaris "61% elementwise" finding does NOT contradict this.**
   Elementwise work did not shrink; NCCL swamped the denominator. Shares are
   relative — always state the hardware alongside them.

⇒ For any Pangu optimisation work on elementwise/pointwise kernels, **profile on
Polaris or on an H200 node (NV6 full mesh)**. On this node the signal is buried
under communication.

## Port notes — three environment differences, none of them code faults

1. **`tensorly` missing.** `sfnonet.py` imports it for factorizations; the S2S
   venv lacks it (job 53539649). Fixed by using the shared SFNO env rather than
   installing into a venv S2S and the Lightning port also depend on.
2. **CUDA module vs torch build.** That env is cu129, so `module load cuda/12.6`
   would put mismatched CUDA libs ahead of torch's bundled ones. `nsys` is taken
   from an explicit `NSYS_BIN` path instead — the same pattern the Polaris script
   already uses.
3. **Stats level-dimension naming** (job 53539745, `KeyError: 'Z_2'`).
   `load_mean_std(use_sigma_levels=True)` indexes every non-`zg` variable on a
   `Z_2` coordinate; the Midway stats name that dimension `Z`, and this dataset
   calls geopotential `Z3`, so nothing took the `Z` branch.
   `midway_prepare_e3sm_stats.py` renames the dimension. Values are unchanged —
   `Z` already held the sigma levels, verified equal to the config's
   `sigma_levels` to 3 dp. It must be the DIMENSION, not a coordinate alias:
   `where(..., dims=['Z_2'])` broadcasts rather than selects otherwise.

Unlike Polaris, no `climatology.nc` conversion is needed — the Midway copy opens
directly with xarray.
