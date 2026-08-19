"""Measure the real GPU-to-GPU link bandwidth on any node, portably.

WHY: `nvidia-smi topo -m` reports the *claimed* topology; this measures it. On
midway3-0423 (H100 NVL) the two disagree in the way that matters -- GPU0<->GPU1
is NVLink while GPU0<->GPU2 crosses PCIe and a NUMA boundary, and the second
costs ACE2 +29% per training step (jobs 53538838/53538839).

Run inside an allocation on any cluster (Midway, Delta, DeltaAI, Polaris):
    python gpu_topology_check.py

Reads nothing, writes nothing, needs no distributed setup: it times a
device-to-device copy between every GPU pair, which is exactly the path a
gradient all-reduce uses. Interpretation:
    hundreds of GB/s   -> NVLink / NVSwitch
    ~20-30 GB/s        -> PCIe Gen4/Gen5, i.e. NO NVLink on that pair
A matrix that is fast in 2x2 blocks and slow off-diagonal is the H100 NVL
pair-bridge pattern; uniformly fast is a proper mesh (SXM/HGX/GH200).
"""

import subprocess
import sys
import time

import torch

MB = 256  # per transfer; large enough to be bandwidth- not latency-bound


def main():
    print(f"host:  {subprocess.getoutput('hostname')}")
    print(f"torch: {torch.__version__}  cuda_built={torch.version.cuda}  "
          f"available={torch.cuda.is_available()}")
    print(f"arch:  {subprocess.getoutput('uname -m')}")
    if not torch.cuda.is_available():
        print("ERROR torch.cuda.is_available() is False -- run this INSIDE a GPU "
              "allocation, and check CUDA_VISIBLE_DEVICES is not empty")
        return
    n = torch.cuda.device_count()
    print(f"GPUs:  {n}")
    for i in range(n):
        try:
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i}  {p.name}  {p.total_memory / 2**30:.0f} GiB")
        except Exception as err:                     # noqa: BLE001
            print(f"  cuda:{i}  <properties unavailable: {type(err).__name__}: {err}>")
    print()
    topo = subprocess.getoutput("nvidia-smi topo -m")
    print(topo.split("Legend")[0] if topo else "(nvidia-smi topo -m produced nothing)")

    if n < 2:
        print(f"only {n} GPU(s) visible -- nothing to measure")
        return

    # Shrink the transfer if any visible GPU is small or already busy: a fixed
    # 256 MiB x 2 allocation is fine on a 94 GiB card but can OOM on a shared or
    # smaller one, and an OOM here looks like a tool bug rather than a full GPU.
    free = min((torch.cuda.mem_get_info(i)[0] for i in range(n)), default=0)
    mb = MB if free > 4 * MB * 1024 * 1024 else max(16, int(free / 4 / 1024 / 1024))
    if mb != MB:
        print(f"note: reducing transfer to {mb} MiB -- least-free GPU has "
              f"{free / 2**30:.1f} GiB free")
    elems = mb * 1024 * 1024 // 4
    print(f"measured pairwise copy bandwidth, {mb} MiB per transfer (GB/s):")
    print("      " + "".join(f"{j:>9}" for j in range(n)))
    for i in range(n):
        row = f"cuda:{i}"
        src = torch.empty(elems, dtype=torch.float32, device=f"cuda:{i}")
        for j in range(n):
            if i == j:
                row += f"{'-':>9}"
                continue
            try:
                dst = torch.empty(elems, dtype=torch.float32, device=f"cuda:{j}")
            except Exception as err:                 # noqa: BLE001
                # one unusable pair must not kill the whole matrix
                row += f"{type(err).__name__[:8]:>9}"
                continue
            for _ in range(3):                      # warm up the path
                dst.copy_(src)
            # BOTH devices: torch.cuda.synchronize() with no argument syncs only
            # the CURRENT device, so timing a cuda:i -> cuda:j copy that way
            # measures nothing once i != current. That bug produced impossible
            # 14,000 GB/s readings on the first run of this script.
            torch.cuda.synchronize(i)
            torch.cuda.synchronize(j)
            t0 = time.perf_counter()
            reps = 10
            for _ in range(reps):
                dst.copy_(src)
            torch.cuda.synchronize(i)
            torch.cuda.synchronize(j)
            gbps = reps * elems * 4 / (time.perf_counter() - t0) / 1e9
            row += f"{gbps:>9.1f}"
            del dst
        print(row)
        del src
    print("\nfast 2x2 blocks + slow off-diagonal = H100 NVL pair bridges;")
    print("uniformly fast = full mesh (SXM/HGX/GH200); all ~25 = no NVLink at all.")


if __name__ == "__main__":
    main()
