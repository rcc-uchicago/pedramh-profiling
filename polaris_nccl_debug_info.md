# Polaris NCCL `NCCL_DEBUG=INFO` evidence

Transport selection and init behaviour. Bandwidth numbers live in
[`polaris_nccl_metrics.md`](polaris_nccl_metrics.md). Every block below is copied
from a per-rank debug file under `$MEMBER_ROOT/runs/`, deduplicated across ranks
and with the `host:pid:tid [dev]` prefix stripped.

Captured with:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV,GRAPH,TUNING
# per-rank files instead of interleaved stdout -- 16 ranks into one stream is
# unreadable, and the perf table gets buried
NCCL_DEBUG_FILE="$OUT/<arm>.nccl.%h.%p"
```

---

## 1. The stack

NCCL cannot address Slingshot directly. The chain is

```
NCCL -> aws-ofi-nccl plugin -> libfabric -> cxi provider -> Slingshot 11 NIC
```

so `NCCL_NET` selects the plugin, and the plugin — not any NCCL variable — decides
whether CXI is usable. That is the whole story of this document.

| component | version |
|---|---|
| NCCL | 2.28.3+cuda12.9 (`/soft/libraries/nccl/nccl_2.28.3-1+cuda12.9_x86_64`) |
| libfabric | 2.3.1 (`/opt/cray/libfabric/2.3.1`) — the only version installed |
| plugin (works) | aws-ofi-nccl **1.6.0** (`/soft/libraries/aws-ofi-nccl/v1.6.0-libfabric-1.22.0`) |
| plugin (fails) | aws-ofi-nccl **1.21.1**, self-built against libfabric 2.3.1 |
| CUDA | toolkit 12.9.1; driver 13000, runtime 12090 |

`fi_info -p cxi` on a compute node — Slingshot **is** visible to libfabric, which is
what makes the plugin failure below a negotiation problem rather than a hardware or
driver one:

```
provider: cxi   fabric: cxi   domain: cxi0   version: 0.1   type: FI_EP_RDM
provider: cxi   fabric: cxi   domain: cxi1   version: 0.1   type: FI_EP_RDM
mr_mode: [ FI_MR_ALLOCATED, FI_MR_PROV_KEY, FI_MR_ENDPOINT ]
```

## 2. WORKING — aws-ofi-nccl 1.6.0, 16 ranks / 4 nodes

Job 7629096, `B_4node_allreduce.nccl.x3005c0s25b0n0.2125405`.

```
NCCL INFO NCCL version 2.28.3+cuda12.9
NCCL INFO NET/Plugin: Loaded net plugin AWS Libfabric (v6)
NCCL INFO NET/OFI Using aws-ofi-nccl 1.6.0
NCCL INFO NET/OFI Selected Provider is cxi (found 2 nics)
NCCL INFO Using network AWS Libfabric
NCCL INFO ncclCommInitRankConfig comm 0x79ec210 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 7000 commId 0xecab91cee5913cc9 - Init START
NCCL INFO NCCL_CROSS_NIC set by environment to 1.
NCCL INFO comm 0x79ec210 rank 0 nRanks 16 nNodes 4 localRanks 4 localRank 0 MNNVL 0
NCCL INFO Channel 00/04 :  0  2  1  7  4  6  5 11  8 10  9 15 12 14 13  3
NCCL INFO Channel 01/04 :  0  2  3  5  4  6  7  9  8 10 11 13 12 14 15  1
NCCL INFO Channel 02/04 :  0  2  1  7  4  6  5 11  8 10  9 15 12 14 13  3
NCCL INFO Channel 03/04 :  0  2  3  5  4  6  7  9  8 10 11 13 12 14 15  1
NCCL INFO Trees [0] -1/-1/-1->0->2 [1] 3/-1/-1->0->1 [2] -1/-1/-1->0->2 [3] 3/-1/-1->0->1
NCCL INFO NCCL_PROTO set by environment to Simple
NCCL INFO ncclCommInitRankConfig comm 0x79ec210 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 7000 commId 0xecab91cee5913cc9 - Init COMPLETE
NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 16 total 0.19 (kernels 0.09, alloc 0.04, bootstrap 0.03, allgathers 0.01, topo 0.02, graphs 0.00, connections 0.01, rest 0.00)
NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM/read
NCCL INFO Channel 03/0 : 0[0] -> 1[1] via P2P/CUMEM/read
NCCL INFO Channel 00/0 : 0[0] -> 2[2] via P2P/CUMEM/read
NCCL INFO Channel 02/0 : 0[0] -> 2[2] via P2P/CUMEM/read
NCCL INFO Channel 01/0 : 0[0] -> 3[3] via P2P/CUMEM/read
NCCL INFO Channel 03/0 : 0[0] -> 3[3] via P2P/CUMEM/read
NCCL INFO Channel 01/0 : 0[0] -> 2[2] via P2P/CUMEM/read
NCCL INFO Channel 03/0 : 0[0] -> 2[2] via P2P/CUMEM/read
NCCL INFO Connected all rings, use ring PXN 0 GDR 1
NCCL INFO comm 0x79ec210 rank 0 nranks 16 cudaDev 0 busId 7000 - Destroy COMPLETE
```

Three things to notice, all of which cost bandwidth:

* `Loaded net plugin AWS Libfabric (v6)` — the plugin implements **net API v6**
  while this NCCL supports **v11**. The configuration that works is the oldest one.
* `Channel 00/04`..`03/04` — only **4 channels** across 16 ranks with 2 NICs/node.
* `use ring PXN 0` — PXN is **off**, so no PCI-x-NVLink aggregation of NIC traffic.

`GDR 1` confirms GPUDirect RDMA is active, so the 28.5 GB/s plateau is not a
staging-through-host-memory artefact.

## 3. FAILING — aws-ofi-nccl 1.21.1, same nodes, same NCCL

Job 7629065, `B_4node_allgather.nccl.x3002c0s31b0n0.3333526`. This is the version HPE's own `shs-ccl-docs`
recommends as its default.

```
NCCL INFO NCCL version 2.28.3+cuda12.9
NCCL INFO NET/Plugin: Loaded net plugin AWS Libfabric (v11)
NCCL INFO NET/OFI Initializing aws-ofi-nccl 1.21.1
NCCL INFO NET/OFI Using Libfabric version 2.3
NCCL INFO NET/OFI Using CUDA driver version 13000 with runtime 12090
NCCL INFO NET/OFI Plugin selected platform: Default
NCCL INFO NET/OFI Requesting progress model AUTO
NCCL INFO NET/OFI No eligible providers were found
[2026-09-17 04:02:30] x3002c0s31b0n0:3333526:3333526 [0] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):334 NCCL WARN NET/OFI Unable to find a protocol that worked.  Failing initialization.
[2026-09-17 04:02:30] x3002c0s31b0n0:3333526:3333526 [0] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):418 NCCL WARN NET/OFI aws-ofi-nccl initialization failed
[2026-09-17 04:02:30] x3002c0s31b0n0:3333526:3333526 [0] ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t):79 NCCL WARN NET/OFI Initializing plugin failed
[2026-09-17 04:02:30] x3002c0s31b0n0:3333526:3333526 [0] ncclResult_t nccl_net_ofi_fini():96 NCCL WARN NET/OFI Finalizing already finalized plugin
[2026-09-17 04:02:30] x3002c0s31b0n0:3333526:3333526 [0] plugin/net.cc:306 NCCL WARN Failed to initialize any NET plugin
```

The plugin loads (as **v11**), finds libfabric 2.3, then rejects every provider
before CXI is ever reached.

## 4. WHY — libfabric's own verdict

Job 7629082, arm M1, run with `FI_LOG_LEVEL=info FI_LOG_PROV=cxi`. The CXI
provider states the mismatch outright:

```
core:core:cuda_hmem_detect_dmabuf_support():681<info> cuda dmabuf support status: 0
cxi:core:ofi_check_mr_mode():611<info> Invalid memory registration mode
cxi:core:ofi_check_mr_mode():612<info> Expected: FI_MR_ALLOCATED, FI_MR_PROV_KEY, FI_MR_ENDPOINT
cxi:core:ofi_check_mr_mode():612<info> Given: FI_MR_ALLOCATED, FI_MR_ENDPOINT
cxi:core:ofi_check_info():1157<info> Unsupported capabilities
cxi:core:ofi_check_info():1158<info> Supported: FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_COLLECTIVE, FI_READ, FI_WRITE, FI_RECV, FI_SEND, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_MULTI_RECV, FI_TRIGGER, FI_FENCE, FI_SOURCE_ERR, FI_LOCAL_COMM, FI_REMOTE_COMM, FI_RMA_EVENT, FI_SOURCE, FI_NAMED_RX_CTX, FI_AV_USER_ID, FI_PEER, FI_HMEM
cxi:core:ofi_check_info():1158<info> Requested: FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_COLLECTIVE, FI_READ, FI_WRITE, FI_RECV, FI_SEND, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_MULTI_RECV, FI_TRIGGER, FI_FENCE, FI_LOCAL_COMM, FI_REMOTE_COMM, FI_RMA_EVENT, FI_NAMED_RX_CTX, FI_DIRECTED_RECV, FI_AV_USER_ID, FI_PEER, FI_HMEM
cxi:core:ofi_check_ep_attr():882<info> Tag size exceeds supported size
cxi:core:ofi_check_ep_attr():883<info> Supported: 733007751850
cxi:core:ofi_check_ep_attr():883<info> Requested: 211106299641855
cxi:core:ofi_check_info():1158<info> Requested: FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_COLLECTIVE, FI_READ, FI_WRITE, FI_RECV, FI_SEND, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_MULTI_RECV, FI_TRIGGER, FI_FENCE, FI_LOCAL_COMM, FI_REMOTE_COMM, FI_NAMED_RX_CTX, FI_DIRECTED_RECV, FI_AV_USER_ID, FI_PEER
cxi:core:ofi_check_ep_attr():768<info> Unsupported protocol
```

The plugin's `fi_getinfo` hints omit **`FI_MR_PROV_KEY`** — it expects to supply its
own memory-registration keys (the EFA model), while CXI **requires**
provider-generated keys. `mr_mode` hints are set in plugin source, so no `FI_*`
environment variable can add the missing bit. That is exactly why forcing
`FI_PROVIDER=cxi` and both `OFI_NCCL_PROTOCOL` values still failed
(metrics file section 5, arms M2-M4).

Both plugins resolve the **same** runtime library —

```
libnccl-net.so (v1.6.0)  -> libfabric.so.1 => /opt/cray/libfabric/2.3.1/lib64/libfabric.so.1
libnccl-net.so (1.21.1)  -> libfabric.so.1 => /opt/cray/libfabric/2.3.1/lib64/libfabric.so.1
```

— so this is a **plugin-version** difference, not a library-loading difference.

## 5. `NCCL_PROTO=Simple` is currently mandatory

With the v1.6.0 plugin at **>=3 nodes**, leaving the protocol at default deadlocks
during setup; pinning `Simple` clears it. The log confirms the pin took:

```
NCCL INFO NCCL_PROTO set by environment to Simple
```

The cost is real: disabling LL/LL128 measured ~26% slower on the single-node arm
of our trainer (114.9 -> 144.7 ms/step). We are trading latency-optimised protocols
for fabric stability.

Even with the pin, **`all_gather` at 16 ranks stalls at 512 KB** while `all_reduce`
completes the full sweep to 4 GiB (metrics file section 4) — so `Simple` is a
mitigation, not a fix.

## 6. Against HPE's published guidance

`github.com/HewlettPackard/shs-ccl-docs` (`ccl_env.sh`, `nccl/build_nccl_environment.sh`)
differs from what we run, in ways worth closing:

| item | HPE | here |
|---|---|---|
| aws-ofi-nccl | `v1.21.1` | 1.21.1 cannot init; forced back to **1.6.0** |
| libfabric to build against | `/opt/cray/libfabric/**1.22.0**` | only **2.3.1** installed on Polaris |
| `NCCL_NET` | `"OFI"` | `"AWS Libfabric"` (v6 plugin registers under that name) |
| CXI rendezvous | `FI_CXI_RDZV_PROTO=alt_read`, `RDZV_EAGER_SIZE=0`, `RDZV_THRESHOLD=0`, `RDZV_GET_MIN=0`, `DEFAULT_TX_SIZE=2048`, `RX_MATCH_MODE=hybrid` | **not set** |
| launcher | "When running with PBS, the flag `--disable_rdzv_get` is required" | **not passed** |

The rendezvous block and `--disable_rdzv_get` are now available in
`polaris_nccl_tests.pbs` behind `-v HPE_ENV=1`; the measurement of that arm is still
outstanding. The libfabric-version gap is not ours to close — 1.22.0 is not
installed on this machine.

## 7. Open questions

1. Which aws-ofi-nccl is supported with **NCCL 2.28.3 + libfabric 2.3.1** on
   Slingshot? We are running a **v6-API plugin under a v11-API NCCL**. What is lost,
   and is there a 1.21.x that negotiates CXI's `FI_MR_PROV_KEY` requirement?
2. Can the `NCCL_PROTO=Simple` requirement be removed (section 5)?
3. Is **4 channels / PXN 0** expected at 16 ranks with 2 NICs/node, and would PXN
   lift the 28.5 GB/s plateau?
4. `cuda dmabuf support status: 0` while `GDR 1` — does the missing dmabuf path
   constrain registration or achievable inter-node bandwidth here?
5. Why does `all_gather` wedge at 512 KB where `all_reduce` completes?

## 8. Job index

| job | configuration | outcome |
|---|---|---|
| 7629065 | 4 nodes, self-built 1.21.1 | init failure (section 3) |
| 7629082 | 2 nodes, 6-arm plugin matrix + `FI_LOG` | root cause (section 4) |
| 7629096 | 4 nodes, v1.6.0 + `NCCL_PROTO=Simple` | all_reduce OK, all_gather wedged |

Raw logs: `/eagle/projects/lighthouse-uchicago/members/mehta5/runs/`.
