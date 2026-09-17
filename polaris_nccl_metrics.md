# Polaris NCCL metrics — NVIDIA `nccl-tests` v2.20.0

Bandwidth and correctness only; the `NCCL_DEBUG=INFO` transport evidence lives in
[`polaris_nccl_debug_info.md`](polaris_nccl_debug_info.md). Both are generated from
the raw logs under `$MEMBER_ROOT/runs/`, not retyped.

Measured 2026-09-17 on ALCF Polaris (4x A100-SXM4-**40GB**/node, HPE Slingshot 11,
2 NICs/node). NCCL **2.28.3+cuda12.9**, libfabric **2.3.1**, nccl-tests built
`MPI=1 NVCC_GENCODE=sm_80` against that NCCL and cray-mpich 9.1.0.

`busbw` is the number to compare across rank counts; `algbw` is not
(nccl-tests scales busbw by the collective's communication volume). `#wrong` is
the correctness column and is the PASS gate here — an exit code is not, because a
killed run still returns 0 through a pipe.

---

## 1. Summary

| arm | ranks x nodes | transport | peak busbw | avg busbw | correctness |
|---|---|---|---|---|---|
| all_reduce | 4 x 1 | NVLink (plugin not involved) | **205.03 GB/s** | 63.1457 | `0 OK` |
| all_reduce | 16 x 4 | Slingshot / cxi, aws-ofi-nccl **1.6.0** | **30.75 GB/s** | 10.4127 | `0 OK` |
| all_gather | 16 x 4 | Slingshot / cxi, aws-ofi-nccl 1.6.0 | — | — | **WEDGED at 512 KB** (section 4) |

Inter-node sustains ~28.5 GB/s against a 205.03 GB/s intra-node ceiling — a
**7.2x step down** the moment a collective crosses a node
boundary.

## 2. `all_reduce_perf` — 4 ranks, 1 node (NVLink only)

Job 7629096, arm A. The OFI plugin is not in this path at all, so this is the
intra-node ceiling every multi-node number should be read against.

```
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)        
           8             2     float     sum      -1    39.60    0.00    0.00       0    38.24    0.00    0.00       0
          16             4     float     sum      -1    38.86    0.00    0.00       0    37.60    0.00    0.00       0
          32             8     float     sum      -1    40.87    0.00    0.00       0    40.53    0.00    0.00       0
          64            16     float     sum      -1    45.76    0.00    0.00       0    45.19    0.00    0.00       0
         128            32     float     sum      -1    45.54    0.00    0.00       0    59.58    0.00    0.00       0
         256            64     float     sum      -1    45.36    0.01    0.01       0    45.28    0.01    0.01       0
         512           128     float     sum      -1    45.36    0.01    0.02       0    46.72    0.01    0.02       0
        1024           256     float     sum      -1    45.39    0.02    0.03       0    45.66    0.02    0.03       0
        2048           512     float     sum      -1    44.65    0.05    0.07       0    45.11    0.05    0.07       0
        4096          1024     float     sum      -1    45.48    0.09    0.14       0    44.90    0.09    0.14       0
        8192          2048     float     sum      -1    46.08    0.18    0.27       0    45.88    0.18    0.27       0
       16384          4096     float     sum      -1    51.68    0.32    0.48       0    52.08    0.31    0.47       0
       32768          8192     float     sum      -1    58.44    0.56    0.84       0    58.24    0.56    0.84       0
       65536         16384     float     sum      -1    62.03    1.06    1.58       0    60.78    1.08    1.62       0
      131072         32768     float     sum      -1    62.54    2.10    3.14       0    62.56    2.10    3.14       0
      262144         65536     float     sum      -1    63.57    4.12    6.19       0    63.24    4.15    6.22       0
      524288        131072     float     sum      -1    65.05    8.06   12.09       0    64.88    8.08   12.12       0
     1048576        262144     float     sum      -1    68.54   15.30   22.95       0    68.41   15.33   22.99       0
     2097152        524288     float     sum      -1    75.82   27.66   41.49       0    75.73   27.69   41.54       0
     4194304       1048576     float     sum      -1    90.20   46.50   69.75       0    88.50   47.40   71.09       0
     8388608       2097152     float     sum      -1   116.69   71.89  107.83       0   114.08   73.53  110.30       0
    16777216       4194304     float     sum      -1   195.46   85.83  128.75       0   193.46   86.72  130.08       0
    33554432       8388608     float     sum      -1   317.64  105.64  158.46       0   317.82  105.58  158.36       0
    67108864      16777216     float     sum      -1   599.52  111.94  167.91       0   596.31  112.54  168.81       0
   134217728      33554432     float     sum      -1  1119.32  119.91  179.87       0  1115.34  120.34  180.51       0
   268435456      67108864     float     sum      -1  2141.59  125.34  188.02       0  2139.27  125.48  188.22       0
   536870912     134217728     float     sum      -1  4131.40  129.95  194.92       0  4141.22  129.64  194.46       0
  1073741824     268435456     float     sum      -1  8106.60  132.45  198.68       0  8109.02  132.41  198.62       0
  2147483648     536870912     float     sum      -1  15895.0  135.10  202.66       0  15893.3  135.12  202.68       0
  4294967296    1073741824     float     sum      -1  31422.1  136.69  205.03       0  31432.5  136.64  204.96       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 63.1457 
```

## 3. `all_reduce_perf` — 16 ranks, 4 nodes (over Slingshot)

Job 7629096, arm B. `NCCL_PROTO=Simple` pinned (see debug-info file §5 for why
that pin is currently mandatory).

```
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)        
           8             2     float     sum      -1   142.18    0.00    0.00       0   137.47    0.00    0.00       0
          16             4     float     sum      -1   133.38    0.00    0.00       0   135.14    0.00    0.00       0
          32             8     float     sum      -1   133.94    0.00    0.00       0   132.06    0.00    0.00       0
          64            16     float     sum      -1   131.14    0.00    0.00       0   132.50    0.00    0.00       0
         128            32     float     sum      -1   133.82    0.00    0.00       0   133.74    0.00    0.00       0
         256            64     float     sum      -1   133.38    0.00    0.00       0   132.84    0.00    0.00       0
         512           128     float     sum      -1   132.74    0.00    0.01       0   130.10    0.00    0.01       0
        1024           256     float     sum      -1   132.57    0.01    0.01       0   131.64    0.01    0.01       0
        2048           512     float     sum      -1   132.28    0.02    0.03       0   133.10    0.02    0.03       0
        4096          1024     float     sum      -1   136.21    0.03    0.06       0   136.13    0.03    0.06       0
        8192          2048     float     sum      -1   141.48    0.06    0.11       0   140.18    0.06    0.11       0
       16384          4096     float     sum      -1   150.42    0.11    0.20       0   149.31    0.11    0.21       0
       32768          8192     float     sum      -1   183.23    0.18    0.34       0   180.38    0.18    0.34       0
       65536         16384     float     sum      -1   187.09    0.35    0.66       0   183.50    0.36    0.67       0
      131072         32768     float     sum      -1   187.76    0.70    1.31       0   185.28    0.71    1.33       0
      262144         65536     float     sum      -1   226.42    1.16    2.17       0   221.98    1.18    2.21       0
      524288        131072     float     sum      -1   284.93    1.84    3.45       0   280.36    1.87    3.51       0
     1048576        262144     float     sum      -1   412.68    2.54    4.76       0   411.99    2.55    4.77       0
     2097152        524288     float     sum      -1   663.86    3.16    5.92       0   659.01    3.18    5.97       0
     4194304       1048576     float     sum      -1   752.79    5.57   10.45       0   750.39    5.59   10.48       0
     8388608       2097152     float     sum      -1   649.48   12.92   24.22       0   649.33   12.92   24.22       0
    16777216       4194304     float     sum      -1  1090.55   15.38   28.85       0  1179.42   14.23   26.67       0
    33554432       8388608     float     sum      -1  2045.76   16.40   30.75       0  2099.26   15.98   29.97       0
    67108864      16777216     float     sum      -1  4311.90   15.56   29.18       0  4314.47   15.55   29.16       0
   134217728      33554432     float     sum      -1  8776.81   15.29   28.67       0  8860.42   15.15   28.40       0
   268435456      67108864     float     sum      -1  17560.8   15.29   28.66       0  17598.6   15.25   28.60       0
   536870912     134217728     float     sum      -1  35232.5   15.24   28.57       0  35217.0   15.24   28.58       0
  1073741824     268435456     float     sum      -1  70570.3   15.22   28.53       0  70527.4   15.22   28.55       0
  2147483648     536870912     float     sum      -1   141223   15.21   28.51       0   141004   15.23   28.56       0
  4294967296    1073741824     float     sum      -1   283197   15.17   28.44       0   282835   15.19   28.47       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 10.4127 
```

Bandwidth peaks at 32 MB and then *falls back* to a ~28.5 GB/s plateau, rather
than continuing to climb as the intra-node arm does to 4 GiB.

## 4. `all_gather_perf` — 16 ranks, 4 nodes — INCOMPLETE

Same job, same communicator settings, same plugin. all_reduce completed the full
sweep to 4 GiB; all_gather stopped producing rows after 262144 B and the log did
not grow again before walltime. Recorded as an open failure, **not** as a
measurement — an arm with no terminating `Out of bounds values` line never
finished, and quoting its partial rows as a result would be wrong.

```
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)        
           0             0     float    none      -1     0.50    0.00    0.00       0     0.41    0.00    0.00       0
           0             0     float    none      -1     0.43    0.00    0.00       0     0.42    0.00    0.00       0
           0             0     float    none      -1     0.43    0.00    0.00       0     0.39    0.00    0.00       0
           0             0     float    none      -1     0.40    0.00    0.00       0     0.42    0.00    0.00       0
           0             0     float    none      -1     0.42    0.00    0.00       0     0.47    0.00    0.00       0
         256             4     float    none      -1   157.93    0.00    0.00       0   153.55    0.00    0.00       0
         512             8     float    none      -1   154.51    0.00    0.00       0   199.56    0.00    0.00       0
        1024            16     float    none      -1   156.99    0.01    0.01       0   154.02    0.01    0.01       0
        2048            32     float    none      -1   160.83    0.01    0.01       0   159.96    0.01    0.01       0
        4096            64     float    none      -1   154.88    0.03    0.02       0   152.98    0.03    0.03       0
        8192           128     float    none      -1   155.35    0.05    0.05       0   153.33    0.05    0.05       0
       16384           256     float    none      -1   159.81    0.10    0.10       0   159.53    0.10    0.10       0
       32768           512     float    none      -1   172.01    0.19    0.18       0   168.82    0.19    0.18       0
       65536          1024     float    none      -1   181.23    0.36    0.34       0   176.00    0.37    0.35       0
      131072          2048     float    none      -1   206.22    0.64    0.60       0   203.26    0.64    0.60       0
      262144          4096     float    none      -1   206.36    1.27    1.19       0   205.52    1.28    1.20       0
```

## 5. Plugin / transport comparison — 8 ranks, 2 nodes

Job 7629082. Short sweep (`-b 8 -e 64M -f 4`) on identical nodes with exactly one
variable changed per arm. The question is whether init succeeds, not what the
bandwidth is, so each arm costs seconds.

```
M1_baseline_filog FAIL -
M2_fi_provider_cxi FAIL -
M3_sendrecv FAIL -
M4_rdma FAIL -
M5_soft_plugin_v1_6_0 OK 18.03
M6_socket_control OK 3.46
```

| arm | what changed | result |
|---|---|---|
| M1 | self-built aws-ofi-nccl **1.21.1** | FAIL — `No eligible providers were found` |
| M2 | 1.21.1 + `FI_PROVIDER=cxi` | FAIL — identical |
| M3 | 1.21.1 + `OFI_NCCL_PROTOCOL=SENDRECV` | FAIL |
| M4 | 1.21.1 + `OFI_NCCL_PROTOCOL=RDMA` | FAIL |
| M5 | `/soft` aws-ofi-nccl **v1.6.0** | **OK — 18.03 GB/s**, `Selected Provider is cxi` |
| M6 | `NCCL_NET=Socket` (TCP control) | OK — 3.46 GB/s |

M6 is the control that makes M1-M4 interpretable: multi-node NCCL itself is
healthy on these nodes, so the failure is confined to the OFI path. CXI is
**5.2x** TCP, which is the size of the prize for getting the plugin right.

## 6. Reproduction

```bash
# 1-node + N-node sweep; OFI_PLUGIN / NCCL_PROTO / HPE_ENV pass through
qsub -v OFI_PLUGIN=/soft/libraries/aws-ofi-nccl/v1.6.0-libfabric-1.22.0/lib,NCCL_PROTO=Simple \
     polaris_nccl_tests.pbs
# the 6-arm plugin matrix of section 5
qsub polaris_nccl_ofi_matrix.pbs
```

Scripts: `/eagle/projects/lighthouse-uchicago/members/mehta5/sw/nccl-tests-polaris/`.
Raw logs: `$MEMBER_ROOT/runs/nccl_tests/<jobid>/` and `runs/nccl_ofi_matrix/<jobid>/`
(`<arm>.log` = tables, `<arm>.nccl.<host>.<pid>` = per-rank `NCCL_DEBUG=INFO`).
