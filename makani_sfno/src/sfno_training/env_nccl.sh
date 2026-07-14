# env_nccl.sh — sourced by multi-GPU SLURM scripts.
#
# NCCL_DEBUG: log level only; no behavioural change. Default WARN is
#   quiet; set NCCL_DEBUG=INFO at submit time for collective hangs.
# NCCL_ASYNC_ERROR_HANDLING / TORCH_NCCL_ASYNC_ERROR_HANDLING:
#   *behavioural*. With these set to 1, NCCL aborts the process on an
#   async error (e.g. a peer rank dying) instead of hanging
#   indefinitely. This is what we want for SLURM jobs — failures
#   surface as job exits within minutes rather than walltime exhausts.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
# export NCCL_DEBUG_SUBSYS=COLL    # uncomment when debugging collectives
