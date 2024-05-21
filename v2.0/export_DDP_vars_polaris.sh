export RANK=$PBS_VNODENUM
export WORLD_RANK=$PBS_VNODENUM
export LOCAL_RANK=$PBS_NUM_NODES
export WORLD_SIZE=$PBS_NP
export MASTER_ADDR='localhost' ####
export MASTER_PORT=29500 # default from torch launcher
export WANDB_START_METHOD="thread"
