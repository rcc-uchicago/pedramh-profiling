#!/bin/bash -l
#SBATCH --account=m4416
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH -o outs/%x_%j.out
#SBATCH -e outs/%x_%j.err

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export HDF5_USE_FILE_LOCKING=FALSE


#./home/awikner/miniconda3/bin/conda init; bash
module load conda
conda activate aires_panguplasim
#export cuda_version=12.1
#export CUDA_HOME=/usr/local/cuda-${cuda_version}
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#export PATH=$CUDA_HOME/bin:$PATH

# source activate /scratch/midway3/tvallabh/tarun_pangu



# Change to working directory
cd /pscratch/sd/a/awikner/PanguWeather-ens/v2.0
#source export_DDP_vars.sh
which conda
#python test_torch.py
export WANDB_MODE=offline

nvidia-smi

# MPI and OpenMP settings
NNODES=`wc -l < $SLURM_JOB_NODELIST`
#Follwing will be the number of GPUs on each node, so 4 in our case as each node has 4 GPUs
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#NUM_TASKS_PER_NODE=2
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

# Set up the PyTorch distributed environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE
export RANK=$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=1

#init_nc_filepaths=/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_0/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_1/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_2/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_3/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_4/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_5/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_6/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_7/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_8/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_9/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_10/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_11/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_12/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_13/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_14/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_15/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_16/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_17/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_18/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_19/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_20/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_21/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_22/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_23/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_24/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_25/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_26/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_27/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_28/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_29/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_30/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_31/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_32/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_33/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_34/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_35/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_36/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_37/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_38/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_39/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_40/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_41/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_42/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_43/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_44/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_45/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_46/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_47/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_48/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_49/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_50/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_51/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_52/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_53/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_54/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_55/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_56/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_57/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_58/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_59/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_60/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_61/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_62/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_63/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_64/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_65/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_66/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_67/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_68/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_69/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_70/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_71/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_72/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_73/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_74/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_75/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_76/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_77/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_78/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_79/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_80/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_81/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_82/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_83/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_84/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_85/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_86/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_87/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_88/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_89/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_90/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_91/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_92/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_93/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_94/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_95/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_96/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_97/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_98/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_99/test_data_postproc.nc
init_nc_filepaths=/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_0/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_1/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_2/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_3/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_4/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_5/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_6/test_data_postproc.nc,/pscratch/sd/a/awikner/PLASIM/data/test_data/copy_7/test_data_postproc.nc


# Launch your script using torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=$NUM_TASKS_PER_NODE ensemble_inference.py --yaml_config=$2 --run_num=$1 --init_nc_filepaths=$init_nc_filepaths --async_save
# --enable_amp if needed
