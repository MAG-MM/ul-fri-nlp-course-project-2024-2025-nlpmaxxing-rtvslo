#!/bin/bash

export SINGULARITYENV_HF_HOME="/d/hpc/projects/onj_fri/nlpmaxxing/hf"
export APPTAINERENV_HF_HOME="/d/hpc/projects/onj_fri/nlpmaxxing/hf"
export SINGULARITYENV_CUDA_LAUNCH_BLOCKING=1
export APPTAINERENV_CUDA_LAUNCH_BLOCKING=1

#export PATH="/d/hpc/home/ms88481/.local/bin:$PATH"
#source ~/.bash_profile


# multi gpu env vars
#export NCCL_DEBUG=INFO
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export TOKENIZERS_PARALLELISM=true
#export NCCL_P2P_DISABLE=1
#export NCCL_IGNORE_CPU_AFFINITY=1

# multi gpu
#singularity exec --nv /d/hpc/home/ms88481/containers/container-marjan-cuda-torch.sif /d/hpc/home/ms88481/.local/bin/accelerate launch --config_file /d/hpc/projects/onj_fri/nlpmaxxing/hf/accelerate/default_config.yaml --num_processes=2 train.py

# multi gpu
#singularity exec --nv /d/hpc/home/ms88481/containers/container-marjan-cuda-torch.sif torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=29500 --master_addr=127.0.0.1 train.py

singularity exec --nv /d/hpc/home/ms88481/containers/container-marjan-cuda-torch.sif python train.py



