#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=03:59:59
#SBATCH --mem=40G
#SBATCH --exclusive
#SBATCH --output=../logs/train.out
#SBATCH --error=../logs/train.err
#SBATCH --job-name="NLPMaxxing Train"

srun train.sh

#srun singularity exec --nv ../containers/container-marjan-torch.sif nvidia-smi


# Schedule with sbatch schedule-train.sh