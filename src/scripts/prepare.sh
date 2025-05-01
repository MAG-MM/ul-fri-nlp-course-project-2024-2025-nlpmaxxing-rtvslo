#!/bin/bash

# This script is to create and provision a container

singularity build /d/hpc/home/ms88481/containers/container-marjan-cuda-torch.sif docker://pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

singularity exec /d/hpc/home/ms88481/containers/container-marjan-cuda-torch.sif pip install transformers datasets accelerate peft trl bitsandbytes protobuf blobfile sentencepiece polars

# singularity exec /d/hpc/home/ms88481/containers/container-marjan-cuda-torch.sif /d/hpc/home/ms88481/.local/bin/accelerate config
# accelerate configuration saved at /d/hpc/projects/onj_fri/nlpmaxxing/hf/accelerate/default_config.yaml

echo "Completed, you can now train"