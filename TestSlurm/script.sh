#!/bin/bash

mkdir -p /scratch/s02210430/test/data
srun --gres=gpu:1 --mem=20G --container-image /scratch/s02210430/test/noisy-cells.sqsh --container-mounts /scratch/s02210430/test/data:/workspace/data bash -c 'python3 code/nn.py' > output.log 2>&1
