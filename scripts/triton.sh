#!/bin/bash

#SBATCH --job-name=triton
#SBATCH --output=triton.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=3
#SBATCH --time=05:00:00         # Hours:Mins:Secs

source ~/miniforge3/bin/activate torch_env

pip install triton