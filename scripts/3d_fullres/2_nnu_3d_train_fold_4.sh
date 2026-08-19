#!/bin/bash

#SBATCH --job-name=nnu_3d_train_fold_4
#SBATCH --output=nnu_3d_train_fold_4.out
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate nnu_env

export nnUNet_raw="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw" 
export nnUNet_preprocessed="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_preprocessed" 
export nnUNet_results="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models"

# Fix for the Triton compilation error
export nnUNet_compile=False

# --- Command Execution Change --- 
# srun orchestrates the launch across the 4 tasks/GPUs
srun nnUNetv2_train 1 3d_fullres 4 -device cuda --c
# srun nnUNetv2_train 1 3d_fullres nnUNetTrainer_resenc_3d_fullres 4 -device cuda
# --------------------------------