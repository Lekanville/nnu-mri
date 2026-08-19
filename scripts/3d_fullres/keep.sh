#!/bin/bash

#SBATCH --job-name=nnu_3d_train_fold_0
#SBATCH --output=nnu_3d_train_fold_0.out
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
# USE SRUN to launch the process across the allocated resources (tasks/GPUs)
srun nnUNetv2_train 1 3d_fullres 0 -device cuda
# srun nnUNetv2_train 1 3d_fullres nnUNetTrainer_resenc_3d_fullres 0 -device cuda