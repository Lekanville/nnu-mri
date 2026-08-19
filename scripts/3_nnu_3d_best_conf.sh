#!/bin/bash

#SBATCH --job-name=nnu_3d_best_conf
#SBATCH --output=3_nnu_3d_best_conf.out
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate nnu_env

export nnUNet_raw="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw" 
export nnUNet_preprocessed="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_preprocessed" 
export nnUNet_results="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models"

# Fix for the Triton compilation error
export nnUNet_compile=False

# --- Command Execution Change --- 
# srun orchestrates the launch across the 4 tasks/GPUs
# srun nnUNetv2_determine_postprocessing 1 3d_fullres
srun nnUNetv2_find_best_configuration 1 -c 3d_fullres
# --------------------------------

