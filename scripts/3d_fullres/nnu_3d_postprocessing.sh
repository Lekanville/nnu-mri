#!/bin/bash

#SBATCH --job-name=nnu_3d_postprocessing
#SBATCH --output=nnu_3d_postprocessing.out
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate tnnu_env

export nnUNet_raw="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw" 
export nnUNet_preprocessed="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_preprocessed" 
export nnUNet_results="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models"

# Fix for the Triton compilation error
export nnUNet_compile=False

# --- Command Execution Change --- 
# srun orchestrates the launch across the 4 tasks/GPUs
# srun nnUNetv2_determine_postprocessing 1 3d_fullres
srun nnUNetv2_determine_postprocessing \
    -i "${nnUNet_results}/Dataset001_GMB_T1_Segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres" \
    -ref "${nnUNet_preprocessed}/Dataset001_GMB_T1_Segmentation/gt_segmentations" \
    -plans_json "${nnUNet_preprocessed}/Dataset001_GMB_T1_Segmentation/nnUNetPlans.json" \
    -dataset_json "${nnUNet_preprocessed}/Dataset001_GMB_T1_Segmentation/dataset.json" \
    -np 8 
# --------------------------------