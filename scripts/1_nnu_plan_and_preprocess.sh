#!/bin/bash

#SBATCH --job-name=nnu_plan_and_preprocess
#SBATCH --output=nnu_plan_and_preprocess.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=3
#SBATCH --time=05:00:00         # Hours:Mins:Secs

source ~/miniforge3/bin/activate nnu_env

export nnUNet_raw="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw" 
export nnUNet_preprocessed="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_preprocessed" 
export nnUNet_results="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models"

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity