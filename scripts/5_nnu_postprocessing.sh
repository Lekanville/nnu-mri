#!/bin/bash

#SBATCH --job-name=5_nnu_postprocess
#SBATCH --output=5_nnu_postprocess.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Load CUDA just to keep the environment consistent, though not strictly required
module load cuda/12.6

source ~/miniforge3/bin/activate nnu_env

export OMP_NUM_THREADS=1
export nnUNet_raw="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw" 
export nnUNet_preprocessed="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_preprocessed" 
export nnUNet_results="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models"

# --- Post-processing Command ---
nnUNetv2_apply_postprocessing \
    -i ${nnUNet_results}/Dataset001_GMB_T1_Segmentation/test_predictions \
    -o ${nnUNet_results}/Dataset001_GMB_T1_Segmentation/test_predictions_final \
    -pp_pkl_file ${nnUNet_results}/Dataset001_GMB_T1_Segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
    -np 8 \
    -plans_json ${nnUNet_results}/Dataset001_GMB_T1_Segmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json