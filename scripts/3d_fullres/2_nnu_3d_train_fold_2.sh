#!/bin/bash

#SBATCH --job-name=nnu_3d_train_fold_2_CONT
#SBATCH --output=nnu_3d_train_fold_2_CONT.out
# ---------------------------------------------
# ADD REQUEUE OPTION
#SBATCH --requeue
# ---------------------------------------------
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


# === DDP FIX: Explicitly set Master Node Info ===
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
# ===============================================

# Use RESENC trainer AND the -r flag to signal restart/resume
srun nnUNetv2_train 1 3d_fullres 2 -device cuda --c 