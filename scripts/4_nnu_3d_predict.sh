#!/bin/bash

#SBATCH --job-name=4_nnu_3d_predict
#SBATCH --output=4_nnu_3d_predict.out
#SBATCH --gres=gpu:1              
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# 1. Load the CORRECT version found by spider
module load cuda/12.6

source ~/miniforge3/bin/activate nnu_env

# 2. Environment Tuning
export nnUNet_compile=False
export OMP_NUM_THREADS=1
export nnUNet_raw="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw" 
export nnUNet_preprocessed="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_preprocessed" 
export nnUNet_results="/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models"

# 3. Debug Check
echo "--- GPU Debug Info ---"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version PyTorch wants: {torch.version.cuda}')"

# 4. Prediction Command
nnUNetv2_predict \
    -d Dataset001_GMB_T1_Segmentation \
    -i ${nnUNet_raw}/Dataset001_GMB_T1_Segmentation/imagesTs \
    -o ${nnUNet_results}/Dataset001_GMB_T1_Segmentation/test_predictions \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -c 3d_fullres \
    -p nnUNetPlans \
    -npp 1 \
    -nps 1 \
    -device cuda
# --------------------------------