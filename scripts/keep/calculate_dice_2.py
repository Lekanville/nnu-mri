import os
import nibabel as nib
import numpy as np
import pandas as pd

def calculate_dice_safe(pred, gt, label):
    """Calculates Dice using rounding to avoid float precision issues."""
    # Rounding to nearest integer handles 2.99999 or 3.00001 issues
    pred_mask = (np.rint(pred) == label)
    gt_mask = (np.rint(gt) == label)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_area = pred_mask.sum() + gt_mask.sum()
    
    if total_area == 0:
        return 1.0 
    
    return (2. * intersection) / total_area

# --- CONFIG ---
pred_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models/Dataset001_GMB_T1_Segmentation/test_predictions_final"
gt_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw/Dataset001_GMB_T1_Segmentation/labelsTs"
case = "UPENN-GBM-00018_11.nii.gz"

# Test one case
p_nii = nib.load(os.path.join(pred_dir, case))
g_nii = nib.load(os.path.join(gt_dir, case))

p_data = p_nii.get_fdata()
g_data = g_nii.get_fdata()

print(f"--- Precision Test for {case} ---")
for i in [1, 2, 3]:
    score = calculate_dice_safe(p_data, g_data, i)
    print(f"Label {i} Dice (with rounding): {score:.4f}")

# Check data types just in case
print(f"\nPred Data Type: {p_data.dtype}")
print(f"GT Data Type: {g_data.dtype}")