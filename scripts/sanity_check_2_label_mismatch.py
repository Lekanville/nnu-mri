import nibabel as nib
import numpy as np
import os

# Using the case we know has a high total overlap
case = "UPENN-GBM-00018_11.nii.gz"
gt_path = f"/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw/Dataset001_GMB_T1_Segmentation/labelsTs/{case}"
pred_path = f"/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models/Dataset001_GMB_T1_Segmentation/test_predictions_final/{case}"

gt_data = nib.load(gt_path).get_fdata()
pred_data = nib.load(pred_path).get_fdata()

print(f"Checking labels for: {case}")
print(f"Unique labels in Prediction:   {np.unique(pred_data)}")
print(f"Unique labels in Ground Truth: {np.unique(gt_data)}")

# Check for the specific "Label 3 vs 4" mismatch
gt_has_4 = np.any(gt_data == 4)
gt_has_3 = np.any(gt_data == 3)

if gt_has_4 and not gt_has_3:
    print("\nFOUND IT: Your Ground Truth still uses Label 4, but your model predicts Label 3.")
    print("This is why your Dice score is 0.0000.")