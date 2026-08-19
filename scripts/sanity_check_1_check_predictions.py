import nibabel as nib
import numpy as np
import os

pred_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models/Dataset001_GMB_T1_Segmentation/test_predictions_final"

print(f"{'Case ID':<25} | {'Unique Labels in Prediction'}")
print("-" * 50)

for filename in sorted(os.listdir(pred_dir))[:10]: # Check first 10
    if filename.endswith(".nii.gz"):
        img = nib.load(os.path.join(pred_dir, filename)).get_fdata()
        unique = np.unique(img)
        print(f"{filename:<25} | {unique}")