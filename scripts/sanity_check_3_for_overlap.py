import nibabel as nib
import numpy as np
import os

pred_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models/Dataset001_GMB_T1_Segmentation/test_predictions_final"
gt_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw/Dataset001_GMB_T1_Segmentation/labelsTs"

case = "UPENN-GBM-00018_11.nii.gz"

pred_img = nib.load(os.path.join(pred_dir, case)).get_fdata()
gt_img = nib.load(os.path.join(gt_dir, case)).get_fdata()

# Check if there is ANY overlap between foreground (labels 1,2,3)
overlap = np.logical_and(pred_img > 0, gt_img > 0).sum()
pred_vol = (pred_img > 0).sum()
gt_vol = (gt_img > 0).sum()

print(f"Analysis for {case}:")
print(f"Prediction Volume: {pred_vol} voxels")
print(f"Ground Truth Volume: {gt_vol} voxels")
print(f"Intersection: {overlap} voxels")

if overlap == 0 and pred_vol > 0 and gt_vol > 0:
    print("\nALERT: Spatial Mismatch! Both files have tumors, but they don't touch.")