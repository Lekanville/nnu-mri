import os
import nibabel as nib
import numpy as np
import pandas as pd

def calculate_dice_safe(pred, gt, label):
    """Calculates Dice using rounding to avoid float precision issues."""
    # Round to nearest integer and compare
    pred_mask = (np.rint(pred) == label)
    gt_mask = (np.rint(gt) == label)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_area = pred_mask.sum() + gt_mask.sum()
    
    if total_area == 0:
        return 1.0 
    
    return (2. * intersection) / total_area

# --- CONFIGURATION ---
pred_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models/Dataset001_GMB_T1_Segmentation/test_predictions_final"
gt_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw/Dataset001_GMB_T1_Segmentation/labelsTs"

labels_map = {1: "Necrotic", 2: "Edema", 3: "Enhancing"}
results = []

print(f"\n{'Case ID':<25} | {'Necrotic':<8} | {'Edema':<8} | {'Enhancing':<8} | {'Mean':<8}")
print("-" * 75)

files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])

for filename in files:
    case_id = filename.replace(".nii.gz", "")
    try:
        p_data = nib.load(os.path.join(pred_dir, filename)).get_fdata()
        g_data = nib.load(os.path.join(gt_dir, filename)).get_fdata()
        
        row = {"Case_ID": case_id}
        scores = []
        for val, name in labels_map.items():
            s = calculate_dice_safe(p_data, g_data, val)
            row[name] = s
            scores.append(s)
            
        mean_s = sum(scores) / 3
        row["Mean_Dice"] = mean_s
        results.append(row)
        
        print(f"{case_id:<25} | {row['Necrotic']:0.4f}   | {row['Edema']:0.4f}   | {row['Enhancing']:0.4f}    | {mean_s:0.4f}")
              
    except Exception as e:
        print(f"Error {case_id}: {e}")

# Summary
df = pd.DataFrame(results)
print("\n--- OVERALL MEAN SCORES ---")
print(df.drop(columns="Case_ID").mean().to_string())

os.makedirs("output", exist_ok=True)
df.to_csv("output/final_test_dice_scores_fixed.csv", index=False)