import os
import nibabel as nib
import numpy as np
import pandas as pd

def calculate_dice(pred, gt, label):
    """Calculates Dice score for a specific label value."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_area = pred_mask.sum() + gt_mask.sum()
    
    if total_area == 0:
        # If both are empty, the model correctly predicted 'no tumor'
        return 1.0 
    
    return (2. * intersection) / total_area

# --- CONFIGURATION ---
# Paths to your folders
pred_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models/Dataset001_GMB_T1_Segmentation/test_predictions_final"
gt_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw/Dataset001_GMB_T1_Segmentation/labelsTs"

# Mapping labels to names for the report
labels_map = {
    1: "Necrotic",
    2: "Edema",
    3: "Enhancing"
}

results = []

# Table Header with specific padding for alignment
print(f"\n{'Case ID':<25} | {'Label 1':<10} | {'Label 2':<10} | {'Label 3':<10} | {'Mean':<10}")
print("-" * 80)

# Process all .nii.gz files
files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])

for filename in files:
    case_id = filename.replace(".nii.gz", "")
    
    # Load NIfTI data
    try:
        pred_img = nib.load(os.path.join(pred_dir, filename)).get_fdata()
        gt_img = nib.load(os.path.join(gt_dir, filename)).get_fdata()
        
        case_scores = {"Case_ID": case_id}
        total_score = 0
        
        # Calculate Dice for each of the 3 tumor classes
        for val, name in labels_map.items():
            score = calculate_dice(pred_img, gt_img, val)
            case_scores[name] = score
            total_score += score
            
        mean_dice = total_score / 3
        case_scores["Mean_Dice"] = mean_dice
        results.append(case_scores)
        
        # Print formatted row
        print(f"{case_id:<25} | "
              f"{case_scores['Necrotic']:0.4f}     | "
              f"{case_scores['Edema']:0.4f}     | "
              f"{case_scores['Enhancing']:0.4f}     | "
              f"{mean_dice:0.4f}")
              
    except Exception as e:
        print(f"Error processing {case_id}: {e}")

# --- SUMMARY STATISTICS ---
df = pd.DataFrame(results)

print("\n" + "="*40)
print("       OVERALL PERFORMANCE SUMMARY")
print("="*40)
summary = df.drop(columns="Case_ID").mean()
print(summary.to_string())
print("="*40)

# Save results to a CSV file for your records
output_csv = "final_test_dice_scores.csv"
df.to_csv(output_csv, index=False)
print(f"\nDetailed results saved to: {output_csv}")