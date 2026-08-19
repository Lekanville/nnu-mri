import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os

# --- CONFIGURATION ---
img_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw/Dataset001_GMB_T1_Segmentation/imagesTs"
gt_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_raw/Dataset001_GMB_T1_Segmentation/labelsTs"
pred_dir = "/home/b5ac/olalekan.b5ac/mri_nnu/nnUNet_trained_models/Dataset001_GMB_T1_Segmentation/test_predictions_final"

# Lists from your results
cases_to_plot = {
    "TOP_1": "UPENN-GBM-00189_11",
    "TOP_2": "UPENN-GBM-00146_11",
    "TOP_3": "UPENN-GBM-00147_11",
    "BOTTOM_1": "UPENN-GBM-00388_11",
    "BOTTOM_2": "UPENN-GBM-00173_11",
    "BOTTOM_3": "UPENN-GBM-00080_11"
}

def create_report_figure(category, case_id):
    img_path = os.path.join(img_dir, f"{case_id}_0000.nii.gz")
    gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
    pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
    
    # Load and force to integer for clean plotting
    img_data = nib.load(img_path).get_fdata()
    gt_data = np.rint(nib.load(gt_path).get_fdata())
    pred_data = np.rint(nib.load(pred_path).get_fdata())
    
    # Identify the slice with the maximum tumor volume in Ground Truth
    slice_idx = np.argmax(np.sum(gt_data > 0, axis=(0, 1)))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f"{category}: {case_id} (Central Slice: {slice_idx})", fontsize=18, fontweight='bold', y=0.95)

    # Column 1: T1 MRI
    axes[0].imshow(img_data[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0].set_title("Input T1 MRI", fontsize=14)
    axes[0].axis('off')

    # Column 2: Ground Truth Overlay
    axes[1].imshow(img_data[:, :, slice_idx].T, cmap='gray', origin='lower')
    # Using 'gist_rainbow' or 'jet' to see the 1, 2, 3 labels clearly
    gt_overlay = np.ma.masked_where(gt_data <= 0, gt_data)
    axes[1].imshow(gt_overlay[:, :, slice_idx].T, cmap='gist_rainbow', alpha=0.6, origin='lower')
    axes[1].set_title("Ground Truth Labels", fontsize=14)
    axes[1].axis('off')

    # Column 3: Model Prediction Overlay
    axes[2].imshow(img_data[:, :, slice_idx].T, cmap='gray', origin='lower')
    pred_overlay = np.ma.masked_where(pred_data <= 0, pred_data)
    axes[2].imshow(pred_overlay[:, :, slice_idx].T, cmap='gist_rainbow', alpha=0.6, origin='lower')
    axes[2].set_title("nnU-Net Prediction", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    filename = f"output/report_{category}_{case_id}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()

# Execute
for category, cid in cases_to_plot.items():
    try:
        create_report_figure(category, cid)
    except Exception as e:
        print(f"Skipping {cid}: {e}")