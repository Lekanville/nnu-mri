import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load the original image and the prediction
img = nib.load('path/to/test_image_0000.nii.gz').get_fdata()
seg = nib.load('path/to/test_predictions/test_image.nii.gz').get_fdata()

# Pick a slice (e.g., middle of the Z-axis)
z_slice = img.shape[2] // 2

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("T1 MRI")
plt.imshow(np.rot90(img[:, :, z_slice]), cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Model Prediction")
plt.imshow(np.rot90(img[:, :, z_slice]), cmap='gray')
plt.imshow(np.rot90(seg[:, :, z_slice]), cmap='jet', alpha=0.5) # Overlay
plt.show()