# Step 1: Import necessary libraries
import nibabel as nib
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import resample_img
from skimage import exposure
from scipy.ndimage import gaussian_filter

# Step 2: Load the NIfTI file using nibabel
file_path = 'abide/ABIDE_pcp/cpac/nofilt_noglobal/OHSU_0050142_func_preproc.nii.gz'

# Load the NIfTI image
img = nib.load(file_path)

# Get the image data as a NumPy array (3D or 4D)
data = img.get_fdata()

# Step 3: Print the shape and voxel dimensions of the image
print("Image shape:", data.shape)
print("Voxel dimensions (affine matrix):", img.header.get_zooms())

# Step 4: Handle 4D data
if len(data.shape) == 4:
    # Compute the mean across the time dimension for 4D data
    mean_volume = np.mean(data, axis=3)  # Compute mean across time
else:
    # Use the data directly for 3D images
    mean_volume = data

# Step 5: Convert to a standard array to avoid MaskedArray issues
mean_volume = np.asarray(mean_volume)

# Step 6: Normalize the mean image data for better visualization
mean_volume = (mean_volume - np.min(mean_volume)) / (np.max(mean_volume) - np.min(mean_volume))

# Step 7: Apply edge-preserving smoothing (Gaussian filter)
smoothed_volume = gaussian_filter(mean_volume, sigma=0.5)  # Adjust sigma for smoothing level

# Step 8: Resample the smoothed image to a higher resolution (1x1x1 mm)
target_affine = np.diag([1, 1, 1])  # Resample to 1x1x1 mm isotropic voxels
smoothed_img = nib.Nifti1Image(smoothed_volume, img.affine)
resampled_img = resample_img(smoothed_img, target_affine=target_affine, force_resample=True, copy_header=True)

# Step 9: Enhance contrast using histogram equalization
resampled_data = resampled_img.get_fdata()
contrast_enhanced_volume = exposure.equalize_hist(resampled_data)

# Step 10: Visualize slices using matplotlib
slice_index = contrast_enhanced_volume.shape[2] // 2  # Middle slice (Z-dimension)

plt.figure(dpi=200)  # Set high DPI for clear visualization
plt.imshow(contrast_enhanced_volume[:, :, slice_index], cmap='gray', interpolation='bicubic')
plt.title(f"Enhanced middle slice (Z-dimension index {slice_index})")
plt.axis('off')
plt.show()

# Step 11: Use Nilearn's interactive viewer for the contrast-enhanced image
contrast_enhanced_img = nib.Nifti1Image(contrast_enhanced_volume, resampled_img.affine)
plotting.view_img(contrast_enhanced_img)
