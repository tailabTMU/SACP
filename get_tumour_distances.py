import os
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def load_nifti(file_path):
    """Load a NIfTI file and return its data and pixel dimensions."""
    nii = nib.load(str(file_path))
    return nii.get_fdata(), nii.header.get_zooms()


def compute_distance_map(binary_mask, pixel_dims):
    """Compute the Euclidean distance transform for a binary mask."""
    return distance_transform_edt(~binary_mask, sampling=pixel_dims)


def process_file(organ_file, output_file, target_label=10):
    """Process an organ file to compute distances from label 10."""
    print(f"\nProcessing:\n  Organ: {organ_file.name}")

    organ_seg, organ_pixdim = load_nifti(organ_file)

    organ_mask = (organ_seg == target_label)
    if 10 not in organ_seg:
        print("not in")
        return

    if not np.any(organ_mask):
        print(f"Warning: No voxels found for label {target_label}")
        distances = np.sqrt(np.sum(np.array(organ_seg.shape) ** 2))
    else:
        distances = compute_distance_map(organ_mask, organ_pixdim)
        print(f"Pixel dimensions: {organ_pixdim}")
        print(f"Distance map shape: {distances.shape}")

    print(f"Saving results to {output_file}")
    np.save(str(output_file), distances)


def main(folder_a, folder_c):
    """
    Process all files in folder A, saving results in folder C.

    Args:
        folder_a: Path to folder containing ORGAN_tumor files
        folder_c: Path to output folder for distance maps
    """
    folder_a = Path(folder_a)
    folder_c = Path(folder_c)

    folder_c.mkdir(parents=True, exist_ok=True)

    organ_files = list(folder_a.glob("ORGAN_tumour*.nii.gz"))

    print(f"Found {len(organ_files)} files in folder A")

    for organ_file in organ_files:
        basename = organ_file.name.replace("ORGAN_tumour", "").replace(".nii.gz", "")
        output_file = folder_c / f"DISTANCETUMOUR{basename}.npy"

        try:
            process_file(organ_file, output_file)
        except Exception as e:
            print(f"Error processing {basename}: {str(e)}")


if __name__ == "__main__":
    folder_a = Path(
        r"your/path")
    folder_c = Path(
        r"your/path")

    main(folder_a, folder_c)