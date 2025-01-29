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


def process_pair(organ_file, vessel_file, output_file, vessel_labels=[2, 3, 5, 7, 9]):
    """Process a pair of organ and vessel files to compute distances."""
    print(f"\nProcessing:\n  Organ: {organ_file.name}\n  Vessel: {vessel_file.name}")

    organ_seg, organ_pixdim = load_nifti(organ_file)
    vessel_seg, vessel_pixdim = load_nifti(vessel_file)

    if not np.allclose(organ_pixdim, vessel_pixdim):
        raise ValueError("Pixel dimensions of organ and vessel files do not match")

    distances = np.zeros((len(vessel_labels), *organ_seg.shape))

    for i, label in enumerate(vessel_labels):
        print(f"Computing distances for vessel label {label}")
        vessel_mask = (vessel_seg == label)

        if not np.any(vessel_mask):
            print(f"Warning: No voxels found for vessel label {label}")
            distances[i] = np.sqrt(np.sum(np.array(organ_seg.shape) ** 2))
            continue

        dist_map = compute_distance_map(vessel_mask, vessel_pixdim)
        print(f"Pixel dimensions: {vessel_pixdim}")
        print(f"Distance map shape: {dist_map.shape}")

        distances[i] = dist_map

    print(f"Final distances shape: {distances.shape}")

    print(f"Saving results to {output_file}")
    np.save(str(output_file), distances)


def main(folder_a, folder_b, folder_c):
    """
    Process all matching files in folders A and B, saving results in folder C.

    Args:
        folder_a: Path to folder containing ORGAN_tumor files
        folder_b: Path to folder containing VESSEL files
        folder_c: Path to output folder for distance maps
    """
    folder_a = Path(folder_a)
    folder_b = Path(folder_b)
    folder_c = Path(folder_c)

    folder_c.mkdir(parents=True, exist_ok=True)

    organ_files = list(folder_a.glob("ORGAN_tumour*.nii.gz"))

    print(f"Found {len(organ_files)} files in folder A")

    for organ_file in organ_files:
        basename = organ_file.name.replace("ORGAN_tumour", "").replace(".nii.gz", "")
        vessel_file = folder_b / f"Vessel{basename}.nii.gz"

        if not vessel_file.exists():
            print(f"Warning: No matching vessel file found for {organ_file.name}")
            continue

        output_file = folder_c / f"DISTANCE{basename}.npy"

        try:
            process_pair(organ_file, vessel_file, output_file)
        except Exception as e:
            print(f"Error processing {basename}: {str(e)}")


if __name__ == "__main__":
    folder_a = Path(
        r"your/path")
    folder_b = Path(
        r"your/path")
    folder_c = Path(
        r"your/path")

    main(folder_a, folder_b, folder_c)