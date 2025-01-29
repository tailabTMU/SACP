import os
import numpy as np
import nibabel as nib
from pathlib import Path


def preprocess_volume(volume, is_probability_map=False):
    """
    Preprocess volume to ensure consistent orientation and dimensions.

    Args:
        volume: Input volume array
        is_probability_map: Boolean indicating if volume is probability map

    Returns:
        Processed volume with consistent orientation
    """
    if is_probability_map:
        # For probability maps with shape (C, D3, D1, D2)
        # First transpose to get (C, D1, D2, D3)
        volume = np.transpose(volume, (0, 2, 3, 1))
        # Then rotate in the spatial dimensions
        volume = np.rot90(volume, k=1, axes=(1, 2))
        volume = np.flip(volume, axis=1)

    return volume


def find_bounding_box_3d(segmentation, labels):
    """Find the smallest 3D bounding box containing all voxels with specified labels."""
    mask = np.zeros_like(segmentation, dtype=bool)
    for label in labels:
        label_mask = (segmentation == label)
        print(f"Label {label} has {np.sum(label_mask)} voxels")
        mask = mask | label_mask

    coords = np.array(np.where(mask)).T
    if coords.size == 0:
        raise ValueError("No voxels found with specified labels")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox = tuple((mins[i], maxs[i]) for i in range(3))
    return bbox


def crop_volume(volume, bbox, is_probability_map=False):
    """
    Crop a volume using bounding box coordinates.

    Args:
        volume: Input volume to crop
        bbox: Bounding box coordinates
        is_probability_map: Boolean indicating if volume is probability map
    """
    print(f"Input volume shape: {volume.shape}")
    print(f"Input bbox: {bbox}")

    if is_probability_map:
        # For probability maps (C, H, W, D)
        cropped = volume[:,  # Keep all classes
                  bbox[0][0]:bbox[0][1] + 1,  # H
                  bbox[1][0]:bbox[1][1] + 1,  # W
                  bbox[2][0]:bbox[2][1] + 1]  # D
    else:
        # For regular volumes (H, W, D)
        cropped = volume[bbox[0][0]:bbox[0][1] + 1,  # H
                  bbox[1][0]:bbox[1][1] + 1,  # W
                  bbox[2][0]:bbox[2][1] + 1]  # D

    print(f"Output shape: {cropped.shape}")
    if cropped.size == 0:
        raise ValueError(f"Cropping resulted in empty array! Input shape {volume.shape}")
    return cropped


def process_files(input_dir, basename, output_dir, target_labels):
    """Process all files related to a basename."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    print(f"\nProcessing files for basename: {basename}")

    gt_file = input_dir / f"GT{basename}.nii.gz"
    gt_nii = nib.load(str(gt_file))
    gt_seg = gt_nii.get_fdata().astype(np.int32)

    gt_seg = preprocess_volume(gt_seg)
    bbox = find_bounding_box_3d(gt_seg, target_labels)

    file_patterns = {
        'vessels': f"Vessel{basename}.nii.gz",
        'organ': f"ORGAN_tumour{basename}.nii.gz",
        'organ_prob': f"ORGAN_tumour{basename}_probabilities.npz",
        'gt': f"GT{basename}.nii.gz",
        'original': f"{basename}_0000.nii.gz"
    }

    for file_type, pattern in file_patterns.items():
        input_path = input_dir / pattern
        output_path = output_dir / pattern
        print(f"\nProcessing {file_type}: {input_path}")

        try:
            if pattern.endswith('.nii.gz'):
                img = nib.load(str(input_path))
                data = img.get_fdata()
                if any(key in file_type for key in ['vessels', 'organ', 'gt']):
                    data = data.astype(np.int32)

                data = preprocess_volume(data)
                cropped_data = crop_volume(data, bbox)

                new_header = img.header.copy()
                new_header.set_data_shape(cropped_data.shape)
                cropped_nii = nib.Nifti1Image(cropped_data, img.affine, new_header)
                nib.save(cropped_nii, str(output_path))

            elif pattern.endswith('.npz'):
                data = np.load(str(input_path))
                cropped_data = {}
                for key, value in data.items():
                    processed_value = preprocess_volume(value, is_probability_map=True)
                    cropped_data[key] = crop_volume(processed_value, bbox, is_probability_map=True)
                np.savez(str(output_path), **cropped_data)

            print(f"Successfully saved: {output_path}")

        except Exception as e:
            print(f"Error processing {pattern}: {str(e)}")
            continue


def find_base_filenames(directory):
    """Find all unique base filenames in directory that have the required file patterns."""
    directory = Path(directory)
    base_filenames = set()
    for nii_file in directory.glob("*.nii.gz"):
        base = nii_file.name.replace(".nii.gz", "").replace("Vessel", "")
        input_dir = nii_file.parent

        required_patterns = [
            f"ORGAN_tumour{base}.nii.gz",
            f"ORGAN_tumour{base}_probabilities.npz",
            f"Vessel{base}.nii.gz",
            f"GT{base}.nii.gz"
        ]

        if all((input_dir / pattern).exists() for pattern in required_patterns):
            base_filenames.add(base)
            print(f"Found complete set for basename: {base}")

    return list(base_filenames)


def main(input_directory, output_directory=None, target_labels=[7, 8, 9, 10]):
    """Main function to process all files in directory."""
    input_directory = Path(input_directory)
    if output_directory is None:
        output_directory = input_directory / 'cropped'
    else:
        output_directory = Path(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)
    base_filenames = find_base_filenames(input_directory)

    for base_filename in base_filenames:
        try:
            process_files(input_directory, base_filename, output_directory, target_labels)
            print(f"\nSuccessfully processed {base_filename}")
        except Exception as e:
            print(f"\nError processing {base_filename}: {str(e)}")


if __name__ == "__main__":
    input_dir = r'your/path'
    output_dir = Path(input_dir) / 'cropped_results'
    main(input_dir, output_dir)
