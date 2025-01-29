import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib


def calculate_average_vessel_distance_width(predset_dir: str, gt_dir: str) -> dict:
    """
    Calculate average prediction set size (width) across all cases and vessels.

    Args:
        predset_dir (str): Directory containing prediction set .npy files
        gt_dir (str): Directory containing ground truth and distance files

    Returns:
        dict: Dictionary containing averaged width statistics per distance
    """
    distances = np.arange(0, 21, 1)  # 0-20mm in 1mm steps

    total_stats = {f'{d}mm': {'total_size': 0, 'total_count': 0} for d in distances}
    predset_files = sorted(Path(predset_dir).glob("*_predset.npy"))

    for predset_file in predset_files:
        case_id = predset_file.name.replace("_predset.npy", "")
        gt_file = Path(gt_dir) / f"GT{case_id}.nii.gz"
        distance_file = Path(gt_dir) / f"DISTANCE{case_id}.npy"

        if not (gt_file.exists() and distance_file.exists()):
            print(f"Skipping case {case_id} - files not found")
            continue

        predset = np.load(predset_file)
        if predset.shape[0] == 11:
            predset = np.transpose(predset, (1, 2, 3, 0))

        vessel_distances = np.load(distance_file)

        has_label_10 = np.any(predset == 10, axis=-1)

        total_labels_per_voxel = np.sum(predset > 0, axis=-1)
        predset_size = np.where(has_label_10, total_labels_per_voxel, 0)

        for vessel_idx in range(len(vessel_distances)):
            curr_distances = vessel_distances[vessel_idx]

            for d in distances:
                if d < 20:
                    distance_mask = (curr_distances >= d) & (curr_distances < (d + 1))
                else:
                    distance_mask = (curr_distances >= d)

                valid_voxels = has_label_10 & distance_mask
                total_voxels = np.sum(valid_voxels)
                total_predset_size = np.sum(predset_size[valid_voxels])

                if total_voxels > 0:
                    total_stats[f'{d}mm']['total_size'] += total_predset_size
                    total_stats[f'{d}mm']['total_count'] += total_voxels

    average_widths = {}
    for dist in total_stats:
        if total_stats[dist]['total_count'] > 0:
            average_widths[dist] = total_stats[dist]['total_size'] / total_stats[dist]['total_count']
        else:
            average_widths[dist] = 0

    return average_widths


def calculate_coverage_by_distance(predset_dir: str, gt_dir: str) -> dict:
    """
    Calculate coverage statistics per distance interval.
    """
    distances = np.arange(0, 21, 1)  # 0-20mm in 1mm steps
    total_stats = {f'{d}mm': {'covered_pixels': 0, 'total_pixels': 0} for d in distances}

    predset_files = sorted(Path(predset_dir).glob("*_predset.npy"))

    for predset_file in predset_files:
        case_id = predset_file.name.replace("_predset.npy", "")
        gt_file = Path(gt_dir) / f"GT{case_id}.nii.gz"
        distance_file = Path(gt_dir) / f"DISTANCE{case_id}.npy"

        if not (gt_file.exists() and distance_file.exists()):
            continue

        predset = np.load(predset_file)
        if predset.shape[0] == 11:
            predset = np.transpose(predset, (1, 2, 3, 0))

        gt = nib.load(gt_file).get_fdata().astype(int)
        vessel_distances = np.load(distance_file)

        label10_mask = (gt == 10)
        gt_covered = np.any(predset == 10, axis=-1)

        for vessel_idx in range(len(vessel_distances)):
            curr_distances = vessel_distances[vessel_idx]

            for d in distances:
                if d < 20:
                    distance_mask = (curr_distances >= d) & (curr_distances < (d + 1))
                else:
                    distance_mask = (curr_distances >= d)

                relevant_mask = label10_mask & distance_mask
                total_pixels = np.sum(relevant_mask)
                covered_pixels = np.sum(gt_covered & relevant_mask)

                total_stats[f'{d}mm']['total_pixels'] += total_pixels
                total_stats[f'{d}mm']['covered_pixels'] += covered_pixels

    coverage_by_distance = {}
    for dist in total_stats:
        if total_stats[dist]['total_pixels'] > 0:
            coverage_by_distance[dist] = (total_stats[dist]['covered_pixels'] /
                                          total_stats[dist]['total_pixels']) * 100
        else:
            coverage_by_distance[dist] = 0

    return coverage_by_distance


def plot_combined_heatmap(width_stats: dict, coverage_stats: dict, save_path: str = None):
    """
    Create a heatmap visualization combining RWR and coverage statistics with separate color scales,
    showing annotations only for every second interval but keeping all colors.
    """
    distances = list(width_stats.keys())
    width_values = list(width_stats.values())
    coverage_values = [coverage_stats[d] for d in distances]

    width_annot_fmt = np.array([[f'{x:.2f}' if i % 2 == 0 else '' for i, x in enumerate(width_values)]])
    coverage_annot_fmt = np.array([[f'{x:.2f}' if i % 2 == 0 else '' for i, x in enumerate(coverage_values)]])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    sns.heatmap([width_values],
                xticklabels=distances,
                yticklabels=['Average RWR'],
                cmap='YlOrRd',
                annot=width_annot_fmt,
                fmt='',
                cbar_kws={'label': 'RWR Value'},
                ax=ax1)

    sns.heatmap([coverage_values],
                xticklabels=distances,
                yticklabels=['Coverage (%)'],
                cmap='Blues',
                annot=coverage_annot_fmt,
                fmt='',
                cbar_kws={'label': 'Coverage %'},
                ax=ax2)

    ax2.set_xlabel('Distance from Vessel (mm)')

    plt.setp(ax2.get_xticklabels(), rotation=45)
    plt.setp(ax1.get_xticklabels(), visible=False)

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
        plt.close()
    else:
        plt.show()


predset_dir = r"your/path"
gt_dir = r"your/path"

average_widths = calculate_average_vessel_distance_width(predset_dir, gt_dir)
coverage_by_distance = calculate_coverage_by_distance(predset_dir, gt_dir)

plot_combined_heatmap(average_widths, coverage_by_distance, "figure2-PAN.pdf")
