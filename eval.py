import numpy as np
import nibabel as nib
import os
from pathlib import Path
from datetime import datetime
from scipy.stats import wilcoxon
import numpy as np


def calculate_vessel_distance_width(predset_dir: str, gt_dir: str) -> dict:
    """
    Calculate average prediction set size (width) for specific vessel distance thresholds,
    considering only voxels that contain label 10 in their prediction set.

    Args:
        predset_dir (str): Directory containing prediction set .npy files
        gt_dir (str): Directory containing ground truth and distance files

    Returns:
        dict: Dictionary containing width statistics per vessel and threshold
    """
    vessel_names = ['Vessel1', 'Vessel2', 'Vessel3', 'Vessel4', 'Vessel5']
    thresholds = [2, 5, 10, 20]

    width_stats = {
        'per_vessel': {vessel: {
            **{f'<{thresh}mm': {'total_voxels': 0, 'total_predset_size': 0}
               for thresh in thresholds},
            '>20mm': {'total_voxels': 0, 'total_predset_size': 0}
        } for vessel in vessel_names},
        'average': {}
    }

    predset_files = sorted(Path(predset_dir).glob("*_predset.npy"))

    for predset_file in predset_files:
        basename = predset_file.name.replace("_predset.npy", "")
        gt_file = Path(gt_dir) / f"GT{basename}.nii.gz"
        distance_file = Path(gt_dir) / f"DISTANCE{basename}.npy"

        if not (gt_file.exists() and distance_file.exists()):
            continue

        predset = np.load(predset_file)
        if predset.shape[0] == 11:
            predset = np.transpose(predset, (1, 2, 3, 0))

        gt = nib.load(gt_file).get_fdata().astype(int)
        distances = np.load(distance_file)

        has_label_10 = np.any(predset == 10, axis=-1)
        total_labels_per_voxel = np.sum(predset > 0, axis=-1)

        predset_size = np.where(has_label_10, total_labels_per_voxel, 0)

        for vessel_idx, vessel_name in enumerate(vessel_names):
            vessel_distances = distances[vessel_idx]

            for thresh in thresholds:
                distance_mask = (vessel_distances <= thresh)

                valid_voxels = has_label_10 & distance_mask
                total_voxels = np.sum(valid_voxels)
                total_predset_size = np.sum(predset_size[valid_voxels])

                width_stats['per_vessel'][vessel_name][f'<{thresh}mm']['total_voxels'] += total_voxels
                width_stats['per_vessel'][vessel_name][f'<{thresh}mm']['total_predset_size'] += total_predset_size

            distance_mask_over20 = (vessel_distances > 20)
            valid_voxels = has_label_10 & distance_mask_over20
            total_voxels = np.sum(valid_voxels)
            total_predset_size = np.sum(predset_size[valid_voxels])

            width_stats['per_vessel'][vessel_name]['>20mm']['total_voxels'] += total_voxels
            width_stats['per_vessel'][vessel_name]['>20mm']['total_predset_size'] += total_predset_size

    for vessel in vessel_names:
        for thresh in thresholds:
            thresh_key = f'<{thresh}mm'
            total_voxels = width_stats['per_vessel'][vessel][thresh_key]['total_voxels']
            total_predset_size = width_stats['per_vessel'][vessel][thresh_key]['total_predset_size']
            width_stats['per_vessel'][vessel][thresh_key] = \
                total_predset_size / total_voxels if total_voxels > 0 else 0

        total_voxels = width_stats['per_vessel'][vessel]['>20mm']['total_voxels']
        total_predset_size = width_stats['per_vessel'][vessel]['>20mm']['total_predset_size']
        width_stats['per_vessel'][vessel]['>20mm'] = \
            total_predset_size / total_voxels if total_voxels > 0 else 0

    for thresh in thresholds:
        thresh_key = f'<{thresh}mm'
        vessel_widths = [width_stats['per_vessel'][vessel][thresh_key] for vessel in vessel_names]
        width_stats['average'][thresh_key] = np.mean(vessel_widths)

    vessel_widths_over20 = [width_stats['per_vessel'][vessel]['>20mm'] for vessel in vessel_names]
    width_stats['average']['>20mm'] = np.mean(vessel_widths_over20)

    return width_stats


def calculate_vessel_distance_coverage(predset_dir: str, gt_dir: str, thresholds_mm=[2, 5, 10, 20]) -> dict:
    """
    Calculate coverage statistics for label 10 based on distance from vessels.

    Args:
        predset_dir (str): Directory containing prediction set .npy files
        gt_dir (str): Directory containing ground truth and distance files
        thresholds_mm (list): Distance thresholds in mm to analyze

    Returns:
        dict: Dictionary containing coverage statistics per vessel and threshold
    """
    vessel_names = ['Vessel1', 'Vessel2', 'Vessel3', 'Vessel4', 'Vessel5']

    coverage_stats = {
        'per_vessel': {vessel: {
            **{f'{thresh}mm': {'total_pixels': 0, 'covered_pixels': 0} for thresh in thresholds_mm},
            '>20mm': {'total_pixels': 0, 'covered_pixels': 0}
        } for vessel in vessel_names},
        'average': {
            **{f'{thresh}mm': 0 for thresh in thresholds_mm},
            '>20mm': 0
        }
    }

    predset_files = sorted(Path(predset_dir).glob("*_predset.npy"))

    for predset_file in predset_files:
        basename = predset_file.name.replace("_predset.npy", "")
        gt_file = Path(gt_dir) / f"GT{basename}.nii.gz"
        distance_file = Path(gt_dir) / f"DISTANCE{basename}.npy"

        if not (gt_file.exists() and distance_file.exists()):
            continue

        predset = np.load(predset_file)
        if predset.shape[0] == 11:
            predset = np.transpose(predset, (1, 2, 3, 0))

        gt = nib.load(gt_file).get_fdata().astype(int)
        distances = np.load(distance_file)

        label10_mask = (gt == 10)

        gt_covered = np.zeros_like(gt, dtype=bool)
        for i in range(predset.shape[-1]):
            curr_pred = predset[..., i]
            if curr_pred.shape != gt.shape:
                continue
            gt_covered |= (curr_pred == 10)

        label10_covered = gt_covered & label10_mask

        for vessel_idx, vessel_name in enumerate(vessel_names):
            vessel_distances = distances[vessel_idx]

            for thresh in thresholds_mm:
                distance_mask = (vessel_distances <= thresh)
                thresh_label10_mask = label10_mask & distance_mask
                total_pixels = np.sum(thresh_label10_mask)
                covered_pixels = np.sum(label10_covered & thresh_label10_mask)

                coverage_stats['per_vessel'][vessel_name][f'{thresh}mm']['total_pixels'] += total_pixels
                coverage_stats['per_vessel'][vessel_name][f'{thresh}mm']['covered_pixels'] += covered_pixels

            distance_mask_over20 = (vessel_distances > 20)
            thresh_label10_mask = label10_mask & distance_mask_over20
            total_pixels = np.sum(thresh_label10_mask)
            covered_pixels = np.sum(label10_covered & thresh_label10_mask)

            coverage_stats['per_vessel'][vessel_name]['>20mm']['total_pixels'] += total_pixels
            coverage_stats['per_vessel'][vessel_name]['>20mm']['covered_pixels'] += covered_pixels

    for vessel in vessel_names:
        for thresh in thresholds_mm:
            thresh_key = f'{thresh}mm'
            total = coverage_stats['per_vessel'][vessel][thresh_key]['total_pixels']
            covered = coverage_stats['per_vessel'][vessel][thresh_key]['covered_pixels']
            coverage_stats['per_vessel'][vessel][thresh_key]['coverage'] = \
                covered / total if total > 0 else 0

        total = coverage_stats['per_vessel'][vessel]['>20mm']['total_pixels']
        covered = coverage_stats['per_vessel'][vessel]['>20mm']['covered_pixels']
        coverage_stats['per_vessel'][vessel]['>20mm']['coverage'] = \
            covered / total if total > 0 else 0

    for thresh in thresholds_mm:
        thresh_key = f'{thresh}mm'
        vessel_coverages = [
            coverage_stats['per_vessel'][vessel][thresh_key]['coverage']
            for vessel in vessel_names
        ]
        coverage_stats['average'][thresh_key] = np.mean(vessel_coverages)

    vessel_coverages_over20 = [
        coverage_stats['per_vessel'][vessel]['>20mm']['coverage']
        for vessel in vessel_names
    ]
    coverage_stats['average']['>20mm'] = np.mean(vessel_coverages_over20)

    return coverage_stats


def calculate_coverage(predset_dir: str, gt_dir: str) -> dict:
    """
    Calculate coverage statistics for label 10 prediction sets against ground truth.

    Args:
        predset_dir (str): Directory containing prediction set .npy files
        gt_dir (str): Directory containing ground truth .nii.gz files

    Returns:
        dict: Dictionary containing coverage statistics
    """
    total_label10_pixels = 0
    covered_label10_pixels = 0
    per_case_coverage = {}

    predset_files = sorted(Path(predset_dir).glob("*_predset.npy"))

    for predset_file in predset_files:
        basename = predset_file.name.replace("_predset.npy", "")
        gt_file = Path(gt_dir) / f"GT{basename}.nii.gz"

        if not gt_file.exists():
            continue

        predset = np.load(predset_file)

        if predset.shape[0] == 11:
            predset = np.transpose(predset, (1, 2, 3, 0))

        gt_nib = nib.load(gt_file)
        gt = gt_nib.get_fdata()
        gt = gt.astype(int)

        label10_mask = (gt == 10)

        if not np.any(label10_mask):
            continue

        gt_covered = np.zeros_like(gt, dtype=bool)
        for i in range(predset.shape[-1]):
            curr_pred = predset[..., i]
            if curr_pred.shape != gt.shape:
                continue
            gt_covered |= (curr_pred == 10)

        label10_covered = gt_covered & label10_mask

        case_label10_pixels = label10_mask.sum()
        case_covered_pixels = label10_covered.sum()
        case_coverage = case_covered_pixels / case_label10_pixels if case_label10_pixels > 0 else 0
        per_case_coverage[basename] = case_coverage

        total_label10_pixels += case_label10_pixels
        covered_label10_pixels += case_covered_pixels

    overall_coverage = covered_label10_pixels / total_label10_pixels if total_label10_pixels > 0 else 0

    return {
        'overall_coverage': overall_coverage,
        'per_case_coverage': per_case_coverage,
        'average_case_coverage': np.mean(list(per_case_coverage.values())) if per_case_coverage else 0,
        'total_label10_pixels': total_label10_pixels,
        'covered_label10_pixels': covered_label10_pixels
    }


def analyze_distance_significance(coverage_stats: dict) -> dict:
    """
    Performs statistical significance testing between different distance groups
    using Wilcoxon signed-rank test and Benjamini-Hochberg correction.
    Handles cases where groups have identical values.

    Args:
        coverage_stats: Dictionary containing coverage statistics per vessel and threshold

    Returns:
        dict: Dictionary containing p-values and adjusted p-values for each comparison
    """

    def benjamini_hochberg(p_values):
        """
        Applies Benjamini-Hochberg correction to p-values.
        """
        n = len(p_values)
        ranked_p_values = sorted(range(n), key=lambda i: p_values[i])
        adjusted = [0] * n

        for rank, i in enumerate(ranked_p_values, 1):
            adjusted[i] = min(1, p_values[i] * n / rank)

        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        return adjusted

    distance_groups = ['2mm', '5mm', '10mm', '20mm', '>20mm']
    vessel_names = ['Vessel1', 'Vessel2', 'Vessel3', 'Vessel4', 'Vessel5']

    group_data = {}
    for group in distance_groups:
        if group == '>20mm':
            group_data[group] = [coverage_stats['per_vessel'][vessel][group]['coverage']
                                 for vessel in vessel_names]
        else:
            group_data[group] = [coverage_stats['per_vessel'][vessel][f'{group}']['coverage']
                                 for vessel in vessel_names]

    significance_results = {
        'p_values': {},
        'mean_coverages': {},
        'std_coverages': {}
    }

    for group in distance_groups:
        significance_results['mean_coverages'][group] = np.mean(group_data[group])
        significance_results['std_coverages'][group] = np.std(group_data[group])

    comparisons = []
    p_values = []
    mean_differences = []

    for i, group1 in enumerate(distance_groups):
        for group2 in distance_groups[i + 1:]:
            try:
                if np.allclose(group_data[group1], group_data[group2]):
                    p_value = 1.0
                else:
                    statistic, p_value = wilcoxon(group_data[group1], group_data[group2])
            except Exception as e:
                print(f"Warning: Error in Wilcoxon test for {group1} vs {group2}: {str(e)}")
                p_value = None

            comparisons.append(f'{group1}_vs_{group2}')
            p_values.append(p_value if p_value is not None else 1.0)
            mean_differences.append(
                significance_results['mean_coverages'][group1] -
                significance_results['mean_coverages'][group2]
            )

    valid_p_values = [p for p in p_values if p is not None]
    if valid_p_values:
        adjusted_p_values = benjamini_hochberg(valid_p_values)
    else:
        adjusted_p_values = [1.0] * len(p_values)

    for comp, p_val, adj_p_val, mean_diff in zip(comparisons, p_values, adjusted_p_values, mean_differences):
        significance_results['p_values'][comp] = {
            'p_value': p_val if p_val is not None else 'NA',
            'adjusted_p_value': adj_p_val,
            'significant': adj_p_val < 0.05 if p_val is not None else False,
            'mean_difference': mean_diff,
            'identical_groups': p_val == 1.0 and mean_diff == 0
        }

    return significance_results


def analyze_coverage_significance(coverage_stats: dict) -> dict:
    """
    Performs statistical significance testing on coverage values between cases
    using Wilcoxon signed-rank test.

    Args:
        coverage_stats: Dictionary containing coverage statistics from calculate_coverage()

    Returns:
        dict: Dictionary containing statistical analysis results
    """
    import numpy as np
    from scipy.stats import wilcoxon

    case_coverages = list(coverage_stats['per_case_coverage'].values())

    mean_coverage = np.mean(case_coverages)
    std_coverage = np.std(case_coverages)
    median_coverage = np.median(case_coverages)

    target_coverage = 0.95
    statistic, p_value = wilcoxon(
        [x - target_coverage for x in case_coverages],
        alternative='two-sided'
    )

    return {
        'statistics': {
            'mean': mean_coverage,
            'std': std_coverage,
            'median': median_coverage,
            'n_cases': len(case_coverages)
        },
        'significance_test': {
            'target': target_coverage,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'test_description': f'Wilcoxon signed-rank test against target coverage of {target_coverage}'
        }
    }


def analyze_width_significance(width_stats: dict) -> dict:
    """
    Performs statistical significance testing between different distance groups
    for prediction set widths using Wilcoxon signed-rank test and Benjamini-Hochberg correction.
    Handles cases where groups have identical values.

    Args:
        width_stats: Dictionary containing width statistics per vessel and threshold

    Returns:
        dict: Dictionary containing p-values and adjusted p-values for each comparison
    """

    def benjamini_hochberg(p_values):
        """
        Applies Benjamini-Hochberg correction to p-values.
        """
        n = len(p_values)
        ranked_p_values = sorted(range(n), key=lambda i: p_values[i])
        adjusted = [0] * n

        for rank, i in enumerate(ranked_p_values, 1):
            adjusted[i] = min(1, p_values[i] * n / rank)

        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        return adjusted

    distance_groups = ['<2mm', '<5mm', '<10mm', '<20mm', '>20mm']
    vessel_names = ['Vessel1', 'Vessel2', 'Vessel3', 'Vessel4', 'Vessel5']

    group_data = {}
    for group in distance_groups:
        group_data[group] = [width_stats['per_vessel'][vessel][group] for vessel in vessel_names]

    significance_results = {
        'p_values': {},
        'mean_widths': {},
        'std_widths': {}
    }

    for group in distance_groups:
        significance_results['mean_widths'][group] = np.mean(group_data[group])
        significance_results['std_widths'][group] = np.std(group_data[group])

    comparisons = []
    p_values = []
    mean_differences = []

    for i, group1 in enumerate(distance_groups):
        for group2 in distance_groups[i + 1:]:
            try:
                if np.allclose(group_data[group1], group_data[group2]):
                    p_value = 1.0
                else:
                    statistic, p_value = wilcoxon(group_data[group1], group_data[group2])
            except Exception as e:
                print(f"Warning: Error in Wilcoxon test for {group1} vs {group2}: {str(e)}")
                p_value = None

            comparisons.append(f'{group1}_vs_{group2}')
            p_values.append(p_value if p_value is not None else 1.0)
            mean_differences.append(
                significance_results['mean_widths'][group1] -
                significance_results['mean_widths'][group2]
            )

    valid_p_values = [p for p in p_values if p is not None]
    if valid_p_values:
        adjusted_p_values = benjamini_hochberg(valid_p_values)
    else:
        adjusted_p_values = [1.0] * len(p_values)

    for comp, p_val, adj_p_val, mean_diff in zip(comparisons, p_values, adjusted_p_values, mean_differences):
        significance_results['p_values'][comp] = {
            'p_value': p_val if p_val is not None else 'NA',
            'adjusted_p_value': adj_p_val,
            'significant': adj_p_val < 0.05 if p_val is not None else False,
            'mean_difference': mean_diff,
            'identical_groups': p_val == 1.0 and mean_diff == 0
        }

    return significance_results


def main():
    predset_dir = r"your/path"
    gt_dir = r"your/path"
    log_file = predset_dir + r"\analysis.txt"

    coverage_stats = calculate_coverage(predset_dir, gt_dir)
    vessel_coverage_stats = calculate_vessel_distance_coverage(predset_dir, gt_dir)
    width_stats = calculate_vessel_distance_width(predset_dir, gt_dir)

    distance_significance = analyze_distance_significance(vessel_coverage_stats)
    coverage_significance = analyze_coverage_significance(coverage_stats)
    width_significance = analyze_width_significance(width_stats)

    with open(log_file, 'w') as f:
        f.write(f"Coverage Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prediction Set Directory: {predset_dir}\n")
        f.write(f"Ground Truth Directory: {gt_dir}\n\n")

        # Write coverage statistics with significance analysis
        f.write("OVERALL COVERAGE STATISTICS\n")
        f.write("=" * 30 + "\n")
        f.write(f"Overall Coverage: {coverage_stats['overall_coverage']:.3f}\n")
        f.write(f"Average Case Coverage: {coverage_stats['average_case_coverage']:.3f}\n")
        f.write(f"Total Label 10 Pixels: {coverage_stats['total_label10_pixels']}\n")
        f.write(f"Total Covered Pixels: {coverage_stats['covered_label10_pixels']}\n\n")

        # Add new coverage significance results
        f.write("COVERAGE SIGNIFICANCE ANALYSIS\n")
        f.write("=" * 30 + "\n")
        f.write(f"Number of cases: {coverage_significance['statistics']['n_cases']}\n")
        f.write(
            f"Mean coverage: {coverage_significance['statistics']['mean']:.3f} ± {coverage_significance['statistics']['std']:.3f}\n")
        f.write(f"Median coverage: {coverage_significance['statistics']['median']:.3f}\n")
        f.write(f"\nSignificance Test Results:\n")
        f.write(f"Test: {coverage_significance['significance_test']['test_description']}\n")
        f.write(f"p-value: {coverage_significance['significance_test']['p_value']:.4f}\n")
        f.write(
            f"Significant difference: {'Yes' if coverage_significance['significance_test']['significant'] else 'No'}\n\n")

        f.write("\nPER-CASE COVERAGE\n")
        f.write("=" * 20 + "\n")
        for case, coverage in sorted(coverage_stats['per_case_coverage'].items()):
            f.write(f"{case}: {coverage:.3f}\n")

        f.write("VESSEL DISTANCE COVERAGE STATISTICS\n")
        f.write("=" * 30 + "\n")

        for vessel in vessel_coverage_stats['per_vessel']:
            f.write(f"\n{vessel}:\n")
            f.write("-" * 20 + "\n")
            for thresh, stats in vessel_coverage_stats['per_vessel'][vessel].items():
                f.write(f"{thresh}: {stats['coverage']:.3f}")
                f.write(f" (Covered: {stats['covered_pixels']}/{stats['total_pixels']})\n")

        f.write("\nAverage Across Vessels:\n")
        f.write("-" * 20 + "\n")
        for thresh, coverage in vessel_coverage_stats['average'].items():
            f.write(f"{thresh}: {coverage:.3f}\n")

        f.write("\nDISTANCE GROUP SIGNIFICANCE ANALYSIS (Benjamini-Hochberg corrected)\n")
        f.write("=" * 35 + "\n\n")

        f.write("Mean Coverages (± std):\n")
        f.write("-" * 25 + "\n")
        for group, mean in distance_significance['mean_coverages'].items():
            std = distance_significance['std_coverages'][group]
            f.write(f"{group}: {mean:.3f} ± {std:.3f}\n")
        f.write("\n")

        f.write("Pairwise Comparisons:\n")
        f.write("-" * 25 + "\n")
        for comparison, results in distance_significance['p_values'].items():
            f.write(f"{comparison}:\n")
            f.write(f"  Original p-value: {results['p_value']:.4f}\n")
            f.write(f"  Adjusted p-value: {results['adjusted_p_value']:.4f}\n")
            f.write(f"  Mean difference: {results['mean_difference']:.3f}\n")
            f.write(f"  Significant: {'Yes' if results['significant'] else 'No'}\n")
            f.write("\n")

        f.write("\nPREDICTION SET WIDTH ANALYSIS\n")
        f.write("=" * 30 + "\n")

        # Per-vessel statistics
        f.write("\nPer-vessel Width:\n")
        f.write("-" * 25 + "\n")
        for vessel in width_stats['per_vessel']:
            f.write(f"\n{vessel}:\n")
            for thresh in ['<2mm', '<5mm', '<10mm', '<20mm', '>20mm']:
                f.write(f"{thresh}: {width_stats['per_vessel'][vessel][thresh]:.3f}\n")

        # Average across vessels
        f.write("\nAverage Width Across Vessels:\n")
        f.write("-" * 25 + "\n")
        for thresh in ['<2mm', '<5mm', '<10mm', '<20mm', '>20mm']:
            f.write(f"{thresh}: {width_stats['average'][thresh]:.3f}\n")

        f.write("\nWIDTH SIGNIFICANCE ANALYSIS\n")
        f.write("=" * 30 + "\n")

        f.write("\nMean Widths (± std):\n")
        f.write("-" * 25 + "\n")
        for group, mean in width_significance['mean_widths'].items():
            std = width_significance['std_widths'][group]
            f.write(f"{group}: {mean:.3f} ± {std:.3f}\n")

        f.write("\nPairwise Comparisons:\n")
        f.write("-" * 25 + "\n")
        for comparison, results in width_significance['p_values'].items():
            f.write(f"{comparison}:\n")
            f.write(f"  Original p-value: {results['p_value']:.4f}\n")
            f.write(f"  Adjusted p-value: {results['adjusted_p_value']:.4f}\n")
            f.write(f"  Mean difference: {results['mean_difference']:.3f}\n")
            f.write(f"  Significant: {'Yes' if results['significant'] else 'No'}\n")
            f.write("\n")

    print(f"Analysis complete. Results written to {log_file}")


if __name__ == "__main__":
    main()
