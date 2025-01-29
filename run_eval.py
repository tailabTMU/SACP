import os
import numpy as np
from multiprocessing import Pool
import time
from pathlib import Path
from functools import partial
from datetime import datetime


def process_confidence_interval(base_dir, gt_dir, confidence):
    conf_str = f"{confidence:.2f}"

    predset_dir = Path(base_dir) / f"multilabel_results_sacp{conf_str}"

    predset_dir.mkdir(parents=True, exist_ok=True)

    try:
        from eval import (
            calculate_coverage,
            calculate_vessel_distance_coverage,
            calculate_vessel_distance_width,
            analyze_distance_significance,
            analyze_coverage_significance,
            analyze_width_significance
        )

        coverage_stats = calculate_coverage(str(predset_dir), gt_dir)
        vessel_coverage_stats = calculate_vessel_distance_coverage(str(predset_dir), gt_dir)
        width_stats = calculate_vessel_distance_width(str(predset_dir), gt_dir)

        distance_significance = analyze_distance_significance(vessel_coverage_stats)
        coverage_significance = analyze_coverage_significance(coverage_stats)
        width_significance = analyze_width_significance(width_stats)

        log_file = predset_dir / f"analysis_confidence_{conf_str}.txt"

        with open(log_file, 'w') as f:
            f.write(f"Coverage Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Prediction Set Directory: {predset_dir}\n")
            f.write(f"Ground Truth Directory: {gt_dir}\n\n")

            f.write("OVERALL COVERAGE STATISTICS\n")
            f.write("=" * 30 + "\n")
            f.write(f"Overall Coverage: {coverage_stats['overall_coverage']:.3f}\n")
            f.write(f"Average Case Coverage: {coverage_stats['average_case_coverage']:.3f}\n")
            f.write(f"Total Label 10 Pixels: {coverage_stats['total_label10_pixels']}\n")
            f.write(f"Total Covered Pixels: {coverage_stats['covered_label10_pixels']}\n\n")

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

            f.write("\nPer-vessel Width:\n")
            f.write("-" * 25 + "\n")
            for vessel in width_stats['per_vessel']:
                f.write(f"\n{vessel}:\n")
                for thresh in ['<2mm', '<5mm', '<10mm', '<20mm', '>20mm']:
                    f.write(f"{thresh}: {width_stats['per_vessel'][vessel][thresh]:.3f}\n")

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

    except Exception as e:
        return f"Error processing confidence {conf_str}: {str(e)}"


def main():
    base_dir = r"your_path"
    gt_dir = r"your_path"

    confidence_intervals = np.arange(0.05, 1.00, 0.05)
    process_func = partial(process_confidence_interval, base_dir, gt_dir)
    num_cores = os.cpu_count()

    print(f"Starting parallel processing with {num_cores} cores...")
    start_time = time.time()

    with Pool(processes=num_cores) as pool:
        results = pool.map(process_func, confidence_intervals)

    end_time = time.time()
    print(f"\nCompleted all analyses in {end_time - start_time:.2f} seconds")

    summary_file = Path(base_dir) / "confidence_analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Confidence Interval Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total processing time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Number of intervals processed: {len(confidence_intervals)}\n\n")
        for conf, result in zip(confidence_intervals, results):
            f.write(f"Confidence {conf:.2f}: {result}\n")


if __name__ == "__main__":
    main()