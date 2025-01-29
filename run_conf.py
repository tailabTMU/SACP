# run_parallel_analysis.py
import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count
import datetime
from pathlib import Path
import logging
from conf import main as run_conformal_prediction


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('parallel_analysis.log')
        ]
    )


def run_single_analysis(params: tuple) -> tuple:
    """Run analysis for a single confidence level"""
    conf_level, data_folder = params
    try:
        conf_str = f"{conf_level:.2f}"
        output_dir = f"PANsacp/multilabel_results_sacp{conf_str}"

        logging.info(f"Starting analysis for confidence level {conf_str}")

        run_conformal_prediction(
            calibration_dir=data_folder,
            confidence_level=conf_level,
            output_dir=output_dir
        )

        logging.info(f"Successfully completed analysis for confidence level {conf_str}")
        return (conf_level, True, "", "")

    except Exception as e:
        logging.error(f"Exception occurred for confidence level {conf_str}: {str(e)}")
        return (conf_level, False, "", str(e))


def combine_results(confidence_levels: list):
    """Combine results from all analyses into a single summary file"""
    summary_file = Path("PANsacp/combined_summary.txt")

    with open(summary_file, 'w') as f:
        f.write("Combined Summary of All Confidence Levels\n")
        f.write("=======================================\n\n")
        f.write(f"Analysis completed at: {datetime.datetime.now()}\n\n")

        for conf_level in confidence_levels:
            conf_str = f"{conf_level:.2f}"
            result_file = Path(f"PANsacp/multilabel_results_sacp{conf_str}/multilabel_test_results.txt")

            f.write(f"\nConfidence Level: {conf_str}\n")
            f.write("------------------------\n")

            if result_file.exists():
                with open(result_file, 'r') as rf:
                    f.write(rf.read())
                f.write("\n")
            else:
                f.write(f"Results not found for confidence level {conf_str}\n")

    logging.info(f"Created summary file at: {summary_file}")


def main(data_folder: str):
    """Main function to run parallel analyses"""
    setup_logging()

    os.makedirs("PANsacp", exist_ok=True)

    confidence_levels = np.arange(0.05, 1.0, 0.05)
    params = [(conf_level, data_folder) for conf_level in confidence_levels]

    n_processes = max(1, cpu_count() - 1)
    logging.info(f"Running analyses using {n_processes} processes")

    start_time = datetime.datetime.now()
    logging.info(f"Starting parallel analysis at: {start_time}")

    with Pool(processes=n_processes) as pool:
        results = pool.map(run_single_analysis, params)

    successful = 0
    failed = 0
    for conf_level, success, stdout, stderr in results:
        if success:
            successful += 1
        else:
            failed += 1
            logging.error(f"Failed analysis for confidence level {conf_level:.2f}")

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    logging.info(f"\nAnalysis Summary:")
    logging.info(f"Total analyses: {len(confidence_levels)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Total duration: {duration}")

    if successful > 0:
        logging.info("Creating combined summary file...")
        combine_results(confidence_levels)

    logging.info("All processes completed!")


if __name__ == "__main__":
    DATA_FOLDER = r"your/path"

    if not os.path.exists(DATA_FOLDER):
        logging.error(f"Data folder not found: {DATA_FOLDER}")
        sys.exit(1)

    main(DATA_FOLDER)