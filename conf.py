from typing import List, Tuple, Dict
from glob import glob
import numpy as np
import nibabel as nib
import os


def compute_dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.sum(pred * gt)
    if intersection == 0:
        return 0.0
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt))


class MultiLabelConformalPredictor:
    def __init__(self, alpha_max: float = 1.5, beta: float = 0.2):
        self.alpha_max = alpha_max
        self.beta = beta

        self.anatomical_labels = {
            0: 'background',
            1: 'kidney_right',
            2: 'kidney_left',
            3: 'adrenal_right',
            4: 'adrenal_left',
            5: 'spleen',
            6: 'liver',
            7: 'gallbladder',
            8: 'pancreas',
            9: 'duodenum',
            10: 'tumor'
        }

        self.vessel_weights = {
            'CeTr': 1.5,  # Celiac trunk
            'HA': 1.5,  # Hepatic artery
            'SMA': 1.5,  # Superior mesenteric artery
            'PV': 1.2,  # Portal vein
            'SMV': 1.2  # Superior mesenteric vein
        }

        self.vessel_indices = {
            'CeTr': 0,
            'HA': 1,
            'SMA': 2,
            'PV': 3,
            'SMV': 4
        }

    def load_data(self, gt_file: str, prob_file: str, distance_file: str, tumor_distance_file: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        gt_nib = nib.load(gt_file)
        true_seg = gt_nib.get_fdata().astype(int)

        probs_data = np.load(prob_file)
        probs = probs_data['probabilities']

        distances = np.load(distance_file)

        tumor_distances = np.load(tumor_distance_file)

        if true_seg.shape != probs.shape[1:] or true_seg.shape != distances.shape[
                                                                  1:] or true_seg.shape != tumor_distances.shape:
            raise ValueError("Dimension mismatch after reordering")

        return probs, true_seg, distances, tumor_distances

    def compute_vessel_distances_from_array(self, distances: np.ndarray) -> Dict[str, np.ndarray]:
        if distances.shape[0] != len(self.vessel_indices):
            raise ValueError(f"Expected {len(self.vessel_indices)} vessel channels")

        distances = distances / 10.0  # distances in cm
        min_dist = np.min(distances, axis=0)
        argmin_dist = np.argmin(distances, axis=0)

        index_to_weight = np.zeros(len(self.vessel_indices), dtype=float)
        for vessel, idx in self.vessel_indices.items():
            index_to_weight[idx] = self.vessel_weights[vessel]

        vessel_weights_array = index_to_weight[argmin_dist]

        distance_maps = {}
        for vessel_name, index in self.vessel_indices.items():
            distance_maps[vessel_name] = distances[index]

        return distance_maps, min_dist, vessel_weights_array

    def compute_anatomical_weights(self, min_dist, vessel_weights, probs, tumor_dist, lam=1, alpha=10.0):
        tumor_dist = tumor_dist / 10.0

        dist_percentile = np.percentile(np.abs(min_dist), 95)
        clipped_dist = np.clip(min_dist, -dist_percentile, dist_percentile)
        scaled_dist = clipped_dist / dist_percentile  # in [-1,1]

        log_probs = -np.log(probs + 1e-5)
        prob_percentile = np.percentile(log_probs, 95)
        clipped_probs = np.clip(log_probs, 0, prob_percentile)
        scaled_probs = clipped_probs / prob_percentile  # in [0,1]

        ww = vessel_weights * ((scaled_dist * scaled_probs) + (lam * tumor_dist))
        return 1 / (1 + np.exp(-alpha * ww))

    def compute_nonconformity_scores(self, probs: np.ndarray, true_seg: np.ndarray, distances: np.ndarray,
                                     tumor_distances: np.ndarray, test=False):

        distance_maps, min_dist, vessel_weights = self.compute_vessel_distances_from_array(distances)
        scores = {}

        for label in range(0, 11):
            label_idx = label if label < probs.shape[0] else 0
            probs_label = probs[label_idx]

            if np.sum(true_seg == label) > 0:
                base_score = (1 - probs_label)
                if label == 10:
                    final_score = base_score

                    if test:
                        anatomical_weights = self.compute_anatomical_weights(min_dist, vessel_weights, probs_label,
                                                                             tumor_distances)
                        scores[label] = (final_score * anatomical_weights).flatten()
                    else:
                        tp_mask = (true_seg == label)
                        scores[label] = final_score[tp_mask]
                else:
                    scores[label] = base_score.flatten() if test else base_score[true_seg == label]

        return scores

    def compute_quantiles(self, calibration_files: List[Tuple[str, str, str, str]],
                          quantile: float = 0.9) -> Dict[int, float]:
        all_scores = {label: [] for label in self.anatomical_labels.keys()}
        processed_files = 0

        for gt_file, prob_file, distance_file, tumor_dist_file in calibration_files:

            try:
                probs, true_seg, distances, tumor_dist = self.load_data(gt_file, prob_file, distance_file,
                                                                        tumor_dist_file)
                scores = self.compute_nonconformity_scores(probs, true_seg, distances, tumor_dist)

                for label, label_scores in scores.items():
                    all_scores[label].extend(label_scores)
                processed_files += 1

            except Exception as e:
                print(f"Error processing files: {str(e)}")
                continue

        quantiles = {}
        for label in self.anatomical_labels.keys():
            if len(all_scores[label]) > 0:
                quantiles[label] = np.percentile(all_scores[label], quantile * 100)

        return quantiles


def get_file_lists(directory: str) -> List[Tuple[str, str, str]]:
    gt_files = sorted(glob(os.path.join(directory, 'GT*.nii.gz')))

    files = []
    for gt_file in gt_files:
        basename = gt_file[gt_file.find('GT') + 2:gt_file.find('.nii.gz')]

        prob_file = os.path.join(directory, f'ORGAN_tumour{basename}_probabilities.npz')
        distance_file = os.path.join(directory, f'DISTANCE{basename}.npy')
        tumordistance_file = os.path.join(directory, f'DISTANCETUMOUR{basename}.npy')

        if not os.path.exists(prob_file):
            print(f"Warning: Probability file not found")
            continue
        if not os.path.exists(distance_file):
            print(f"Warning: Distance file not found")
            continue
        if not os.path.exists(tumordistance_file):
            print(f"Warning: Tumor-Distance file not found")
            continue

        files.append((gt_file, prob_file, distance_file, tumordistance_file))

    print(f"\nFound {len(files)} complete sets of files")
    return files


def split_files(files: List[Tuple[str, str, str]], n_calibration: int = 20) -> Tuple[
    List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    sorted_files = sorted(files)
    calibration_files = sorted_files[:n_calibration]
    test_files = sorted_files[n_calibration:]

    print(f"\nSplit {len(files)} files into:")
    print(f"Calibration set: {len(calibration_files)} files")
    print(f"Test set: {len(test_files)} files")

    return calibration_files, test_files


def evaluate_test_set(predictor: MultiLabelConformalPredictor,
                      test_files: List[Tuple[str, str, str]],
                      quantiles: Dict[int, float],
                      output_dir: str):
    print("\nEvaluating test set...")

    results = {label: {'file_names': [], 'n_voxels': [], 'coverage': [], 'dice_scores': []}
               for label in predictor.anatomical_labels.keys()}

    for gt_file, prob_file, distance_file, tumor_dist_file in test_files:
        probs, true_seg, distances, tumor_dist = predictor.load_data(gt_file, prob_file, distance_file, tumor_dist_file)
        scores = predictor.compute_nonconformity_scores(probs, true_seg, distances, tumor_dist, test=True)

        prediction_set = np.zeros((len(predictor.anatomical_labels), *true_seg.shape), dtype=np.uint8)
        gt_nib = nib.load(gt_file)

        label_count = np.zeros(true_seg.shape)

        for i, (label, label_name) in enumerate(predictor.anatomical_labels.items()):
            if label in scores and label in quantiles:
                score_values = scores[label].reshape(true_seg.shape)
                quantile = quantiles[label]

                score_mask = score_values <= quantile

                binary_pred = score_mask.astype(np.uint8)

                label_mask = (true_seg == label)
                label_count += binary_pred

                prediction_set[i] = binary_pred * label

                binary_nifti = nib.Nifti1Image(binary_pred, gt_nib.affine)
                base_name = os.path.splitext(os.path.splitext(os.path.basename(gt_file))[0])[0][2:]
                binary_file = os.path.join(output_dir, f'{base_name}_label{label}_pred.nii.gz')
                nib.save(binary_nifti, binary_file)

                coverage = np.mean(binary_pred[label_mask])
                dice_score = compute_dice_score(binary_pred, label_mask)

                results[label]['file_names'].append(os.path.basename(gt_file))
                results[label]['n_voxels'].append(np.sum(label_mask))
                results[label]['coverage'].append(coverage)
                results[label]['dice_scores'].append(dice_score)

        base_name = os.path.splitext(os.path.splitext(os.path.basename(gt_file))[0])[0][2:]
        pred_file = os.path.join(output_dir, f'{base_name}_predset.npy')
        np.save(pred_file, prediction_set)

    output_file = os.path.join(output_dir, 'multilabel_test_results.txt')
    with open(output_file, 'w') as f:
        f.write("Multi-Label Test Set Results\n")
        f.write("==========================\n\n")

        for label in predictor.anatomical_labels.keys():
            if label in results and len(results[label]['coverage']) > 0:
                f.write(f"\n{predictor.anatomical_labels[label]} (Label {label}):\n")
                f.write(
                    f"Mean coverage: {np.mean(results[label]['coverage']):.4f} ± {np.std(results[label]['coverage']):.4f}\n")
                f.write(
                    f"Mean Dice score: {np.mean(results[label]['dice_scores']):.4f} ± {np.std(results[label]['dice_scores']):.4f} "
                    f"Min: {np.min(results[label]['dice_scores']):.4f} Max: {np.max(results[label]['dice_scores']):.4f}\n")
                f.write(f"Number of cases: {len(results[label]['file_names'])}\n")

    return results


def main(calibration_dir: str, confidence_level: float = 0.1, output_dir: str = None):
    """
    Run the conformal prediction analysis

    Args:
        calibration_dir (str): Directory containing the calibration data
        confidence_level (float): Confidence level for the prediction (0.05-0.95)
        output_dir (str, optional): Output directory. If None, will be created based on confidence level
    """
    if output_dir is None:
        output_dir = f'your_path/multilabel_results_sacp{confidence_level:.2f}'

    print(f"\nStarting multi-label anatomically-aware conformal prediction")
    print(f"Calibration directory: {calibration_dir}")
    print(f"Confidence level: {confidence_level}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    predictor = MultiLabelConformalPredictor()

    all_files = get_file_lists(calibration_dir)
    calibration_files, test_files = split_files(all_files, n_calibration=10)

    try:
        print("\nComputing quantiles using calibration set...")
        quantiles = predictor.compute_quantiles(calibration_files, quantile=confidence_level)

        output_file = os.path.join(output_dir, 'anatomical_nonconformity_quantiles.txt')
        with open(output_file, 'w') as f:
            f.write("Multi-label anatomically-aware nonconformity quantiles:\n")
            f.write("=============================================\n\n")
            f.write(f"Computed using {len(calibration_files)} calibration cases\n")
            f.write(f"Confidence level: {confidence_level}\n\n")
            for label, quantile in quantiles.items():
                f.write(f"{predictor.anatomical_labels[label]} (Label {label}): {quantile:.6f}\n")

        results = evaluate_test_set(predictor, test_files, quantiles, output_dir)
        print("\nEvaluation complete!")

    except Exception as e:
        print(f"\nError during computation: {str(e)}")
        raise


if __name__ == "__main__":
    folder_path = r'your/path'
    main(folder_path)
