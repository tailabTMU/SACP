# Anatomically-Aware Conformal Prediction for Pancreatic Cancer Segmentation

This repository implements SACP (Spatially-Aware Conformal Prediction), a novel framework that extends conformal prediction to incorporate varying spatial importance in medical image segmentation. While traditional conformal prediction assumes uniform uncertainty across prediction regions, SACP specifically addresses the challenges in pancreatic ductal adenocarcinoma (PDAC) segmentation where different spatial regions - particularly near critical vascular structures - demand distinct levels of certainty.

## Data Requirements

The following files are required for each case:
- Ground truth segmentation files (`GT*.nii.gz`)
- Vessel segmentation files (`VESSELS*.nii.gz`)
- Probability maps (`ORGAN_tumour*_probabilities.npz`)
- Vessel distance maps (`DISTANCE*.npy`)
- Tumor distance maps (`DISTANCETUMOUR*.npy`)

## Pipeline

### 1. Data Preparation
```bash
# Optional: Crop data for computational efficiency
python crop.py

# Generate vessel distance maps
python get_vessel_distances.py

# Generate tumor distance maps
python get_tumour_distances.py
```

### 2. Conformal Prediction

```bash
Option A: Multiple Thresholds
Run conformal prediction for thresholds between 0.05 and 0.95 (0.05 intervals):
python run_conf.py

Option B: Single Threshold
Set desired threshold in conf.py and run:
python conf.py
```

### 3. Evaluation
Generate evaluation report on the test set:
```bash
For multiple thresholds:
python run_eval.py

For single threshold:
python eval.py
```

File Structure
conf.py: Core conformal prediction implementation
crop.py: Data preprocessing for computational efficiency
eval.py: Evaluation metrics for single threshold
figure2.py, figure3.py: Visualization scripts
get_tumour_distances.py: Generate tumor distance maps
get_vessel_distances.py: Generate vessel distance maps
run_conf.py: Batch processing for multiple thresholds
run_eval.py: Batch evaluation for multiple thresholds

### Requirements

numpy>=1.20.0
nibabel>=3.2.0  
scipy>=1.7.0    
tqdm>=4.60.0    
pathlib>=1.0.0
black        
pylint        
pytest 

## Citation
If you use this code in your research, please cite: TODO
