# Experimental Protocol

This document provides a comprehensive mapping between the experiments reported in the paper and their corresponding configurations, commands, and loss functions in this repository.

## Overview

This repository implements a pose estimation pipeline for fish keypoint detection using:
- **Dataset**: COCO-formatted fish keypoint annotations (20 keypoints)
- **Paradigm**: Top-Down pose estimation
- **Model**: ViTPose with Vision Transformer (ViT-Base) backbone
- **Framework**: MMPose (OpenMMLab)

## Dataset Structure

The experiments use COCO-format annotations with 20 keypoints per fish:
- Training set: `fish20kpt_all_train_2nd-run.json`
- Validation set: `fish20kpt_all_val_2nd-run.json`
- Data root: `/data/Datasets/Fish/CavFish`

### Keypoint Definition
The model predicts 20 keypoints (kp1-kp20) corresponding to anatomical landmarks on the fish body.

## Core Experimental Conditions

The repository implements **four primary experimental conditions** that differ in loss functions and logging mechanisms:

### 1. **Baseline** (Standard MSE Loss)
**Configuration file**: [`configs/experiment1_baseline_mse.py`](configs/experiment1_baseline_mse.py)

**Loss function**: `KeypointMSELoss`
- Standard Mean Squared Error loss on heatmaps
- `use_target_weight=True` (per-keypoint weighting)

**Key characteristics**:
- Head type: `HeatmapHead` (standard MMPose head)
- No phenotypic loss term
- Validation interval: every 10 epochs (`val_interval=10`)

**Training command**:
```bash
python tools/train.py \
    configs/experiment1_baseline_mse.py
```

**Output directory**: `/data/Pupils/Josue/weights/Fish/ViTPose_base_2nd-run`

---

### 2. **Baseline + Logging** (MSE with Log Transform)
**Configuration file**: [`configs/experiment2_baseline_logmse.py`](configs/experiment2_baseline_logmse.py)

**Loss function**: `KeypointLogMSELoss`
- Applies logarithmic transformation to MSE: `log(MSE + ε)`
- `use_target_weight=True`
- `eps=1e-6` for numerical stability

**Key characteristics**:
- Head type: `PhenoLossHead` (extended head for loss composition)
- **No active phenotypic loss** (`alpha_pheno=0`)
- Only the log-transformed MSE is active
- Validation interval: every 300 epochs (`val_interval=300`)

**Training command**:
```bash
python tools/train.py \
    configs/experiment2_baseline_logmse.py
```

**Output directory**: `/data/Pupils/Josue/weights/Fish/ViTPose_20kpt_base-log`

---

### 3. **PhenoLoss** (MSE + Phenotypic Distance Loss)
**Configuration file**: [`configs/experiment3_phenoloss_mse.py`](configs/experiment3_phenoloss_mse.py)

**Loss function**: `KeypointMSELoss` + `PhenotypeDistanceLoss`
- Primary: Standard MSE loss on heatmaps
- Secondary: Phenotypic distance loss on 14 keypoint pairs
- Weight: `alpha_pheno=1e-2` (0.01)

**Key characteristics**:
- Head type: `PhenoLossHead`
- **Active phenotypic loss** with α=0.01
- Validation interval: every 300 epochs

**Phenotypic Loss Configuration**:
```python
loss_pheno=dict(
    type='PhenotypeDistanceLoss',
    pairs=[(0,1), (2,3), (4,5), (6,7), (8,9), (10,11),
           (12,3), (0,13), (14,15), (14,3),
           (0,16), (2,17), (18,19)],  # 13 anatomical measurement pairs
    degree_normalize=True,
    scale_by_SL=False,
    normalization="min_gt",      # normalize by minimum GT distance
    percentile=None,
    detach_scale=True,           # detach gradient from scale
    clamp_min=1e-3,              # numerical stability
    clamp_max=None,
    beta=10.0                    # soft-argmax temperature
)
```

**Training command**:
```bash
python tools/train.py \
    configs/experiment3_phenoloss_mse.py
```

**Output directory**: `/data/Pupils/Josue/weights/Fish/ViTPose_20kpt_phenoloss_2nd-run`

---

### 4. **PhenoLoss + Logging** (LogMSE + Phenotypic Distance Loss)
**Configuration file**: [`configs/experiment4_phenoloss_logmse.py`](configs/experiment4_phenoloss_logmse.py)

**Loss function**: `KeypointLogMSELoss` + `PhenotypeDistanceLoss`
- Primary: Log-transformed MSE loss on heatmaps
- Secondary: Phenotypic distance loss (same configuration as experiment 3)
- Weight: `alpha_pheno=1e-2`

**Key characteristics**:
- Head type: `PhenoLossHead`
- **Both** log-MSE and phenotypic loss are active
- Validation interval: every 300 epochs

**Training command**:
```bash
python tools/train.py \
    configs/experiment4_phenoloss_logmse.py
```

**Output directory**: `/data/Pupils/Josue/weights/Fish/ViTPose_20kpt_base-log_2nd-run_pred`

---

## Experimental Comparison Matrix

| Experiment | Config Suffix | Head Type | Primary Loss | Phenotypic Loss | α_pheno | val_interval |
|------------|---------------|-----------|--------------|-----------------|---------|--------------|
| **1. Baseline** | `100etrain.py` | `HeatmapHead` | `KeypointMSELoss` | None | N/A | 10 |
| **2. Baseline+Log** | `base-log.py` | `PhenoLossHead` | `KeypointLogMSELoss` | Disabled | 0 | 300 |
| **3. PhenoLoss** | `phenoloss.py` | `PhenoLossHead` | `KeypointMSELoss` | `PhenotypeDistanceLoss` | 0.01 | 300 |
| **4. PhenoLoss+Log** | `phenoloss-log.py` | `PhenoLossHead` | `KeypointLogMSELoss` | `PhenotypeDistanceLoss` | 0.01 | 300 |

---

## Loss Function Details

### KeypointMSELoss
Standard MMPose heatmap loss:
```
L_mse = (1/N) Σ (pred_heatmap - gt_heatmap)²
```

### KeypointLogMSELoss
Custom loss applying log transformation:
```
L_log = log(L_mse + ε)
```
- Implementation: [`phenolosses/keypoint_log_mse_loss.py`](phenolosses/keypoint_log_mse_loss.py)
- Purpose: Emphasizes relative rather than absolute errors

### PhenotypeDistanceLoss
Enforces anatomical constraints via pairwise distance consistency:
```
L_pheno = Σ w_ij [(d_pred(i,j) - d_gt(i,j)) / scale]²
```
Where:
- `d(i,j)` = Euclidean distance between keypoints i and j
- `w_ij` = degree-normalized weight for pair (i,j)
- `scale` = normalization factor (minimum GT distance among all pairs)

**Implementation**: [`phenolosses/phenoloss_distance_loss.py`](phenolosses/phenoloss_distance_loss.py)

**Head implementation**: [`phenolosses/vitpose_head_phenoloss.py`](phenolosses/vitpose_head_phenoloss.py)

---

## Training Hyperparameters

All experiments share the following training configuration:

### Optimizer
```python
optimizer=dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1
)
```

### Learning Rate Schedule
- Warmup: Linear warmup for 500 iterations (start_factor=0.001)
- Schedule: MultiStepLR with milestones at epochs [170, 200], gamma=0.1
- Total epochs: 300

### Layer Decay (ViT-specific)
```python
layer_decay_rate=0.75  # per-layer learning rate decay
num_layers=12          # ViT-Base has 12 transformer blocks
```

### Data Augmentation
```python
train_pipeline = [
    LoadImage(),
    GetBBoxCenterScale(padding=1.15),
    RandomBBoxTransform(),
    TopdownAffine(input_size=(192, 256), use_udp=True),
    GenerateTarget(encoder=codec),
    PackPoseInputs()
]
```

### Batch Configuration
- Batch size: 32 per GPU
- Number of workers: 8
- Auto-scaling: Enabled (base_batch_size=512)

### Codec
```python
codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),    # input image size (W, H)
    heatmap_size=(48, 64),    # output heatmap size (W, H)
    sigma=2                   # Gaussian kernel sigma
)
```

---

## Evaluation Metrics

All experiments use the same evaluation metrics:

### Test Configuration
```python
test_evaluator = [
    dict(type='PCKAccuracy', thr=0.05),  # Percentage of Correct Keypoints @ 5% threshold
    dict(type='AUC')                      # Area Under Curve
]
```

### Running Evaluation
```bash
python tools/test.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-fish9_8xb32-100etest.py \
    /path/to/checkpoint.pth
```

**Test configuration**: Use the same config as training (e.g., [`configs/experiment1_baseline_mse.py`](configs/experiment1_baseline_mse.py))

---

## Inference

### Single Image Inference
```bash
python demo/image_demo.py \
    <image_path> \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-fish9_8xb32-100etest.py \
    <checkpoint_path> \
    --out-file <output_path>
```

### Batch Inference
```bash
python run_inference_all.py \
    --config configs/experiment1_baseline_mse.py \
    --checkpoint <checkpoint_path> \
    --img-dir <input_directory> \
    --out-dir <output_directory>
```

See [`notebook_inference.ipynb`](notebook_inference.ipynb) for detailed inference examples and visualization.

---

## Repository Scope

This repository contains **only** the components used in the paper experiments. All exploratory and legacy code has been removed to maintain clarity and reproducibility. 

**What's included**:
- ✅ 5 configuration files (4 experiments + 1 test)
- ✅ 3 custom loss implementations (PhenoLoss, PhenoLossHead, KeypointLogMSELoss)
- ✅ 2 demo scripts (single image + batch inference)
- ✅ 1 inference notebook (interactive tutorial)
- ✅ Core utilities (dataset converter, custom transforms)

**All files serve the reproducible experimental protocol.**

---

## Reproducing Paper Results

To reproduce the results from the paper:

1. **Set up the environment** (see main [README.md](README.md))

2. **Prepare the dataset**:
   - Place fish images in `/data/Datasets/Fish/CavFish/`
   - Ensure annotation files are present:
     - `fish20kpt_all_train_2nd-run.json`
     - `fish20kpt_all_val_2nd-run.json`

3. **Download pretrained backbone**:
   ```bash
   mkdir -p checkpoints
   wget -P checkpoints/ <MAE_PRETRAINED_VIT_BASE_URL>
   ```

4. **Run each experiment** using the commands in sections 1-4 above

5. **Evaluate checkpoints** using the test configuration

6. **Compare results** using the metrics:
   - PCK@0.05 (primary metric)
   - AUC (secondary metric)

---

## Mapping to Paper Figures/Tables

<!-- TODO: Fill in after paper structure is finalized -->
- **Table X**: Performance comparison → Results from Experiments 1-4 evaluated on validation set
- **Figure Y**: Loss convergence → Training logs from `work_dir` for each experiment
- **Figure Z**: Qualitative results → Generated using inference scripts on test images

---

## Questions and Support

For questions about reproducing these experiments, please:
1. Check this document first
2. Review the specific configuration file
3. Examine the loss function implementation
4. Open an issue on the repository with details about your setup

---

## Citation

If you use this code or experimental protocol, please cite:

```bibtex
@article{PAPER_REFERENCE,
  title={},
  author={},
  journal={},
  year={2026}
}
```

---

*Last updated: February 2026*
