# CaVFish Pose Estimation Configurations

This directory contains configuration files for training and evaluating ViTPose models on the CaVFish 20-keypoint dataset.

## Configuration Structure

All configurations follow MMPose's modular design and inherit from `_base_/default_runtime.py`.

## Available Configurations

### 1. Baseline ViTPose
**File**: `vitpose_base_cavfish.py`
- **Purpose**: Baseline ViTPose-Base model with standard MSE loss
- **Architecture**: Vision Transformer (ViT-Base) backbone + Heatmap head
- **Loss**: KeypointMSELoss
- **Use**: Reproducing baseline results from the paper

### 2. ViTPose + Phenotype Loss
**File**: `vitpose_base_cavfish_phenoloss.py`
- **Purpose**: ViTPose with phenotype-aware distance loss (main contribution)
- **Architecture**: ViT-Base + Custom HeatmapHeadPheno
- **Loss**: KeypointMSELoss + PhenotypeDistanceLoss (α=0.01)
- **Use**: Reproducing phenotype-constrained training results

### 3. ViTPose + Minimal Phenotype Loss
**File**: `vitpose_base_cavfish_minphenoloss.py`
- **Purpose**: ViTPose with minimal phenotype constraints
- **Architecture**: ViT-Base + Custom HeatmapHeadMinPheno
- **Loss**: KeypointMSELoss + MinimalPhenotypeDistanceLoss
- **Use**: Ablation study variant

## Dataset Format

All configurations expect COCO-format JSON annotation files:
- Training: `fish20kpt_all_train_2nd-run.json`
- Validation: `fish20kpt_all_val_2nd-run.json`

The dataset should be structured as:
```
/path/to/data_root/
├── fish20kpt_all_train_2nd-run.json
├── fish20kpt_all_val_2nd-run.json
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Training

```bash
# Baseline
python tools/train.py configs/cavfish/vitpose_base_cavfish.py

# With Phenotype Loss
python tools/train.py configs/cavfish/vitpose_base_cavfish_phenoloss.py

# Distributed training (4 GPUs)
bash tools/dist_train.sh configs/cavfish/vitpose_base_cavfish_phenoloss.py 4
```

## Testing

```bash
python tools/test.py configs/cavfish/vitpose_base_cavfish.py \
    work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth
```

## Key Hyperparameters

- **Input size**: 192×256
- **Heatmap size**: 48×64
- **Batch size**: 32 per GPU (256 total for 8 GPUs)
- **Optimizer**: AdamW (lr=5e-4, weight_decay=0.1)
- **Layer decay**: 0.75 (ViT-specific)
- **Training epochs**: 300
- **LR schedule**: Warmup (500 iters) + MultiStep [170, 200]
- **Phenotype loss weight (α)**: 0.01

## Pretrained Weights

The ViT-Base backbone is initialized with MAE pretrained weights:
- `checkpoints/mae_pretrain_vit_base_20230913.pth`

Download from: [MMPretrain Model Zoo](https://github.com/open-mmlab/mmpretrain)

## Keypoint Schema

20 keypoints for cave fish morphometry:
- **Upper body**: kp1, kp2, kp3, kp5, kp6, kp8, kp10, kp12, kp13, kp14, kp18, kp19
- **Lower body**: kp4, kp7, kp9, kp11, kp15, kp16, kp17, kp20

See individual config files for detailed keypoint definitions and visualization settings.
