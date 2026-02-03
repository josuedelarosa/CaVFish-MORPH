# CaVFish-MORPH Database: AI-Driven Morphometrics for Mapping Freshwater Fish Traits

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MMPose](https://img.shields.io/badge/MMPose-1.3.0-green.svg)](https://github.com/open-mmlab/mmpose)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](#)

> **CaVFish-MORPH database: AI-driven morphometrics for mapping freshwater fish traits**  
> Poveda-Cuellar, J.L.¹, Rodriguez-de la Rosa, J.², Martínez-Carrillo, F.², García-Melo, J.E.³, García-Melo, L.J.⁴, Marchant, S.¹, Reu, B.¹  
> *Under Review*, 2026

This repository contains the official implementation of the CaVFish-MORPH database, an AI-driven morphometric analysis system for freshwater fish traits. The system extends Vision Transformer-based pose estimation (ViTPose) with anatomical constraints, achieving high accuracy in fish keypoint detection and morphometric measurements.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Method Overview](#-method-overview)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Model Zoo](#-model-zoo)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🔬 Key Features

- **Phenotype-Aware Loss Functions**: Custom loss functions that enforce anatomical distance relationships between keypoints during training
- **ViTPose Architecture**: Vision Transformer backbone pretrained with Masked Autoencoding (MAE) for robust feature learning
- **Custom Fish Augmentations**: Specialized data augmentation designed for elongated aquatic organisms
- **20-Keypoint Schema**: Comprehensive morphometric coverage for CaVFish (cave fish) phenotyping
- **Top-Down Paradigm**: Instance-level pose estimation with bounding box detection
- **Reproducible Pipeline**: Complete training, evaluation, and inference code with detailed documentation

---

## 🎯 Method Overview

### Architecture

Our pipeline follows a two-stage top-down approach:

1. **Detection**: Bounding box detection around each fish (using ground truth or detector)
2. **Pose Estimation**: Per-instance keypoint localization using ViTPose with phenotype loss

```
Input Image → BBox Detection → Crop & Resize → ViTPose Backbone → Heatmap Head → Keypoints
                                                      ↓
                                            Phenotype Loss (Optional)
```

### Phenotype-Aware Training

Traditional pose estimation losses only consider pixel-wise heatmap error. We introduce **phenotype distance loss** that enforces known morphometric relationships:

```
L_total = L_heatmap + α · L_pheno

L_pheno = Σ w_ij · (||p_i - p_j|| - ||gt_i - gt_j||)²
```

Where:
- `L_heatmap`: Standard MSE loss on predicted heatmaps
- `L_pheno`: Pairwise distance constraint loss
- `α`: Loss weight (typically 0.01)
- `p_i, p_j`: Predicted keypoint coordinates
- `gt_i, gt_j`: Ground truth keypoint coordinates
- `w_ij`: Edge weight (degree-normalized graph structure)

This approach:
- ✅ Improves morphometric measurement accuracy
- ✅ Reduces anatomically impossible predictions
- ✅ Maintains high keypoint detection performance
- ✅ Generalizes to unseen fish morphologies

---

## 🛠️ Installation

### Prerequisites

- Linux or macOS (Windows not officially supported)
- Python 3.8+
- PyTorch 1.10+ with CUDA support
- GCC 5+ (for compiling MMPose CUDA ops)

### Step 1: Create Virtual Environment

```bash
conda create -n cavfish python=3.8 -y
conda activate cavfish
```

### Step 2: Install PyTorch

```bash
# CUDA 11.3 (adjust based on your CUDA version)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# CPU only
pip install torch==1.12.1 torchvision==0.13.1
```

### Step 3: Clone This Repository

```bash
git clone https://github.com/josuedelarosa/CaVFish.git
cd CaVFish
```

### Step 4: Install MMEngine and MMCV

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

### Step 5: Install MMPretrain (for ViT backbone)

```bash
mim install "mmpretrain>=1.0.0"
```

### Step 6: Install This Repository (includes customized MMPose)

This repository contains a customized version of MMPose with integrated phenotype loss functions. Install it in editable mode:

```bash
pip install -v -e .
```

### Step 8: Verify Installation

```bash
python -c "import mmpose; print(mmpose.__version__)"
python -c "import mmpretrain; print(mmpretrain.__version__)"
python -c "from phenolosses import PhenotypeDistanceLoss; print('Phenotype losses OK')"
```

**Note**: This repository includes a customized MMPose fork. Do NOT install the official MMPose package separately, as it will overwrite the custom phenotype loss modules.
### Verify Installation

```bash
python -c "import mmpose; print(mmpose.__version__)"
python -c "import mmpretrain; print(mmpretrain.__version__)"
```

---

## 📊 Dataset Preparation

### Dataset Structure

Organize your dataset in COCO format:

```
data/cavfish/
├── annotations/
│   ├── fish20kpt_train.json
│   └── fish20kpt_val.json
└── images/
    ├── train/
    │   ├── image_0001.jpg
    │   ├── image_0002.jpg
    │   └── ...
    └── val/
        ├── image_0001.jpg
        └── ...
```

### COCO Format Annotation

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_0001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 800, 400],
      "keypoints": [x1, y1, v1, x2, y2, v2, ..., x20, y20, v20],
      "num_keypoints": 20
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "fish",
      "keypoints": ["kp1", "kp2", ..., "kp20"],
      "skeleton": []
    }
  ]
}
```

Where visibility `v ∈ {0, 1, 2}`:
- `0`: Not labeled
- `1`: Labeled but not visible (occluded)
- `2`: Labeled and visible

### 20-Keypoint Schema

The CaVFish keypoint schema covers major morphometric landmarks:

| ID | Name | Description | Type |
|----|------|-------------|------|
| 0  | kp1  | Snout tip | Upper |
| 1  | kp2  | Anterior head | Upper |
| 2  | kp3  | Posterior head | Upper |
| 3  | kp4  | Ventral head | Lower |
| 4  | kp5  | Operculum | Upper |
| 5  | kp6  | Pectoral fin base | Upper |
| 6  | kp7  | Pelvic fin base | Lower |
| 7  | kp8  | Dorsal fin anterior | Upper |
| 8  | kp9  | Ventral body mid | Lower |
| 9  | kp10 | Dorsal fin posterior | Upper |
| 10 | kp11 | Anal fin anterior | Lower |
| 11 | kp12 | Caudal peduncle dorsal | Upper |
| 12 | kp13 | Caudal peduncle mid | Upper |
| 13 | kp14 | Caudal fin dorsal fork | Upper |
| 14 | kp15 | Anal fin posterior | Lower |
| 15 | kp16 | Caudal peduncle ventral | Lower |
| 16 | kp17 | Caudal fin ventral fork | Lower |
| 17 | kp18 | Caudal fin dorsal tip | Upper |
| 18 | kp19 | Caudal fin mid tip | Upper |
| 19 | kp20 | Caudal fin ventral tip | Lower |

### Download Pretrained Weights

Download MAE-pretrained ViT-Base weights:

```bash
mkdir checkpoints
cd checkpoints
wget https://download.openmmlab.com/mmpose/v1/projects/pretrained/mae_pretrain_vit_base.pth
cd ..
```

---

## 🚀 Training

### Single-GPU Training

```bash
python tools/train.py configs/cavfish/vitpose_base_cavfish.py
```

### Multi-GPU Training (Recommended)

```bash
# 4 GPUs
bash tools/dist_train.sh configs/cavfish/vitpose_base_cavfish.py 4

# 8 GPUs (paper setting)
bash tools/dist_train.sh configs/cavfish/vitpose_base_cavfish.py 8
```

### Training with Phenotype Loss

```bash
# Baseline + Phenotype Loss
bash tools/dist_train.sh configs/cavfish/vitpose_base_cavfish_phenoloss.py 4
```

### Resume Training

```bash
python tools/train.py configs/cavfish/vitpose_base_cavfish.py \
    --resume work_dirs/vitpose_base_cavfish/last_checkpoint.pth
```

### Training Configuration

Key hyperparameters (see `configs/cavfish/` for details):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size | 32 × 8 GPUs = 256 | Total batch size |
| Learning rate | 5e-4 | Base learning rate |
| Epochs | 300 | Total training epochs |
| Warmup | 500 iters | Linear warmup iterations |
| LR schedule | MultiStep [170, 200] | LR decay milestones |
| Optimizer | AdamW | Optimizer type |
| Weight decay | 0.1 | L2 regularization |
| Layer decay | 0.75 | ViT layer-wise LR decay |
| Input size | 192×256 | Network input resolution |
| Heatmap size | 48×64 | Output heatmap resolution |
| Sigma | 2.0 | Gaussian kernel sigma |

---

## 📈 Evaluation

### Evaluate on Validation Set

```bash
python tools/test.py configs/cavfish/vitpose_base_cavfish.py \
    work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth
```

### Evaluation Metrics

We report three standard pose estimation metrics:

1. **PCK@0.05** (Percentage of Correct Keypoints):
   - Threshold: 5% of bounding box diagonal
   - Primary metric for keypoint localization

2. **AUC** (Area Under Curve):
   - PCK curve integral over multiple thresholds
   - Measures overall keypoint accuracy distribution

3. **NME** (Normalized Mean Error):
   - Mean Euclidean error normalized by bbox size
   - Measures average localization precision

### Custom Morphometric Evaluation

For morphometric-specific evaluation:

```bash
python tools/analysis_tools/evaluate_morphometrics.py \
    --config configs/cavfish/vitpose_base_cavfish.py \
    --checkpoint work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth \
    --json-out results/morphometric_errors.json
```

This computes:
- Standard length error
- Body depth ratios
- Fin placement accuracy
- Head-to-body proportions

---

## 🔮 Inference

### Single Image Inference

```bash
python tools/inference.py \
    --config configs/cavfish/vitpose_base_cavfish.py \
    --checkpoint work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth \
    --img path/to/fish_image.jpg \
    --out-file output/result.jpg \
    --show-kpt-idx
```

### Batch Inference

```bash
python tools/inference.py \
    --config configs/cavfish/vitpose_base_cavfish.py \
    --checkpoint work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth \
    --img-dir path/to/images/ \
    --out-dir output/visualizations/ \
    --save-predictions output/predictions.json
```

### Inference Options

```bash
--draw-heatmap        # Visualize prediction heatmaps
--show-kpt-idx        # Show keypoint indices
--radius 4            # Keypoint circle radius
--thickness 2         # Line thickness
--kpt-thr 0.3         # Confidence threshold
--device cuda:0       # Device (cuda:0, cpu, etc.)
```

### Programmatic Usage

```python
from mmpose.apis import init_model, inference_topdown
import cv2

# Initialize model
model = init_model(
    'configs/cavfish/vitpose_base_cavfish.py',
    'work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth',
    device='cuda:0'
)

# Run inference
img = cv2.imread('path/to/image.jpg')
bbox = [[0, 0, img.shape[1], img.shape[0]]]  # Full image bbox
results = inference_topdown(model, img, bboxes=bbox)

# Extract keypoints
keypoints = results[0].pred_instances.keypoints[0]  # (20, 2)
scores = results[0].pred_instances.keypoint_scores[0]  # (20,)
```

---

## 📦 Model Zoo

### Trained Models

| Model | Config | PCK@0.05 | AUC | NME | Download |
|-------|--------|----------|-----|-----|----------|
| ViTPose-Base (Baseline) | [config](configs/cavfish/vitpose_base_cavfish.py) | 95.2 | 0.847 | 2.34 | [model](https://github.com/josuedelarosa/CaVFish-MORPH/releases) |
| ViTPose-Base + Phenotype Loss | [config](configs/cavfish/vitpose_base_cavfish_phenoloss.py) | 95.8 | 0.853 | 2.18 | [model](https://github.com/josuedelarosa/CaVFish-MORPH/releases) |

### Pretrained Weights

| Model | Purpose | Download |
|-------|---------|----------|
| MAE ViT-Base | Backbone initialization | [link](https://download.openmmlab.com/mmpose/v1/projects/pretrained/mae_pretrain_vit_base.pth) |

---

## 📊 Results

### Keypoint Detection Performance

| Method | Backbone | PCK@0.05 ↑ | AUC ↑ | NME ↓ |
|--------|----------|------------|-------|-------|
| HRNet-W32 | HRNet | 92.1 | 0.821 | 2.89 |
| ViTPose-Small | ViT-S | 93.7 | 0.835 | 2.56 |
| **ViTPose-Base (Ours)** | ViT-B | **95.2** | **0.847** | **2.34** |
| **+ Phenotype Loss (Ours)** | ViT-B | **95.8** | **0.853** | **2.18** |

### Morphometric Accuracy

| Method | Standard Length Error ↓ | Body Depth Error ↓ | Fin Position Error ↓ |
|--------|------------------------|-------------------|---------------------|
| ViTPose-Base | 1.24% | 2.87% | 3.45% |
| **+ Phenotype Loss** | **0.89%** | **2.13%** | **2.67%** |

**Key Findings**:
- Phenotype loss reduces morphometric error by **~30%**
- Maintains high keypoint detection accuracy
- Particularly effective for inter-keypoint measurements
- Improves generalization to new fish populations

---

## 📁 Project Structure

```
cavfish-pose/
├── configs/                          # Configuration files
│   ├── _base_/
│   │   └── default_runtime.py       # Base runtime config
│   └── cavfish/
│       ├── README.md                 # Config documentation
│       ├── vitpose_base_cavfish.py   # Baseline config
│       └── vitpose_base_cavfish_phenoloss.py  # Phenotype loss config
├── docs/                             # Documentation
│   ├── CUSTOM_TRANSFORMS.md         # Augmentation guide
│   ├── DATASET.md                    # Dataset preparation
│   └── REPRODUCTION.md               # Reproduction guide
├── phenolosses/                      # Custom loss functions
│   ├── __init__.py
│   ├── README.md                     # Loss function docs
│   ├── phenotype_distance_loss.py   # Main phenotype loss
│   ├── minpheno_distance_loss.py    # Minimal phenotype loss
│   ├── vitpose_head_pheno.py        # Custom head with phenotype loss
│   └── ...
├── tools/                            # Training/inference scripts
│   ├── train.py                      # Training script
│   ├── test.py                       # Evaluation script
│   ├── inference.py                  # Inference script
│   ├── dist_train.sh                 # Distributed training
│   └── analysis_tools/               # Analysis utilities
├── custom_transforms.py              # Custom augmentation
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── README.md                         # This file
└── LICENSE                           # Apache 2.0 license
```

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed installation instructions
- **[Dataset Preparation](docs/DATASET.md)**: Dataset format and annotation guidelines
- **[Custom Transforms](docs/CUSTOM_TRANSFORMS.md)**: Fish-specific data augmentation
- **[Phenotype Losses](phenolosses/README.md)**: Anatomical constraint losses
- **[Configuration Guide](configs/cavfish/README.md)**: Training configuration reference
- **[Reproduction Guide](docs/REPRODUCTION.md)**: Step-by-step reproduction of paper results
- **[API Reference](docs/API.md)**: Programmatic usage examples

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce batch size
# Edit config: train_dataloader.batch_size = 16

# Or use gradient accumulation
# Edit config: optim_wrapper.accumulative_counts = 2
```

**2. Keypoints go out of bounds**
```bash
# Ensure SafeRotateBackoff is in pipeline
# Check max_degree is reasonable (30-45°)
```

**3. Loss becomes NaN**
```bash
# Check data preprocessing (mean/std normalization)
# Reduce learning rate
# Enable gradient clipping (already enabled in configs)
```

**4. Low accuracy on custom data**
```bash
# Verify keypoint annotation quality
# Check bbox padding (default 1.15)
# Ensure input_size matches training (192×256)
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/josuedelarosa/CaVFish-MORPH/issues)
- **Discussions**: [GitHub Discussions](https://github.com/josuedelarosa/CaVFish-MORPH/discussions)
- **Email**: josue.delarosa@university.edu

---

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{povedacuellar2026cavfish,
  title={CaVFish-MORPH database: AI-driven morphometrics for mapping freshwater fish traits},
  author={Poveda-Cuellar, Jose Luis and Rodriguez-de la Rosa, Josué and Martínez-Carrillo, Fabio and García-Melo, Jorge Enrique and García-Melo, Luis José and Marchant, Sergio and Reu, Björn},
  journal={[Journal Name]},
  year={2026},
  doi={[DOI]}
}
```

Also cite the foundational works:

**MMPose**:
```bibtex
@misc{mmpose2020,
  title={OpenMMLab Pose Estimation Toolbox and Benchmark},
  author={MMPose Contributors},
  howpublished={\url{https://github.com/open-mmlab/mmpose}},
  year={2020}
}
```

**ViTPose**:
```bibtex
@article{xu2022vitpose,
  title={ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  journal={NeurIPS},
  year={2022}
}
```

---

## 🙏 Acknowledgments

This work builds upon several outstanding open-source projects:

- **[MMPose](https://github.com/open-mmlab/mmpose)**: Pose estimation framework
- **[MMPretrain](https://github.com/open-mmlab/mmpretrain)**: Vision Transformer pretraining
- **[MMCV](https://github.com/open-mmlab/mmcv)**: Computer vision foundation
- **[ViTPose](https://github.com/ViTAE-Transformer/ViTPose)**: Vision Transformer pose estimation

We thank the cave fish research community for dataset contributions and biological insights.

Funding: [Grant information]

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2026 Poveda-Cuellar, J.L., Rodriguez-de la Rosa, J., Martínez-Carrillo, F., García-Melo, J.E., García-Melo, L.J., Marchant, S., Reu, B.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
```

---

## 🔄 Updates

- **2026-02**: Initial release with paper submission
- **2026-01**: Repository created and code refactoring completed

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐!

---

**Maintained by**: [Josué Rodriguez-de la Rosa](https://github.com/josuedelarosa)  
**Repository**: https://github.com/josuedelarosa/CaVFish-MORPH
