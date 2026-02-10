# CaVFish: Anatomically-Constrained Pose Estimation for Fish Morphometrics

<!-- Badges will be added after publication -->

This repository contains the official implementation of the paper **"[Paper Title]"** published in **[Journal/Conference Name, Year]**.

We present a novel approach to fish pose estimation that incorporates phenotypic distance constraints to improve the accuracy and anatomical plausibility of keypoint predictions. Built on MMPose and ViTPose, our method enforces consistency in inter-landmark distances during training, making it particularly suitable for morphometric analysis in evolutionary biology and aquaculture research.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Experimental Protocol](#experimental-protocol)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

**Problem**: Traditional pose estimation models optimize keypoint localization independently, often producing anatomically implausible predictions (e.g., inconsistent body proportions).

**Solution**: We introduce **PhenoLoss**, a phenotypic distance loss that enforces consistency in pairwise keypoint distances based on anatomical relationships. This approach:
- Maintains standard heatmap-based localization accuracy
- Adds biologically-informed geometric constraints
- Improves performance on morphometric measurements
- Generalizes across fish species with diverse body shapes

**Framework**: 
- **Base Model**: ViTPose (Vision Transformer for Pose Estimation)
- **Dataset Format**: COCO keypoints (20 anatomical landmarks)
- **Paradigm**: Top-down pose estimation
- **Framework**: MMPose (OpenMMLab)

---

## ✨ Key Features

- **📐 Anatomically-Constrained Loss**: PhenoLoss enforces biologically plausible inter-landmark distances
- **🔬 Morphometric-Ready**: Designed for scientific measurements requiring high anatomical accuracy
- **🎯 Flexible Loss Composition**: Easy combination of heatmap losses and phenotypic constraints
- **📊 Reproducible Experiments**: Fully documented experimental protocol with 4 core conditions
- **🛠️ MMPose Integration**: Seamless integration with MMPose ecosystem and pretrained models
- **🐟 Fish-Specific**: 20-keypoint annotation scheme optimized for fish morphology

---

## 🚀 Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.8
- CUDA >= 11.0 (for GPU training)

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/CaVFish.git
cd CaVFish
```

### Step 2: Create Environment
```bash
conda create -n cavfish python=3.8 -y
conda activate cavfish
```

### Step 3: Install PyTorch
```bash
# CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Or CUDA 11.6
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Or CPU only
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install MMPose and Dependencies
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
mim install "mmpretrain>=1.0.0"

# Install this package in editable mode
pip install -e .
```

### Step 5: Download Pretrained Backbone
```bash
mkdir -p checkpoints
cd checkpoints
# Download MAE-pretrained ViT-Base
wget https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_base_20230913.pth
cd ..
```

### Verify Installation
```bash
python -c "import mmpose; print(mmpose.__version__)"
python -c "import torch; print(torch.__version__)"
```

---

## 📊 Dataset Preparation

### Dataset Structure
The repository expects COCO-format annotations with the following structure:

```
/data/Datasets/Fish/CavFish/
├── images/
│   ├── fish001.jpg
│   ├── fish002.jpg
│   └── ...
├── fish20kpt_all_train_2nd-run.json
└── fish20kpt_all_val_2nd-run.json
```

### Annotation Format
Each annotation file follows the COCO keypoint format:
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
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

### Keypoint Definitions
The 20 keypoints correspond to the following anatomical landmarks:

| ID | Name | Description |
|----|------|-------------|
| 0 | kp1 | [Anatomical landmark 1] |
| 1 | kp2 | [Anatomical landmark 2] |
| ... | ... | ... |
| 19 | kp20 | [Anatomical landmark 20] |


## ⚡ Quick Start

### Training

#### Baseline (Standard MSE Loss)
```bash
python tools/train.py \
    configs/experiment1_baseline_mse.py \
    --work-dir work_dirs/baseline
```

#### PhenoLoss (Our Method)
```bash
python tools/train.py \
    configs/experiment3_phenoloss_mse.py \
    --work-dir work_dirs/phenoloss
```

### Evaluation
```bash
python tools/test.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-fish9_8xb32-100etest.py \
    work_dirs/phenoloss/best_AP_epoch_XXX.pth
```

### Inference on Images
```bash
python demo/image_demo.py \
    path/to/your/image.jpg \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-fish9_8xb32-100etest.py \
    work_dirs/phenoloss/best_AP_epoch_XXX.pth \
    --out-file output/result.jpg
```

### Batch Inference
```bash
python run_inference_all.py \
    --config configs/experiment1_baseline_mse.py \
    --checkpoint work_dirs/phenoloss/best_AP_epoch_XXX.pth \
    --img-dir path/to/images/ \
    --out-dir path/to/outputs/
```

---

## 🔬 Experimental Protocol

This repository implements **four core experimental conditions** to evaluate the contribution of different loss components:

| # | Experiment | Primary Loss | Phenotypic Loss | Purpose |
|---|------------|--------------|-----------------|---------|
| 1 | **Baseline** | MSE | ✗ | Standard heatmap loss |
| 2 | **Baseline+Log** | Log(MSE) | ✗ | Logarithmic error weighting |
| 3 | **PhenoLoss** | MSE | ✓ | Our method (MSE + anatomical) |
| 4 | **PhenoLoss+Log** | Log(MSE) | ✓ | Our method with log transform |

### Detailed Documentation
For complete details on:
- Configuration files for each experiment
- Loss function parameters
- Training hyperparameters
- Evaluation protocols
- Mapping to paper results

See **[EXPERIMENTS.md](EXPERIMENTS.md)** for the full experimental protocol.

### Morphometric Measurements

The framework computes **13 anatomically-defined measurements** from the 20 detected keypoints:

| # | Measurement | Keypoints | Description |
|---|-------------|-----------|-------------|
| 1 | **BI** | 0 → 1 | Body Index: Upper Snout Tip → Caudal Peduncle Center |
| 2 | **Bd** | 2 → 3 | Body Depth: Dorsal Body → Pelvic Fin Base |
| 3 | **Hd** | 4 → 5 | Head Depth: Upper Head → Barbel Base |
| 4 | **CPd** | 6 → 7 | Caudal Peduncle Depth: Mid-Dorsal Trunk → Ventral Trunk |
| 5 | **CFd** | 8 → 9 | Caudal Fin Depth: Upper Caudal Base → Lower Caudal Fin Tip |
| 6 | **Ed** | 10 → 11 | Eye Diameter: Eye Center → Posterior Eye Margin |
| 7 | **Eh** | 12 → 3 | Eye Height: Lower Eye Margin → Pelvic Fin Base |
| 8 | **JI** | 0 → 13 | Jaw Index: Upper Snout Tip → Lower Jaw Tip |
| 9 | **PFI** | 14 → 15 | Pelvic Fin Index: Operculum Lower Edge → Pelvic Fin Tip |
| 10 | **PFi** | 14 → 3 | Pelvic Fin insertion: Operculum Lower Edge → Pelvic Fin Base |
| 11 | **HL** | 0 → 16 | Head Length: Upper Snout Tip → Operculum Upper Edge |
| 12 | **DL** | 2 → 17 | Dorsal Length: Dorsal Body → Dorsal Fin Tip |
| 13 | **AL** | 18 → 19 | Anal Length: Anal Fin Base → Anal Fin Tip |

All measurements are provided in both absolute (pixels) and **BI-normalized** (size-independent) formats.

### Key Configuration Differences

**Baseline** (`experiment1_baseline_mse.py`):
```python
head=dict(
    type='HeatmapHead',
    loss=dict(type='KeypointMSELoss', use_target_weight=True)
)
```

**PhenoLoss** (`experiment3_phenoloss_mse.py`):
```python
head=dict(
    type='PhenoLossHead',
    loss=dict(type='KeypointMSELoss', use_target_weight=True),
    loss_pheno=dict(
        type='PhenotypeDistanceLoss',
        pairs=[(0,1), (2,3), (4,5), ...],  # 13 anatomical pairs
        alpha_pheno=1e-2,                   # phenotypic loss weight
        normalization="min_gt"              # scale normalization
    )
)
```

---

## 📁 Repository Structure

```
CaVFish/
├── configs/                          # Configuration files
│   ├── _base_/
│   │   └── default_runtime.py       # Base runtime configuration
│   └── body_2d_keypoint/
│       └── topdown_heatmap/
│           └── coco/
│               ├── experiment1_baseline_mse.py              # [✓] Experiment 1: Baseline
│               ├── experiment2_baseline_logmse.py           # [✓] Experiment 2: Baseline+Log
│               ├── experiment3_phenoloss_mse.py             # [✓] Experiment 3: PhenoLoss
│               └── experiment4_phenoloss_logmse.py          # [✓] Experiment 4: PhenoLoss+Log
│
├── phenolosses/                      # Custom loss implementations
│   ├── __init__.py
│   ├── phenoloss_distance_loss.py    # [✓] PhenotypeDistanceLoss (primary)
│   ├── vitpose_head_phenoloss.py     # [✓] PhenoLossHead (primary)
│   ├── keypoint_log_mse_loss.py     # [✓] KeypointLogMSELoss
│   └── *.py                          # [×] Legacy implementations (not used)
│
├── mmpose/                           # MMPose framework (standard)
│   ├── apis/
│   ├── models/
│   ├── datasets/
│   └── ...
│
├── tools/                            # Training and testing scripts
│   ├── train.py                     # Main training script
│   ├── test.py                      # Main testing script
│   └── dataset_converters/          # Dataset conversion utilities
│
├── demo/                             # Inference demos
│   ├── image_demo.py                # Single image inference
│   └── cavfish_batch_inference.py   # Batch processing
│
├── run_inference_all.py              # Batch inference script
├── notebook_inference.ipynb          # [Tutorial] Inference examples
├── EXPERIMENTS.md                    # [Documentation] Detailed experimental protocol
├── README.md                         # This file
├── CITATION.cff                      # Citation metadata
├── LICENSE                           # License information
└── requirements.txt                  # Python dependencies
```

### Legend
- `[✓]` - Used in paper experiments (reproducible)
- `[×]` - Out-of-scope (retained for reference)
- `[Tutorial]` - Educational/demonstration purpose
- `[Documentation]` - Reference documentation

---

## 📈 Results

### Quantitative Results

Performance on the validation set (PCK@0.05):

| Method | PCK@0.05 (%) | AUC | Avg. Error (px) |
|--------|--------------|-----|-----------------|
| Baseline | XX.X | X.XXX | X.XX |
| Baseline+Log | XX.X | X.XXX | X.XX |
| **PhenoLoss (Ours)** | **XX.X** | **X.XXX** | **X.XX** |
| PhenoLoss+Log | XX.X | X.XXX | X.XX |

<!-- TODO: Fill in actual results from paper -->

### Qualitative Results

![Qualitative comparison](resources/qualitative_results.png)
<!-- TODO: Add figure showing baseline vs. PhenoLoss predictions -->

### Morphometric Accuracy

Our method significantly improves the accuracy of derived morphometric measurements:

| Measurement | Baseline Error | PhenoLoss Error | Improvement |
|-------------|----------------|----------------|-------------|
| Body Index (BI) | X.X% | X.X% | XX% |
| Body Depth (Bd) | X.X% | X.X% | XX% |
| Head Length (HL) | X.X% | X.X% | XX% |
| Eye Diameter (Ed) | X.X% | X.X% | XX% |

<!-- TODO: Add morphometric validation results -->

---

## 🔧 Using PhenoLoss in Your Project

### Integration with Existing Models

To add anatomical constraints to your own pose estimation model:

```python
from phenolosses.phenoloss_distance_loss import PhenotypeDistanceLoss
from phenolosses.vitpose_head_phenoloss import PhenoLossHead

# Define anatomical keypoint pairs
anatomical_pairs = [
    (0, 1),   # head to snout
    (3, 2),   # dorsal to ventral
    # ... add your domain-specific pairs
]

# Configure the loss
loss_pheno = PhenotypeDistanceLoss(
    pairs=anatomical_pairs,
    alpha_pheno=0.01,              # weight of phenotypic term
    normalization="min_gt",        # normalize by minimum GT distance
    degree_normalize=True,         # weight by keypoint connectivity
    beta=10.0                      # soft-argmax temperature
)

# Use the extended head
head = PhenoLossHead(
    loss=dict(type='KeypointMSELoss', use_target_weight=True),
    loss_pheno=loss_pheno,
    alpha_pheno=0.01,
    # ... other head parameters
)
```

### Customization

**Adjusting the anatomical pairs**:
Edit the `pairs` parameter in your config file to match your keypoint topology.

**Tuning the loss weight**:
The `alpha_pheno` parameter controls the strength of anatomical constraints:
- `0.0`: No constraint (pure baseline)
- `0.001-0.01`: Weak guidance (recommended starting point)
- `0.01-0.1`: Strong constraints (may hurt localization if too high)

---

## 📜 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{cavfish2026,
  title={Anatomically-Constrained Pose Estimation for Fish Morphometrics},
  author={Author Names},
  journal={Journal Name},
  year={2026},
  volume={XX},
  pages={XXX-XXX},
  doi={XX.XXXX/XXXXXXX}
}
```

Please also consider citing the underlying frameworks:

```bibtex
@inproceedings{xu2022vitpose,
  title={Vitpose: Simple vision transformer baselines for human pose estimation},
  author={Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  booktitle={NeurIPS},
  year={2022}
}

@misc{mmpose2020,
  title={OpenMMLab Pose Estimation Toolbox and Benchmark},
  author={MMPose Contributors},
  howpublished={\url{https://github.com/open-mmlab/mmpose}},
  year={2020}
}
```

---

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).

**Third-party Components**:
- MMPose: Apache 2.0 License
- ViTPose: Apache 2.0 License

See [LICENSES.md](LICENSES.md) for complete licensing information.

---

## 🙏 Acknowledgments

- **MMPose Team**: For the excellent pose estimation framework
- **OpenMMLab**: For the comprehensive ecosystem of computer vision tools
- **ViTPose Authors**: For the vision transformer architecture
- **[Funding Agency]**: For financial support
- **[Institution/Lab]**: For providing computational resources and fish datasets

---

## 🤝 Contributing

We welcome contributions! If you find bugs or have suggestions:

1. Check existing issues
2. Open a new issue with detailed description
3. Submit a pull request with proposed changes

For major changes, please open an issue first to discuss your ideas.

---

## 📧 Contact

For questions or collaboration inquiries:

- **Primary Contact**: [Name] ([email@domain.com](mailto:email@domain.com))
- **Project Page**: [https://your-project-page.com](https://your-project-page.com)
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/CaVFish/issues)

---

## 🔄 Updates

- **[2026-02]**: Initial public release alongside paper publication
- **[2026-XX]**: Pretrained models released
- **[2026-XX]**: Extended documentation and tutorials

---

*This repository is actively maintained. For the latest updates, please check the [Releases](https://github.com/YOUR_USERNAME/CaVFish/releases) page.*
