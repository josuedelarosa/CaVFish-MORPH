# Repository Structure Guide

This document explains the organization of the CaVFish repository and helps you navigate the codebase.

## 📂 Directory Overview

```
CaVFish/
├── 📋 Documentation & Metadata
│   ├── README.md                       # ⭐ Main repository documentation (publication-ready)
│   ├── EXPERIMENTS.md                  # ⭐ Detailed experimental protocol (START HERE)
│   ├── REPOSITORY_STRUCTURE.md         # This file - repository guide
│   ├── QUICK_START.md                  # Quick reference for running experiments

│   ├── CITATION.cff                    # Citation metadata
│   ├── LICENSE                         # Apache 2.0 license
│   └── LICENSES.md                     # Third-party licenses
│
├── ⚙️ Experimental Configurations
│   └── configs/
│       ├── _base_/
│       │   └── default_runtime.py                                  # Base runtime settings
│       └── body_2d_keypoint/topdown_heatmap/coco/
│           ├── ⭐ experiment1_baseline_mse.py              # Experiment 1: Baseline
│           ├── ⭐ experiment2_baseline_logmse.py           # Experiment 2: Baseline+Log
│           ├── ⭐ experiment3_phenoloss_mse.py             # Experiment 3: PhenoLoss
│           └── ⭐ experiment4_phenoloss_logmse.py          # Experiment 4: PhenoLoss+Log
│
├── 🧮 Custom Loss Functions
│   └── phenolosses/
│       ├── __init__.py                         # Module exports
│       ├── ⭐ phenoloss_distance_loss.py        # PhenotypeDistanceLoss (CORE)
│       ├── ⭐ vitpose_head_phenoloss.py         # PhenoLossHead (CORE)
│       └── ✅ keypoint_log_mse_loss.py         # KeypointLogMSELoss
│
├── 🛠️ MMPose Framework (Standard)
│   └── mmpose/
│       ├── apis/              # Inference and training APIs
│       ├── models/            # Model architectures
│       │   ├── backbones/    # Backbone networks
│       │   ├── heads/        # Prediction heads
│       │   └── losses/       # Loss functions
│       ├── datasets/          # Dataset definitions
│       ├── codecs/           # Keypoint encoding/decoding
│       ├── engine/           # Training engines
│       ├── evaluation/       # Evaluation metrics
│       ├── structures/       # Data structures
│       ├── utils/            # Utility functions
│       └── visualization/    # Visualization tools
│
├── 🔧 Tools & Scripts
│   └── tools/
│       ├── ⭐ train.py                         # Main training script
│       ├── ⭐ test.py                          # Main evaluation script
│       ├── dist_train.sh                       # Distributed training
│       ├── dist_test.sh                        # Distributed testing
│       ├── analysis_tools/                     # Analysis utilities
│       ├── misc/                               # Miscellaneous utilities
│       └── dataset_converters/
│           └── ✅ cvat_to_coco_from_split.py   # CVAT XML → COCO JSON
│
├── 🎬 Demos & Inference
│   ├── demo/
│   │   ├── ⭐ image_demo.py                    # Single image inference (PRIMARY)
│   │   └── ✅ cavfish_batch_inference.py       # Batch processing
│   └── ✅ run_inference_all.py                 # Batch inference wrapper script
│
├── 📓 Notebooks & Tutorials
│   └── ⭐ notebook_inference.ipynb             # Inference tutorial with examples
│
├── 🛠️ Utility Scripts
│   ├── custom_transforms.py                    # Custom data augmentation
│   └── cvat_to_coco_from_split.py             # Dataset conversion utility
│
├── 📦 Package Configuration
│   ├── setup.py                # Package setup
│   ├── setup.cfg               # Setup configuration
│   ├── MANIFEST.in             # Package manifest
│   ├── requirements.txt        # Python dependencies
│   └── requirements/           # Detailed requirements
│       ├── runtime.txt         # Runtime dependencies
│       ├── build.txt           # Build dependencies
│       ├── optional.txt        # Optional features
│       └── tests.txt           # Testing dependencies
│
├── 🗂️ Additional Resources
│   ├── resources/              # Images, diagrams, documentation assets
│   ├── projects/               # MMPose projects (RTMPose, etc.)
│   └── tests/                  # Unit tests
│
└── 📊 Metadata
    ├── dataset-index.yml       # Dataset index
    ├── model-index.yml         # Model index
    └── pytest.ini              # Test configuration
```

## Legend

- ⭐ **Critical**: Primary experimental file
- ✅ **Core**: Essential component
- 🛠️ **Utility**: Helper script or tool
- 📁 Folder
- 📄 File

**Note**: This repository contains ONLY the files used in the paper experiments. All exploratory and legacy code has been removed for clarity.

---

## Navigation by Task

### 🎯 I want to reproduce the paper results
**Start here**:
1. [EXPERIMENTS.md](EXPERIMENTS.md) - Complete experimental protocol
2. `configs/experiment*.py` - Configuration files
3. `tools/train.py` - Training script
4. `tools/test.py` - Evaluation script

### 🔬 I want to understand the method
**Read these**:
1. [EXPERIMENTS.md](EXPERIMENTS.md) - High-level overview
2. `phenolosses/phenoloss_distance_loss.py` - Core loss implementation
3. `phenolosses/vitpose_head_phenoloss.py` - Head implementation
4. Configuration files - See how everything connects

### 🚀 I want to run inference on my images
**Use these**:
1. `demo/image_demo.py` - Single image
2. `run_inference_all.py` - Batch processing
3. `notebook_inference.ipynb` - Interactive tutorial

### 📊 I want to prepare my own dataset
**Follow these**:
1. `tools/dataset_converters/cvat_to_coco_from_split.py` - Convert annotations
2. [EXPERIMENTS.md - Dataset Structure](EXPERIMENTS.md#dataset-structure) - Format specification
3. `notebook_inference.ipynb` - Complete example

### 🛠️ I want to integrate PhenoLoss into my project
**Copy these**:
1. `phenolosses/phenoloss_distance_loss.py` - Loss function
2. `phenolosses/vitpose_head_phenoloss.py` - Head module
3. See [README.md](README.md#custom-phenoloss-implementation) for integration guide

### 🧪 I want to explore alternative approaches
**Note**: All exploratory code has been removed. The repository contains only the components used in the paper. If you want to explore variations:
1. Start with the 4 core experiments
2. Modify configuration parameters
3. Create new configs based on documented examples
4. Document your modifications clearly

---

## Key Files Explained

### Configuration Files

Configuration files in MMPose follow a hierarchical structure:

```python
# Base configuration
_base_ = ['../../../_base_/default_runtime.py']

# Model architecture
model = dict(
    type='TopdownPoseEstimator',
    backbone=dict(...),
    head=dict(...)
)

# Loss functions
loss = dict(type='KeypointMSELoss')
loss_pheno = dict(type='PhenotypeDistanceLoss', pairs=[...])

# Dataset
train_dataloader = dict(...)
test_dataloader = dict(...)

# Training schedule
train_cfg = dict(max_epochs=300)
optim_wrapper = dict(optimizer=dict(type='AdamW'))
```

**Key config components**:
- `custom_imports`: Registers custom modules (losses, heads)
- `model`: Defines architecture (backbone + head)
- `codec`: Heatmap encoding/decoding
- `metainfo`: Keypoint definitions
- `train_dataloader`: Training data pipeline
- `optim_wrapper`: Optimizer and learning rate
- `param_scheduler`: Learning rate schedule

### Loss Function Files

**`phenoloss_distance_loss.py`**:
- Defines `PhenotypeDistanceLoss` class
- Computes pairwise distances between keypoints
- Applies normalization and degree weighting
- Registered with `@MODELS.register_module()`

**`vitpose_head_phenoloss.py`**:
- Extends `HeatmapHead` from MMPose
- Adds `loss_pheno` term to standard heatmap loss
- Weighted combination: `loss_total = loss_heatmap + alpha_pheno * loss_pheno`

**`keypoint_log_mse_loss.py`**:
- Variant of MSE loss with log transformation
- `loss = log(MSE + eps)`
- More emphasis on relative errors

### Training and Testing Scripts

**`tools/train.py`**:
```bash
python tools/train.py <config_file> [options]
```
Options:
- `--work-dir`: Output directory for checkpoints and logs
- `--resume`: Resume from checkpoint
- `--amp`: Enable automatic mixed precision

**`tools/test.py`**:
```bash
python tools/test.py <config_file> <checkpoint_file> [options]
```
Options:
- `--out`: Save predictions to file
- `--cfg-options`: Override config parameters

---

## Data Flow

### Training Pipeline
```
Images + Annotations
    ↓
LoadImage, GetBBoxCenterScale
    ↓
RandomBBoxTransform (augmentation)
    ↓
TopdownAffine (crop & resize to 192×256)
    ↓
GenerateTarget (create heatmaps)
    ↓
Model (Backbone → Head)
    ↓
Loss Computation (MSE + optional phenotypic)
    ↓
Backpropagation & Optimization
    ↓
Checkpoint Saving
```

### Inference Pipeline
```
Input Image
    ↓
Preprocessing (resize, normalize)
    ↓
Model Forward Pass
    ↓
Heatmap Decoding (argmax or soft-argmax)
    ↓
Coordinate Prediction
    ↓
Visualization / Output
```

---

## Modifying the Code

### Adding a New Loss Function

1. **Create loss file** in `phenolosses/`:
```python
from mmpose.registry import MODELS
import torch.nn as nn

@MODELS.register_module()
class MyCustomLoss(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        # Initialize

    def forward(self, pred, target):
        # Compute loss
        return loss
```

2. **Update `phenolosses/__init__.py`**:
```python
from .my_custom_loss import MyCustomLoss
__all__ = [..., 'MyCustomLoss']
```

3. **Use in config**:
```python
custom_imports = dict(
    imports=['phenolosses.my_custom_loss']
)

model = dict(
    head=dict(
        loss=dict(type='MyCustomLoss', param1=..., param2=...)
    )
)
```

### Creating a New Experiment

1. **Copy base config**:
```bash
cp configs/experiment1_baseline_mse.py \
   configs/experiment_custom.py
```

2. **Modify parameters** in the new config

3. **Update `work_dir`** to avoid overwriting

4. **Run training**:
```bash
python tools/train.py configs/experiment_custom.py
```

---

## Common Paths

| Purpose | Path |
|---------|------|
| Training configs | `configs/` |
| Custom losses | `phenolosses/` |
| MMPose models | `mmpose/models/` |
| Training script | `tools/train.py` |
| Inference script | `demo/image_demo.py` |
| Checkpoints | `checkpoints/` (create manually) |
| Output/logs | `work_dirs/` (created automatically) |
| Documentation | Root directory (`*.md` files) |

---

## File Naming Conventions

### Configuration Files
Format: `experiment<N>_<variant>_<loss>.py`
- `td-hm`: Top-down heatmap
- `ViTPose-fish9`: Model architecture
- `8xb32`: 8 GPUs × 32 batch size (total 256)
- `100etrain`: 100 epochs training (actually 300 in practice)
- `<variant>`: Experimental variant (e.g., `phenoloss`, `base-log`)

### Module Files
- `*_loss.py`: Loss function implementations
- `*_head.py`: Head module implementations
- `inference_*.py`: Inference scripts
- `demo_*.py`: Demo scripts

---

## Helpful Commands

### Check config
```bash
python tools/misc/print_config.py configs/.../config.py
```

### Visualize dataset
```bash
python tools/misc/browse_dataset.py configs/.../config.py
```

### Profile model
```bash
python tools/analysis_tools/get_flops.py configs/.../config.py
```

### Convert checkpoint
```bash
python tools/misc/publish_model.py <checkpoint> <output>
```

---

## Questions?

- **Experimental protocol**: See [EXPERIMENTS.md](EXPERIMENTS.md)
- **Out-of-scope files**: See [OUT_OF_SCOPE.md](OUT_OF_SCOPE.md)
- **Installation & usage**: See [README.md](README.md)
- **Code structure**: This document
- **Issues**: [GitHub Issues](https://github.com/josuedelarosa/CaVFish-MORPH/issues)

---

*This guide is designed to help you navigate the repository efficiently. For specific implementation details, always refer to the source code and inline comments.*
