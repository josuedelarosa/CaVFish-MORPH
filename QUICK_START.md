# Quick Reference: Running Experiments

This is a quick reference guide for running the four core experiments reported in the paper.

For complete details, see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## 📋 Prerequisites

1. **Environment setup complete** (see [README.md](README.md#installation))
2. **Dataset prepared** at `/data/Datasets/Fish/CavFish/`
3. **Pretrained backbone downloaded** to `checkpoints/mae_pretrain_vit_base_20230913.pth`

---

## 🚀 Experiments

### Experiment 1: Baseline (MSE)

**Purpose**: Standard heatmap-based pose estimation

**Command**:
```bash
python tools/train.py \
    configs/experiment1_baseline_mse.py
```

**Key Settings**:
- Loss: `KeypointMSELoss`
- Head: `HeatmapHead`
- Phenotypic loss: None
- Output: `/data/Pupils/Josue/weights/Fish/ViTPose_base_2nd-run`

---

### Experiment 2: Baseline + Logging (LogMSE)

**Purpose**: Evaluate logarithmic error weighting

**Command**:
```bash
python tools/train.py \
    configs/experiment2_baseline_logmse.py
```

**Key Settings**:
- Loss: `KeypointLogMSELoss`
- Head: `PhenoLossHead`
- Phenotypic loss: Disabled (`alpha_pheno=0`)
- Output: `/data/Pupils/Josue/weights/Fish/ViTPose_20kpt_base-log`

---

### Experiment 3: PhenoLoss (MSE + Phenotypic)

**Purpose**: Our proposed method with anatomical constraints

**Command**:
```bash
python tools/train.py \
    configs/experiment3_phenoloss_mse.py
```

**Key Settings**:
- Primary loss: `KeypointMSELoss`
- Secondary loss: `PhenotypeDistanceLoss`
- Head: `PhenoLossHead`
- Phenotypic weight: `alpha_pheno=0.01`
- Output: `/data/Pupils/Josue/weights/Fish/ViTPose_20kpt_phenoloss_2nd-run`

**Anatomical pairs** (13 pairs):
```python
[(0,1), (2,3), (4,5), (6,7), (8,9), (10,11),
 (12,3), (0,13), (14,15), (14,3),
 (0,16), (2,17), (18,19)]
```

**Measurements computed**:
- BI, Bd, Hd, CPd, CFd, Ed, Eh, JI, PFI, PFi, HL, DL, AL
- All normalized by Body Index (BI) for size-independent analysis

---

### Experiment 4: PhenoLoss + Logging (LogMSE + Phenotypic)

**Purpose**: Combined log-MSE and phenotypic constraints

**Command**:
```bash
python tools/train.py \
    configs/experiment4_phenoloss_logmse.py
```

**Key Settings**:
- Primary loss: `KeypointLogMSELoss`
- Secondary loss: `PhenotypeDistanceLoss`
- Head: `PhenoLossHead`
- Phenotypic weight: `alpha_pheno=0.01`
- Output: `/data/Pupils/Josue/weights/Fish/ViTPose_20kpt_base-log_2nd-run_pred`

---

## 🧪 Evaluation

**Test all experiments**:
```bash
# Experiment 1
python tools/test.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-fish9_8xb32-100etest.py \
    /data/Pupils/Josue/weights/Fish/ViTPose_base_2nd-run/best_AP_epoch_XXX.pth

# Experiment 2
python tools/test.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-fish9_8xb32-100etest.py \
    /data/Pupils/Josue/weights/Fish/ViTPose_20kpt_base-log/best_AP_epoch_XXX.pth

# Experiment 3
python tools/test.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-fish9_8xb32-100etest.py \
    /data/Pupils/Josue/weights/Fish/ViTPose_20kpt_phenoloss_2nd-run/best_AP_epoch_XXX.pth

# Experiment 4
python tools/test.py \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-fish9_8xb32-100etest.py \
    /data/Pupils/Josue/weights/Fish/ViTPose_20kpt_base-log_2nd-run_pred/best_AP_epoch_XXX.pth
```

**Metrics**:
- PCK@0.05 (Percentage of Correct Keypoints at 5% threshold)
- AUC (Area Under Curve)

---

## 📊 Comparison Table

| Exp | Config Suffix | Primary Loss | Pheno Loss | α | val_interval |
|-----|---------------|--------------|------------|---|--------------|
| 1 | `100etrain.py` | MSE | ✗ | - | 10 |
| 2 | `base-log.py` | LogMSE | ✗ | 0 | 300 |
| 3 | `phenoloss.py` | MSE | ✓ | 0.01 | 300 |
| 4 | `phenoloss-log.py` | LogMSE | ✓ | 0.01 | 300 |

---

## ⚙️ Common Options

### Resume Training
```bash
python tools/train.py <config> --resume <checkpoint>
```

### Custom Output Directory
```bash
python tools/train.py <config> --work-dir custom_output_dir/
```

### Override Config Parameters
```bash
python tools/train.py <config> \
    --cfg-options train_cfg.max_epochs=200
```

### Multi-GPU Training
```bash
bash tools/dist_train.sh <config> <num_gpus>
```

### Check GPU Availability
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
```

---

## 📝 Training Logs

Logs are saved in each experiment's `work_dir`:
```
work_dir/
├── <timestamp>.log                    # Text log
├── <timestamp>.log.json               # JSON log (for plotting)
├── best_AP_epoch_XXX.pth             # Best checkpoint by AP
├── latest.pth                         # Latest checkpoint
└── vis_data/                          # Visualization outputs
```

### Monitor Training
```bash
# View log
tail -f work_dir/<timestamp>.log

# Plot metrics (requires parsing JSON log)
python tools/analysis_tools/analyze_logs.py plot_curve \
    work_dir/<timestamp>.log.json \
    --keys loss_kpt loss_pheno \
    --out work_dir/loss_curve.png
```

---

## 🎯 Inference Examples

### Single Image
```bash
python demo/image_demo.py \
    test_image.jpg \
    configs/your_config.py \
    work_dir/checkpoint.pth \
    --out-file result.jpg
```

### Batch Processing
```bash
python run_inference_all.py \
    --config configs/your_config.py \
    --checkpoint work_dir/checkpoint.pth \
    --img-dir test_images/ \
    --out-dir results/
```

### Interactive Notebook
Open [`notebook_inference.ipynb`](notebook_inference.ipynb) in Jupyter Lab:
```bash
jupyter lab notebook_inference.ipynb
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config
--cfg-options train_dataloader.batch_size=16

# Use gradient checkpointing (if supported)
--cfg-options model.backbone.use_checkpoint=True
```

### ImportError for Custom Modules
Make sure `custom_imports` in config includes:
```python
custom_imports = dict(
    imports=[
        'phenolosses.phenoloss_distance_loss',
        'phenolosses.vitpose_head_phenoloss',
        # ...
    ]
)
```

### Dataset Path Issues
Update `data_root` in config or use:
```bash
--cfg-options train_dataloader.dataset.data_root='/your/path'
```

### Checkpoint Not Found
Ensure pretrained backbone is at:
```
checkpoints/mae_pretrain_vit_base_20230913.pth
```

---

## 📚 More Information

- **Detailed protocol**: [EXPERIMENTS.md](EXPERIMENTS.md)
- **Installation guide**: [README.md](README.md)
- **Repository structure**: [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md)
- **Out-of-scope files**: [OUT_OF_SCOPE.md](OUT_OF_SCOPE.md)

---

## 📞 Support

For issues with reproducibility:
1. Check [EXPERIMENTS.md](EXPERIMENTS.md) for detailed settings
2. Verify dataset format and paths
3. Confirm environment installation
4. Open an issue with error logs and system info

---

*Last updated: February 2026*
