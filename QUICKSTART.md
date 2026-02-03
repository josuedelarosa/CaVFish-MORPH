# Quick Start Guide

Get started with CaVFish pose estimation in 10 minutes.

## Prerequisites

- Linux/macOS with Python 3.8+
- 1+ NVIDIA GPU with ≥12GB VRAM
- CUDA 11.3+

## Installation (5 minutes)

```bash
# 1. Create environment
conda create -n cavfish python=3.8 -y
conda activate cavfish

# 2. Install PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Clone this repository
git clone https://github.com/josuedelarosa/CaVFish-MORPH.git
cd CaVFish-MORPH

# 4. Install dependencies (MMEngine, MMCV, MMPretrain)
pip install -U openmim
mim install mmengine mmcv mmpretrain

# 5. Install this repository (includes customized MMPose)
pip install -v -e .
pip install albumentations opencv-python-headless scipy pandas

# 6. Download pretrained weights
mkdir checkpoints
wget -P checkpoints/ \
    https://download.openmmlab.com/mmpose/v1/projects/pretrained/mae_pretrain_vit_base.pth
```

## Dataset Setup (2 minutes)

Organize your dataset in COCO format:

```bash
mkdir -p data/cavfish/annotations
# Place your COCO JSON files in data/cavfish/annotations/
# Place images in data/cavfish/images/
```

Your annotations should follow COCO format with 20 keypoints per fish.

## Training (1 command)

```bash
# Single GPU (for testing)
python tools/train.py configs/cavfish/vitpose_base_cavfish.py

# Multi-GPU (recommended)
bash tools/dist_train.sh configs/cavfish/vitpose_base_cavfish.py 4

# With phenotype loss
bash tools/dist_train.sh configs/cavfish/vitpose_base_cavfish_phenoloss.py 4
```

Training takes ~18 hours on 8× A100 GPUs (300 epochs).

## Inference (1 command)

```bash
# Single image
python tools/inference.py \
    --config configs/cavfish/vitpose_base_cavfish.py \
    --checkpoint work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth \
    --img path/to/fish.jpg \
    --out-file output/result.jpg \
    --show-kpt-idx

# Batch processing
python tools/inference.py \
    --config configs/cavfish/vitpose_base_cavfish.py \
    --checkpoint work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth \
    --img-dir path/to/images/ \
    --out-dir output/ \
    --save-predictions predictions.json
```

## Evaluation

```bash
python tools/test.py \
    configs/cavfish/vitpose_base_cavfish.py \
    work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth
```

## Next Steps

- 📖 Read [README.md](README.md) for complete documentation
- 🔬 Follow [docs/REPRODUCTION.md](docs/REPRODUCTION.md) to reproduce paper results
- 🎨 Check [docs/CUSTOM_TRANSFORMS.md](docs/CUSTOM_TRANSFORMS.md) for augmentation details
- 🧮 Read [phenolosses/README.md](phenolosses/README.md) for loss function details

## Troubleshooting

**CUDA out of memory?**
```python
# Edit config: reduce batch size
train_dataloader = dict(batch_size=16)  # instead of 32
```

**Import errors?**
```bash
# Verify installations
python -c "import mmpose; print(mmpose.__version__)"
python -c "import mmpretrain; print(mmpretrain.__version__)"
```

**Low accuracy?**
- Verify dataset annotations (COCO format with 20 keypoints)
- Check that bbox padding is set correctly (default: 1.15)
- Ensure input size matches training (192×256)

## Getting Help

- **Issues**: https://github.com/josuedelarosa/CaVFish-MORPH/issues
- **Discussions**: https://github.com/josuedelarosa/CaVFish-MORPH/discussions
- **Email**: josue.delarosa@university.edu

## Citation

```bibtex
@article{povedacuellar2026cavfish,
  title={CaVFish-MORPH database: AI-driven morphometrics for mapping freshwater fish traits},
  author={Poveda-Cuellar, Jose Luis and Rodriguez-de la Rosa, Josué and Martínez-Carrillo, Fabio and García-Melo, Jorge Enrique and García-Melo, Luis José and Marchant, Sergio and Reu, Björn},
  journal={...},
  year={2026}
}
```

---

**Ready to go!** 🚀

For complete documentation, see [README.md](README.md).
