# Phenotype-Aware Loss Functions for Fish Pose Estimation

This module contains custom loss functions and model heads that incorporate anatomical constraints into pose estimation training. These losses enforce known morphometric relationships between keypoints, improving the biological accuracy of predictions.

## Overview

Traditional pose estimation losses (MSE, smooth L1) only penalize pixel-wise errors without considering the anatomical relationships between keypoints. For morphometric applications like fish phenotyping, maintaining accurate relative distances between keypoints is crucial.

Our phenotype-aware losses add a regularization term that penalizes predictions violating known anatomical distance relationships.

## Components

### 1. PhenotypeDistanceLoss
**File**: `phenotype_distance_loss.py`

**Description**: Penalizes deviations from expected pairwise distances between keypoints.

**Key Features**:
- Computes soft-argmax coordinates from predicted heatmaps
- Calculates pairwise Euclidean distances between keypoints
- Compares predicted distances to ground truth distances
- Optional normalization by standard length (SL)
- Optional degree-based weighting for graph structure

**Parameters**:
- `pairs`: List of (i, j) tuples defining which keypoint pairs to constrain
- `degree_normalize` (bool): Weight edges by node degree (default: True)
- `scale_by_SL` (bool): Normalize all distances by standard length (default: True)
- `beta` (float): Temperature parameter for soft-argmax (default: 15.0)

**Example Usage**:
```python
loss_pheno = dict(
    type='PhenotypeDistanceLoss',
    pairs=[
        (0, 1),   # Standard length
        (1, 5),   # Head to dorsal fin
        (5, 12),  # Dorsal to caudal
        # Add more morphometric pairs
    ],
    degree_normalize=True,
    scale_by_SL=True,
    beta=15.0
)
```

**Mathematical Formulation**:
```
L_pheno = Σ w_ij * (||p_i - p_j|| - ||gt_i - gt_j||)²

where:
- p_i, p_j: predicted keypoint coordinates (from soft-argmax)
- gt_i, gt_j: ground truth keypoint coordinates  
- w_ij: edge weight (1/degree if normalized, else 1)
- ||·||: L2 distance
```

---

### 2. MinPhenotypeDistanceLoss
**File**: `minpheno_distance_loss.py`

**Description**: Similar to PhenotypeDistanceLoss but with more flexible normalization options and numerical stability improvements.

**Key Features**:
- Multiple normalization modes: by SL, by minimum GT distance, by minimum predicted distance, or none
- Percentile-based min computation (avoids outlier sensitivity)
- Gradient detachment options for the normalization scale
- Clamping bounds for numerical stability
- Efficient coordinate grid caching

**Parameters**:
- `pairs`: List of (i, j) tuples for distance constraints
- `degree_normalize` (bool): Apply degree-based weighting
- `normalization` (str): Normalization mode - "sl", "min_gt", "min_pred", or "none" (default: "min_pred")
- `min_pairs`: Optional list of pairs to use for min calculation
- `percentile` (float): Use nth percentile instead of hard min (e.g., 10.0)
- `detach_scale` (bool): Detach gradients through scale (default: True)
- `clamp_min` (float): Minimum scale value (default: 1e-6)
- `clamp_max` (float): Maximum scale value (default: None)
- `beta` (float): Soft-argmax temperature (default: 10.0)

**Normalization Modes**:
- **"sl"**: Divide by standard length (pair 0-1) - classic fish morphometry
- **"min_gt"**: Divide by smallest GT distance in min_pairs - adaptive scaling
- **"min_pred"**: Divide by smallest predicted distance - self-normalizing
- **"none"**: No normalization - absolute distances

**Example Usage**:
```python
loss_pheno = dict(
    type='MinPhenotypeDistanceLoss',
    pairs=[(0,1), (1,5), (5,12), (12,19)],
    min_pairs=[(0,1), (1,5)],  # Use these for min calculation
    normalization='min_pred',
    percentile=10.0,           # Use 10th percentile instead of min
    detach_scale=True,
    degree_normalize=True,
    beta=10.0
)
```

---

### 3. HeatmapHeadPheno
**File**: `vitpose_head_pheno.py`

**Description**: Custom heatmap head that combines standard heatmap loss with phenotype distance loss.

**Key Features**:
- Inherits from MMPose's HeatmapHead
- Adds phenotype loss term weighted by α
- Reuses predicted heatmaps (no extra forward pass)
- Compatible with all MMPose heatmap decoders

**Parameters**:
- `loss_pheno` (dict): Configuration for phenotype loss (PhenotypeDistanceLoss or MinPhenotypeDistanceLoss)
- `alpha_pheno` (float): Weight for phenotype loss term (default: 0.01)
- All standard HeatmapHead parameters (in_channels, out_channels, etc.)

**Example Usage**:
```python
head = dict(
    type='HeatmapHeadPheno',
    in_channels=768,
    out_channels=20,
    loss=dict(type='KeypointMSELoss', use_target_weight=True),
    loss_pheno=dict(
        type='PhenotypeDistanceLoss',
        pairs=[(0,1), (1,5), (5,12)],
        degree_normalize=True,
        scale_by_SL=True
    ),
    alpha_pheno=0.01  # Phenotype loss weight
)
```

**Total Loss**:
```
L_total = L_heatmap + α * L_pheno

where:
- L_heatmap: standard MSE loss on heatmaps
- L_pheno: phenotype distance loss
- α: weighting factor (typically 0.001 - 0.1)
```

---

### 4. Supporting Modules

**vitpose_head_norm.py**: Variant with normalized phenotype loss
**vitpose_head_minpheno.py**: Variant using MinPhenotypeDistanceLoss
**coord_smooth_l1_loss.py**: Smooth L1 loss for coordinate regression (unused in paper)
**keypoint_log_mse_loss.py**: Log-space MSE loss variant (experimental)

---

## Integration with MMPose

All components are registered with MMPose's registry system and can be used in configuration files:

```python
# In config file
custom_imports = dict(
    imports=[
        'phenolosses.phenotype_distance_loss',
        'phenolosses.vitpose_head_pheno',
    ],
    allow_failed_imports=False
)

model = dict(
    type='TopdownPoseEstimator',
    head=dict(
        type='HeatmapHeadPheno',
        # ... configuration ...
    )
)
```

---

## Implementation Details

### Soft-Argmax Coordinate Extraction

Instead of using the hard maximum of the heatmap, we use differentiable soft-argmax to extract smooth coordinates:

```python
def softargmax2d(heatmaps, beta=10.0):
    # heatmaps: (N, K, H, W)
    logits = heatmaps.view(N, K, -1)
    logits = logits - logits.max(dim=-1, keepdim=True).values  # Numerical stability
    P = F.softmax(beta * logits, dim=-1)  # Soft distribution
    x = (P * x_grid).sum(-1)  # Expected x coordinate
    y = (P * y_grid).sum(-1)  # Expected y coordinate
    return torch.stack([x, y], dim=-1)
```

The `beta` parameter controls the "sharpness":
- Higher beta → closer to hard argmax (more deterministic)
- Lower beta → softer distribution (more robust to noise)

### Visibility Masking

The loss automatically handles invisible keypoints by extracting visibility masks from ground truth annotations:

```python
# Only compute loss for visible keypoints
vis_mask = get_visibility_mask(data_samples)
loss = loss * vis_mask
```

### Coordinate Space Conversion

Ground truth coordinates are provided in input image space (e.g., 192×256) while predictions are in heatmap space (e.g., 48×64). The loss automatically converts GT coordinates to heatmap space:

```python
scale_x = heatmap_width / input_width
scale_y = heatmap_height / input_height
gt_heatmap = gt_input * [scale_x, scale_y]
```

---

## Hyperparameter Tuning Guidelines

### Alpha (α) - Phenotype Loss Weight
- **Start**: 0.01 (1% of total loss)
- **Range**: 0.001 to 0.1
- **Effect**: Higher α enforces stronger anatomical constraints but may reduce keypoint accuracy
- **Tuning**: Monitor validation PCK and morphometric error separately

### Beta (β) - Soft-Argmax Temperature  
- **Default**: 10.0 - 15.0
- **Range**: 5.0 to 50.0
- **Effect**: Higher β makes soft-argmax more deterministic
- **Tuning**: Higher β may improve morphometric accuracy but reduce gradient smoothness

### Normalization Mode
- **"sl"**: Best for standard fish morphometry (when standard length is well-defined)
- **"min_pred"**: Best for diverse fish sizes and orientations
- **"min_gt"**: Best when GT annotations are highly accurate
- **"none"**: Best when absolute distances matter

---

## Citation

If you use these phenotype-aware losses in your research, please cite:

```bibtex
@article{povedacuellar2026cavfish,
  title={CaVFish-MORPH database: AI-driven morphometrics for mapping freshwater fish traits},
  author={Poveda-Cuellar, Jose Luis and Rodriguez-de la Rosa, Josué and Martínez-Carrillo, Fabio and García-Melo, Jorge Enrique and García-Melo, Luis José and Marchant, Sergio and Reu, Björn},
  journal={...},
  year={2026}
}
```

---

## Future Extensions

Possible improvements for future work:
- **Angle constraints**: Enforce angle relationships (e.g., spine curvature)
- **Ratio constraints**: Enforce morphometric ratios (e.g., head/body ratio)
- **Hierarchical constraints**: Multi-scale distance relationships
- **Learned constraints**: Learn constraint weights from data
- **3D extensions**: Apply to 3D pose estimation with volumetric constraints
