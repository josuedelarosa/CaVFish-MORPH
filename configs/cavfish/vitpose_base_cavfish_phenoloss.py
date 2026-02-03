"""
ViTPose-Base + Phenotype Loss for CaveFish 20-Keypoint Dataset

This configuration adds phenotype-aware distance loss to the baseline ViTPose model.
This is the main contribution of the paper, enforcing anatomical constraints between
keypoints during training.

Key Differences from Baseline:
- Uses HeatmapHeadPheno (custom head with phenotype loss)
- Adds PhenotypeDistanceLoss with weight α=0.01
- Otherwise identical architecture and training setup

The phenotype loss penalizes predictions that violate known anatomical distance
relationships between keypoints, improving morphometric accuracy.
"""

_base_ = ['./vitpose_base_cavfish.py']

# ============================================================================
# Custom Module Imports (Extended)
# ============================================================================
custom_imports = dict(
    imports=[
        'mmpretrain.models',
        'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper',
        'custom_transforms',
        'phenolosses.phenotype_distance_loss',  # Custom phenotype loss
        'phenolosses.vitpose_head_pheno',  # Custom head with phenotype loss
    ],
    allow_failed_imports=False
)

# ============================================================================
# Model Architecture (Modified Head)
# ============================================================================
model = dict(
    head=dict(
        type='HeatmapHeadPheno',  # Custom head instead of standard HeatmapHead
        in_channels=768,
        out_channels=20,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(
            type='KeypointMSELoss',
            use_target_weight=True
        ),
        # Additional phenotype loss configuration
        loss_pheno=dict(
            type='PhenotypeDistanceLoss',
            distance_pairs=[
                # Define anatomical distance constraints
                # Format: (kp_i, kp_j, expected_distance, tolerance)
                # These pairs enforce morphometric relationships
                (0, 5, 'standard_length', 0.1),
                (5, 12, 'head_length', 0.1),
                (12, 19, 'caudal_length', 0.1),
                # Add more pairs based on fish morphometry
            ]
        ),
        alpha_pheno=0.01,  # Weight for phenotype loss term
        decoder=dict(
            type='UDPHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2
        )
    )
)

# ============================================================================
# Runtime
# ============================================================================
work_dir = './work_dirs/vitpose_base_cavfish_phenoloss'
