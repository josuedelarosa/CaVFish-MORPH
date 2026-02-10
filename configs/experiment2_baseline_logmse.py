# ===================================================================
# EXPERIMENT 2: BASELINE + LOGGING (LogMSE Loss)
# ===================================================================
# 
# Paper Section: Experimental Protocol - Baseline with Logarithmic Transform
# Purpose: Baseline with log-transformed MSE loss for relative error emphasis
# Loss Function: KeypointLogMSELoss (logarithmic MSE, no phenotypic constraints)
# Head Type: PhenoLossHead (but with phenotypic loss disabled)
# Validation: Every 300 epochs (less frequent than baseline)
# 
# This configuration evaluates the effect of logarithmic error weighting
# without anatomical constraints.
# ===================================================================

default_scope = 'mmpose'
_base_ = ['./_base_/default_runtime.py']
work_dir = './work_dirs/experiment2_baseline_logmse'  # Training output directory


# Configuration variables (adjust these paths as needed for your setup)
data_root = '/Users/josuedelarosa/mnt/BIVL2ab/home/Data/Datasets/Fish/CavFish/'  # Relative path - adjust as needed
checkpoint_dir = '../checkpoints'  # Relative path to project root checkpoints

# Training Configuration
train_cfg = dict(max_epochs=300, val_interval=300)

# Custom imports for MMPose extensions and phenotypic losses
custom_imports = dict(
    imports=[
        'mmpretrain.models',
        'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper',
        'custom_transforms',
        'phenolosses.phenoloss_distance_loss',
        'phenolosses.vitpose_head_phenoloss',
        'mmpose.models.losses.heatmap_loss',
        'phenolosses.keypoint_log_mse_loss',
    ],
    allow_failed_imports=False
)

# Optimizer Configuration
# AdamW with layer-wise learning rate decay for Vision Transformer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,  # ViT-Base has 12 layers
        layer_decay_rate=0.75,  # Reduce learning rate deeper in network
        custom_keys={
            'bias': dict(decay_multi=0.0),  # No decay for bias terms
            'pos_embed': dict(decay_mult=0.0),  # No decay for position embeddings
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),  # No decay for normalization layers
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# Learning Rate Scheduler
# Warmup followed by multi-step decay
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # Warmup for 500 iterations
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],  # Reduce LR at these epochs
        gamma=0.1,
        by_epoch=True),
]

# Automatic Mixed Precision
auto_scale_lr = dict(base_batch_size=512)

# Dataset Configuration
# COCO format for fish keypoint detection (20 landmarks)
dataset_info = dict(
    dataset_name='CavFish',
    paper_info=dict(
        author='Jimenez-Brenes et al.',
        title='Cave Fish Keypoint Detection',
        container='Scientific Data',
        year='2024',
        homepage='',
    ),
    keypoint_info={
        0: dict(name='snout', id=0, color=[255, 0, 0], type='', swap=''),
        1: dict(name='eye', id=1, color=[0, 255, 0], type='', swap=''),
        2: dict(name='gill_start', id=2, color=[0, 0, 255], type='', swap=''),
        3: dict(name='gill_end', id=3, color=[255, 255, 0], type='', swap=''),
        4: dict(name='middle_line', id=4, color=[255, 0, 255], type='', swap=''),
        5: dict(name='back_insertion_dors_fin', id=5, color=[0, 255, 255], type='', swap=''),
        6: dict(name='front_insertion_dors_fin', id=6, color=[128, 0, 128], type='', swap=''),
        7: dict(name='end_dors_fin', id=7, color=[255, 128, 0], type='', swap=''),
        8: dict(name='adipose_fin', id=8, color=[128, 255, 128], type='', swap=''),
        9: dict(name='back_insertion_anal_fin', id=9, color=[128, 128, 255], type='', swap=''),
        10: dict(name='front_insertion_anal_fin', id=10, color=[255, 255, 128], type='', swap=''),
        11: dict(name='end_anal_fin', id=11, color=[255, 128, 255], type='', swap=''),
        12: dict(name='insertion_pelvic_fin', id=12, color=[128, 255, 255], type='', swap=''),
        13: dict(name='insertion_pectoral_fin', id=13, color=[64, 128, 255], type='', swap=''),
        14: dict(name='operculum', id=14, color=[255, 64, 128], type='', swap=''),
        15: dict(name='end_lateral_line', id=15, color=[128, 255, 64], type='', swap=''),
        16: dict(name='caudal_fin_top', id=16, color=[255, 128, 64], type='', swap=''),
        17: dict(name='caudal_fin_middle', id=17, color=[64, 255, 128], type='', swap=''),
        18: dict(name='caudal_fin_bottom', id=18, color=[128, 64, 255], type='', swap=''),
        19: dict(name='end_vertebral_column', id=19, color=[255, 64, 64], type='', swap=''),
    },
    skeleton_info={},  # No skeleton connections for visualization
    joint_weights=[1.0] * 20,  # Equal weights for all 20 keypoints
    sigmas=[0.025] * 20  # Sigma for OKS metric
)

# CODEC Configuration for heatmap-based pose estimation
codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# Automatic Mixed Precision
auto_scale_lr = dict(base_batch_size=512)

# Checkpoint Configuration
default_hooks = dict(
    checkpoint=dict(save_best='AP', rule='greater', max_keep_ckpts=1))

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(256, 256),  # Must match codec input_size
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=f'{checkpoint_dir}/mae_pretrain_vit_base_20230913.pth'),
    ),
    head=dict(
        type='PhenoLossHead',   # Extended head for loss composition
        in_channels=768,           # ViT-Base output channels
        out_channels=20,           # 20 keypoints
        loss=dict(type='KeypointLogMSELoss', use_target_weight=True),
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        decoder=codec,
        loss_pheno=dict(
            type='PhenotypeDistanceLoss',
            # Anatomically meaningful keypoint pairs (0-indexed)
            pairs=[(0,1), (3,2), (4,5), (6,7), (8,9), (10,11),
                   (12,3), (0,3), (0,13), (14,15), (14,3),
                   (0,16), (2,17), (18,19)],
            degree_normalize=True,
            scale_by_SL=False,
            normalization="min_gt",
            percentile=None,
            detach_scale=True,  # Avoid "gaming" the scale
            clamp_min=1e-3,     # Numerical safety
            clamp_max=None,
            beta=10.0   # Soft-argmax temperature
        ),
        alpha_pheno=0  # Phenotypic loss weight (0 = disabled for baseline)
    ),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# Dataset Configuration
dataset_type = 'CocoDataset'
data_mode = 'topdown'

# Training Pipeline
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.15),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# Training DataLoader
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='fish20kpt_all_train_2nd-run.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
        metainfo=dataset_info
    ))

# Validation Configuration (disabled for this experiment)
val_dataloader = None
val_evaluator = None
val_cfg = None

# Test Configuration
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='fish20kpt_all_val_2nd-run.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,  # Reuse training pipeline for testing
        metainfo=dataset_info
    )
)

test_evaluator = [
    dict(type='PCKAccuracy', thr=0.05),
    dict(type='AUC')
]
test_cfg = dict()

# Visualization Configuration
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    radius=3,
    line_width=2,
    alpha=0.8
)