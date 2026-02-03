"""
ViTPose-Base Baseline Configuration for CaveFish 20-Keypoint Dataset

This is the baseline configuration using standard MSE loss for heatmap regression.
Used as the baseline comparison in the paper.

Architecture:
- Backbone: Vision Transformer Base (ViT-B/16) with MAE pretraining
- Head: Heatmap regression head with 2-layer deconvolution
- Decoder: UDP (Unbiased Data Processing) heatmap decoder

Training:
- 300 epochs with validation every 10 epochs
- AdamW optimizer with layer-wise learning rate decay
- Batch size: 32 per GPU (recommended 8 GPUs = 256 total)
"""

_base_ = ['../_base_/default_runtime.py']
default_scope = 'mmpose'

# ============================================================================
# Training Schedule
# ============================================================================
train_cfg = dict(max_epochs=300, val_interval=10)

# ============================================================================
# Custom Module Imports
# ============================================================================
custom_imports = dict(
    imports=[
        'mmpretrain.models',  # For VisionTransformer backbone
        'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper',
        'custom_transforms',  # Fish-specific data augmentation
    ],
    allow_failed_imports=False
)

# ============================================================================
# Optimizer & Learning Rate Schedule
# ============================================================================
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        betas=(0.9, 0.999),
        weight_decay=0.1
    ),
    paramwise_cfg=dict(
        num_layers=12,  # ViT-Base has 12 transformer blocks
        layer_decay_rate=0.75,  # Layer-wise LR decay for ViT
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    # Warmup for first 500 iterations
    dict(
        type='LinearLR',
        begin=0,
        end=500,
        start_factor=0.001,
        by_epoch=False
    ),
    # MultiStep LR decay
    dict(
        type='MultiStepLR',
        begin=0,
        end=300,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True
    )
]

# Automatically scale LR based on actual batch size
auto_scale_lr = dict(base_batch_size=256)

# ============================================================================
# Hooks
# ============================================================================
default_hooks = dict(
    checkpoint=dict(
        save_best='AP',
        rule='greater',
        max_keep_ckpts=1
    )
)

# ============================================================================
# Codec (Heatmap encoding/decoding)
# ============================================================================
codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),  # Width x Height
    heatmap_size=(48, 64),  # 1/4 of input size
    sigma=2  # Gaussian kernel sigma for heatmap generation
)

# ============================================================================
# Dataset Metainfo (20 keypoints for cave fish)
# ============================================================================
metainfo = dict(
    dataset_name='cavfish_20kpt',
    paper_info=dict(
        author='Poveda-Cuellar et al.',
        title='CaVFish-MORPH database: AI-driven morphometrics for mapping freshwater fish traits',
        year='2026'
    ),
    keypoint_info={
        0: dict(name='kp1', id=0, color=[255, 0, 0], type='upper', swap=''),
        1: dict(name='kp2', id=1, color=[255, 85, 0], type='upper', swap=''),
        2: dict(name='kp3', id=2, color=[255, 170, 0], type='upper', swap=''),
        3: dict(name='kp4', id=3, color=[255, 255, 0], type='lower', swap=''),
        4: dict(name='kp5', id=4, color=[170, 255, 0], type='upper', swap=''),
        5: dict(name='kp6', id=5, color=[85, 255, 0], type='upper', swap=''),
        6: dict(name='kp7', id=6, color=[0, 255, 0], type='lower', swap=''),
        7: dict(name='kp8', id=7, color=[0, 255, 85], type='upper', swap=''),
        8: dict(name='kp9', id=8, color=[0, 255, 170], type='lower', swap=''),
        9: dict(name='kp10', id=9, color=[0, 255, 255], type='upper', swap=''),
        10: dict(name='kp11', id=10, color=[0, 170, 255], type='lower', swap=''),
        11: dict(name='kp12', id=11, color=[0, 85, 255], type='upper', swap=''),
        12: dict(name='kp13', id=12, color=[0, 0, 255], type='upper', swap=''),
        13: dict(name='kp14', id=13, color=[85, 0, 255], type='upper', swap=''),
        14: dict(name='kp15', id=14, color=[170, 0, 255], type='lower', swap=''),
        15: dict(name='kp16', id=15, color=[255, 0, 255], type='lower', swap=''),
        16: dict(name='kp17', id=16, color=[255, 0, 170], type='lower', swap=''),
        17: dict(name='kp18', id=17, color=[255, 0, 85], type='upper', swap=''),
        18: dict(name='kp19', id=18, color=[128, 128, 128], type='upper', swap=''),
        19: dict(name='kp20', id=19, color=[64, 64, 64], type='lower', swap='')
    },
    skeleton_info={},  # No skeleton connections for fish
    joint_weights=[1.0] * 20,  # Equal weight for all keypoints
    sigmas=[0.025] * 20,  # OKS sigmas for evaluation
    flip_pairs=[],  # No bilateral symmetry in fish
    num_keypoints=20,
    upper_body_ids=[0, 1, 2, 4, 5, 7, 9, 11, 12, 13, 17, 18],
    lower_body_ids=[3, 6, 8, 10, 14, 15, 16, 19]
)

# ============================================================================
# Model Architecture
# ============================================================================
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # ImageNet mean
        std=[58.395, 57.12, 57.375],  # ImageNet std
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',  # ViT-Base: 12 layers, 768 hidden dim, 12 heads
        img_size=(256, 192),  # Input image size (H, W)
        patch_size=16,  # 16x16 patches
        qkv_bias=True,
        drop_path_rate=0.3,  # Stochastic depth for regularization
        with_cls_token=False,  # No classification token (dense prediction)
        out_type='featmap',  # Output feature maps for dense prediction
        patch_cfg=dict(padding=2),  # Padding for patch embedding
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mae_pretrain_vit_base.pth'
        ),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768,  # ViT-Base hidden dimension
        out_channels=20,  # Number of keypoints
        deconv_out_channels=(256, 256),  # Two deconv layers
        deconv_kernel_sizes=(4, 4),  # 4x4 kernels for upsampling
        loss=dict(
            type='KeypointMSELoss',
            use_target_weight=True  # Weight loss by keypoint visibility
        ),
        decoder=codec
    ),
    test_cfg=dict(
        flip_test=False,  # No test-time augmentation
        flip_mode='heatmap',
        shift_heatmap=False,
    )
)

# ============================================================================
# Dataset & Data Pipeline
# ============================================================================
data_root = 'data/cavfish'  # Change this to your dataset path
dataset_type = 'CocoDataset'
data_mode = 'topdown'

# Training data augmentation pipeline
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.15),  # Extract person bbox
    dict(type='RandomBBoxTransform'),  # Random bbox jittering
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='SafeRotateBackoff', max_angle=45, prob=0.6),  # Custom rotation
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),  # Generate heatmap targets
    dict(type='PackPoseInputs')
]

# Validation/test pipeline (no augmentation)
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.15),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

# ============================================================================
# Data Loaders
# ============================================================================
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/fish20kpt_train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
        metainfo=metainfo
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/fish20kpt_val.json',
        data_prefix=dict(img=''),
        pipeline=val_pipeline,
        metainfo=metainfo
    )
)

test_dataloader = val_dataloader

# ============================================================================
# Evaluation Metrics
# ============================================================================
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.05),  # PCK@0.05
    dict(type='AUC'),  # Area Under Curve
    dict(type='NME', norm_mode='use_norm_item')  # Normalized Mean Error
]

test_evaluator = val_evaluator

# Evaluation config
val_cfg = dict()
test_cfg = dict()

# ============================================================================
# Visualization
# ============================================================================
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    radius=4,
    line_width=2,
    alpha=0.8
)

# ============================================================================
# Runtime
# ============================================================================
work_dir = './work_dirs/vitpose_base_cavfish'
