"""
CaveFish Pose Estimation Inference Script

This script performs keypoint detection inference on cave fish images using trained
ViTPose models. Supports both single image and batch processing.

Usage:
    # Single image
    python tools/inference.py --config configs/cavfish/vitpose_base_cavfish.py \
        --checkpoint work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth \
        --img path/to/image.jpg \
        --out-file output/result.jpg

    # Batch processing
    python tools/inference.py --config configs/cavfish/vitpose_base_cavfish.py \
        --checkpoint work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth \
        --img-dir path/to/images/ \
        --out-dir output/visualizations/ \
        --save-predictions output/predictions.json

Features:
    - Visualizes keypoints with heatmaps
    - Saves prediction coordinates to JSON
    - Supports custom detection thresholds
    - Batch processing for datasets
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import PoseDataSample


def parse_args():
    parser = argparse.ArgumentParser(
        description='CaveFish pose estimation inference')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--img',
        type=str,
        help='Path to single input image'
    )
    group.add_argument(
        '--img-dir',
        type=str,
        help='Path to directory containing images'
    )
    
    # Output options
    parser.add_argument(
        '--out-file',
        type=str,
        help='Path to save output visualization (for single image)'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        help='Directory to save output visualizations (for batch)'
    )
    parser.add_argument(
        '--save-predictions',
        type=str,
        help='Path to save keypoint predictions as JSON'
    )
    
    # Visualization options
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize prediction heatmaps'
    )
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        help='Show keypoint indices on visualization'
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint circle radius'
    )
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Line thickness'
    )
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Keypoint confidence threshold'
    )
    
    # Runtime options
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run inference (cuda:0, cpu, etc.)'
    )
    parser.add_argument(
        '--bbox-padding',
        type=float,
        default=1.15,
        help='Bounding box padding factor'
    )
    
    args = parser.parse_args()
    return args


def get_image_files(img_dir: str) -> List[str]:
    """Get all image files from directory."""
    img_dir = Path(img_dir)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    image_files = []
    for ext in extensions:
        image_files.extend(img_dir.glob(ext))
        image_files.extend(img_dir.glob(ext.upper()))
    
    return sorted([str(f) for f in image_files])


def create_full_image_bbox(img_path: str, padding: float = 1.15) -> Dict:
    """Create bounding box covering the entire image."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    h, w = img.shape[:2]
    
    # Create bbox covering entire image
    bbox = np.array([0, 0, w, h], dtype=np.float32)
    
    return {
        'bbox': bbox,
        'bbox_score': 1.0
    }


def extract_predictions(result: PoseDataSample) -> Dict:
    """Extract keypoint predictions from result."""
    pred_instances = result.pred_instances
    
    keypoints = pred_instances.keypoints[0]  # Shape: (20, 2)
    scores = pred_instances.keypoint_scores[0]  # Shape: (20,)
    
    predictions = {
        'keypoints': keypoints.tolist(),
        'scores': scores.tolist(),
        'bbox': pred_instances.bboxes[0].tolist() if hasattr(pred_instances, 'bboxes') else None
    }
    
    return predictions


def visualize_result(
    img: np.ndarray,
    result: PoseDataSample,
    show_kpt_idx: bool = False,
    radius: int = 4,
    thickness: int = 2,
    kpt_thr: float = 0.3
) -> np.ndarray:
    """Visualize keypoint detection results."""
    vis_img = img.copy()
    
    pred_instances = result.pred_instances
    keypoints = pred_instances.keypoints[0]
    scores = pred_instances.keypoint_scores[0]
    
    # Get colors from metainfo
    cfg = Config.fromfile(args.config)
    metainfo = cfg.metainfo
    
    # Draw keypoints
    for idx, (kpt, score) in enumerate(zip(keypoints, scores)):
        if score < kpt_thr:
            continue
        
        x, y = int(kpt[0]), int(kpt[1])
        
        # Get color for this keypoint
        color = metainfo['keypoint_info'][idx]['color']
        color = tuple(color[::-1])  # RGB to BGR
        
        # Draw circle
        cv2.circle(vis_img, (x, y), radius, color, -1)
        cv2.circle(vis_img, (x, y), radius + 1, (255, 255, 255), thickness)
        
        # Draw keypoint index if requested
        if show_kpt_idx:
            cv2.putText(
                vis_img,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return vis_img


def main():
    args = parse_args()
    
    # Initialize model
    print(f"Loading model from {args.config}")
    model = init_model(args.config, args.checkpoint, device=args.device)
    print(f"Model loaded successfully")
    
    # Get input images
    if args.img:
        image_files = [args.img]
    else:
        print(f"Scanning directory: {args.img_dir}")
        image_files = get_image_files(args.img_dir)
        print(f"Found {len(image_files)} images")
    
    if not image_files:
        print("No images found!")
        return
    
    # Prepare output directories
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Process each image
    all_predictions = {}
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path}")
        
        # Create full-image bounding box
        bbox_dict = create_full_image_bbox(img_path, args.bbox_padding)
        
        # Run inference
        results = inference_topdown(model, img_path, bboxes=[bbox_dict['bbox']])
        
        if not results:
            print(f"  Warning: No detections for {img_path}")
            continue
        
        result = results[0]
        
        # Extract predictions
        predictions = extract_predictions(result)
        all_predictions[img_path] = predictions
        
        # Visualize if requested
        if args.out_file or args.out_dir:
            img = cv2.imread(img_path)
            vis_img = visualize_result(
                img,
                result,
                show_kpt_idx=args.show_kpt_idx,
                radius=args.radius,
                thickness=args.thickness,
                kpt_thr=args.kpt_thr
            )
            
            # Determine output path
            if args.out_file:
                out_path = args.out_file
            else:
                img_name = Path(img_path).name
                out_path = os.path.join(args.out_dir, img_name)
            
            cv2.imwrite(out_path, vis_img)
            print(f"  Saved visualization to: {out_path}")
        
        # Print prediction summary
        avg_score = np.mean(predictions['scores'])
        print(f"  Average keypoint confidence: {avg_score:.3f}")
    
    # Save predictions to JSON
    if args.save_predictions:
        with open(args.save_predictions, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\nSaved predictions to: {args.save_predictions}")
    
    print(f"\nProcessed {len(all_predictions)} images successfully")


if __name__ == '__main__':
    main()
