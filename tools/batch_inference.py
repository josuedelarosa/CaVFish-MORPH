"""
Batch Inference Tool for CaveFish Datasets

This script processes multiple dataset folders in batch, running pose estimation
inference on all images while preserving directory structure. It can reuse
previously generated predictions to avoid re-processing.

Key Features:
- Batch processing of multiple dataset folders
- Preserves directory structure in output
- Reuses existing predictions (skip already processed images)
- JSON output for each image with keypoint coordinates
- Progress tracking and error handling

Usage:
    python tools/batch_inference.py \\
        --config configs/cavfish/vitpose_base_cavfish.py \\
        --checkpoint work_dirs/vitpose_base_cavfish/best_AP_epoch_*.pth \\
        --data-root /path/to/datasets \\
        --output-root /path/to/output \\
        --datasets "Population_A" "Population_B" "Population_C"

Example:
    python tools/batch_inference.py \\
        --config configs/cavfish/vitpose_base_cavfish_phenoloss.py \\
        --checkpoint work_dirs/phenoloss/best_AP_epoch_250.pth \\
        --data-root data/cavfish/field_collections \\
        --output-root results/field_predictions \\
        --datasets "2020_Bojonawi" "2021_Guaviare" "2022_Ayapel"
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch inference on multiple CaVFish dataset folders'
    )
    
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
        help='Path to model checkpoint (.pth)'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory containing dataset folders'
    )
    
    parser.add_argument(
        '--output-root',
        type=str,
        required=True,
        help='Root directory for output (will mirror input structure)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='List of dataset folder names to process'
    )
    
    parser.add_argument(
        '--image-exts',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'],
        help='Image file extensions to process'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip images that already have prediction JSON files'
    )
    
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Draw heatmaps in output visualizations'
    )
    
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        help='Show keypoint indices in visualizations'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run inference (cuda:0, cpu, etc.)'
    )
    
    args = parser.parse_args()
    return args


def is_image_file(filename: str, valid_exts: List[str]) -> bool:
    """Check if file has valid image extension."""
    ext = Path(filename).suffix.lower()
    return ext in [e.lower() for e in valid_exts]


def find_images(folder: str, valid_exts: List[str]) -> List[str]:
    """Recursively find all image files in folder."""
    image_paths = []
    
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if is_image_file(fname, valid_exts):
                image_paths.append(os.path.join(root, fname))
    
    return sorted(image_paths)


def build_output_paths(
    image_path: str,
    data_root: str,
    output_root: str
) -> Tuple[str, str]:
    """
    Build output paths maintaining directory structure.
    
    Returns:
        Tuple of (visualization_path, json_path)
    """
    # Get relative path from data root
    rel_path = os.path.relpath(image_path, data_root)
    rel_dir = os.path.dirname(rel_path)
    img_name = os.path.basename(rel_path)
    
    # Create output directory
    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Build output file paths
    img_stem = Path(img_name).stem
    img_ext = Path(img_name).suffix
    
    vis_path = os.path.join(out_dir, f"{img_stem}_vis{img_ext}")
    json_path = os.path.join(out_dir, f"{img_stem}_keypoints.json")
    
    return vis_path, json_path


def load_existing_prediction(json_path: str) -> dict:
    """
    Try to load existing prediction JSON.
    
    Returns:
        Dictionary with predictions, or None if file doesn't exist or is invalid
    """
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            return None
        
        pred = json.loads(content)
        return pred
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    ⚠️  Invalid JSON file ({e}), will reprocess")
        return None


def run_inference(
    image_path: str,
    config: str,
    checkpoint: str,
    vis_path: str,
    draw_heatmap: bool = False,
    show_kpt_idx: bool = False,
    device: str = 'cuda:0'
) -> bool:
    """
    Run inference on single image using demo script.
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python', 'demo/image_demo.py',
        image_path,
        config,
        checkpoint,
        '--out-file', vis_path,
        '--device', device
    ]
    
    if draw_heatmap:
        cmd.append('--draw-heatmap')
    
    if show_kpt_idx:
        cmd.append('--show-kpt-idx')
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ❌ Inference failed (exit code {result.returncode})")
        if result.stderr:
            print(f"       Error: {result.stderr[:200]}")
        return False
    
    return True


def process_dataset_folder(
    folder_name: str,
    data_root: str,
    output_root: str,
    config: str,
    checkpoint: str,
    image_exts: List[str],
    skip_existing: bool,
    draw_heatmap: bool,
    show_kpt_idx: bool,
    device: str
):
    """Process all images in a dataset folder."""
    
    # Input folder path
    input_folder = os.path.join(data_root, folder_name)
    
    if not os.path.isdir(input_folder):
        print(f"❌ Dataset folder not found: {input_folder}")
        return
    
    # Output folder path
    dataset_output = os.path.join(output_root, folder_name)
    
    print("\n" + "=" * 80)
    print(f"📂 Processing Dataset: {folder_name}")
    print(f"   Input  : {input_folder}")
    print(f"   Output : {dataset_output}")
    print("=" * 80)
    
    # Find all images
    image_paths = find_images(input_folder, image_exts)
    total = len(image_paths)
    
    print(f"\n✓ Found {total} images")
    
    if total == 0:
        print("⚠️  No images found in this dataset")
        return
    
    # Process statistics
    processed = 0
    skipped = 0
    errors = 0
    
    # Process each image
    for idx, img_path in enumerate(image_paths, start=1):
        rel_path = os.path.relpath(img_path, data_root)
        
        # Build output paths
        vis_path, json_path = build_output_paths(img_path, data_root, output_root)
        
        # Check if already processed
        if skip_existing:
            existing = load_existing_prediction(json_path)
            if existing is not None:
                print(f"[{idx}/{total}] ⏭️  Skipping (already processed): {rel_path}")
                skipped += 1
                continue
        
        # Run inference
        print(f"[{idx}/{total}] 🔄 Processing: {rel_path}")
        
        success = run_inference(
            img_path,
            config,
            checkpoint,
            vis_path,
            draw_heatmap,
            show_kpt_idx,
            device
        )
        
        if success:
            # Verify JSON was created
            if os.path.exists(json_path):
                pred = load_existing_prediction(json_path)
                if pred is not None:
                    print(f"    ✓ Success")
                    processed += 1
                else:
                    print(f"    ⚠️  Invalid JSON generated")
                    errors += 1
            else:
                print(f"    ⚠️  JSON file not created")
                errors += 1
        else:
            errors += 1
    
    # Summary for this dataset
    print("\n" + "-" * 80)
    print(f"Dataset '{folder_name}' Summary:")
    print(f"  ✓ Processed: {processed}")
    print(f"  ⏭️  Skipped:   {skipped}")
    print(f"  ❌ Errors:    {errors}")
    print("-" * 80)


def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if not os.path.isdir(args.data_root):
        raise NotADirectoryError(f"Data root not found: {args.data_root}")
    
    # Create output root
    os.makedirs(args.output_root, exist_ok=True)
    
    print("=" * 80)
    print("  CaVFish Batch Inference Tool")
    print("=" * 80)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data Root:  {args.data_root}")
    print(f"Output:     {args.output_root}")
    print(f"Datasets:   {', '.join(args.datasets)}")
    print(f"Device:     {args.device}")
    print("=" * 80)
    
    # Process each dataset
    for dataset_name in args.datasets:
        try:
            process_dataset_folder(
                dataset_name,
                args.data_root,
                args.output_root,
                args.config,
                args.checkpoint,
                args.image_exts,
                args.skip_existing,
                args.draw_heatmap,
                args.show_kpt_idx,
                args.device
            )
        except Exception as e:
            print(f"\n❌ Error processing dataset '{dataset_name}': {e}")
            continue
    
    print("\n" + "=" * 80)
    print("✅ Batch inference complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
