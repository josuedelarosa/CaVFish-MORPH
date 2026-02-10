#!/usr/bin/env python
"""
Compute phenotypic measurements from predicted keypoints.

This script reads the consolidated JSON output from batch inference
and computes morphometric measurements based on pairwise Euclidean distances
between anatomical keypoints.

Usage:
    python demo/compute_phenotypic_measurements.py <input_json> [--output OUTPUT_CSV]

Example:
    python demo/compute_phenotypic_measurements.py ../demo/baseline_mse/inference_files/all_keypoints_predicted_files.json --output measurements.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


# ======================
# PHENOTYPIC MEASUREMENTS
# ======================

# Anatomical keypoint pairs for morphometric measurements
# Based on the CaVFish anatomical definitions
MEASUREMENT_PAIRS = {
    "BI": (0, 1),       # Body Index: Upper Snout Tip → Caudal Peduncle Center
    "Bd": (2, 3),       # Body Depth: Dorsal Body → Pelvic Fin Base
    "Hd": (4, 5),       # Head Depth: Upper Head → Barbel Base
    "CPd": (6, 7),      # Caudal Peduncle Depth: Mid-Dorsal Trunk → Ventral Trunk
    "CFd": (8, 9),      # Caudal Fin Depth: Upper Caudal Base → Lower Caudal Fin Tip
    "Ed": (10, 11),     # Eye Diameter: Eye Center → Posterior Eye Margin
    "Eh": (12, 3),      # Eye Height: Lower Eye Margin → Pelvic Fin Base
    "JI": (0, 13),      # Jaw Index: Upper Snout Tip → Lower Jaw Tip
    "PFI": (14, 15),    # Pelvic Fin Index: Operculum Lower Edge → Pelvic Fin Tip
    "PFi": (14, 3),     # Pelvic Fin insertion: Operculum Lower Edge → Pelvic Fin Base
    "HL": (0, 16),      # Head Length: Upper Snout Tip → Operculum Upper Edge
    "DL": (2, 17),      # Dorsal Length: Dorsal Body → Dorsal Fin Tip
    "AL": (18, 19),     # Anal Length: Anal Fin Base → Anal Fin Tip
}


def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two 2D points."""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def compute_measurements_for_image(keypoints_data: Dict) -> Dict:
    """
    Compute all phenotypic measurements for a single image.
    
    Args:
        keypoints_data: Dictionary with 'keypoints' containing list of 
                       {'name', 'x', 'y', 'score'} entries
                       
    Returns:
        Dictionary with measurement names as keys and distances as values
    """
    # Extract keypoint coordinates
    keypoints = keypoints_data.get('keypoints', [])
    
    # Create coordinate array indexed by keypoint ID
    coords = {}
    for kp in keypoints:
        kp_id = int(kp['name'])  # Keypoint name is the ID (0-19)
        coords[kp_id] = (kp['x'], kp['y'])
    
    # Compute all measurements
    measurements = {}
    measurements['image'] = keypoints_data.get('image', 'unknown')
    
    for measure_name, (idx1, idx2) in MEASUREMENT_PAIRS.items():
        if idx1 in coords and idx2 in coords:
            dist = euclidean_distance(coords[idx1], coords[idx2])
            measurements[measure_name] = dist
        else:
            # Missing keypoint
            measurements[measure_name] = np.nan
            
    # Add BI-normalized measurements (Body Index normalization)
    if 'BI' in measurements and not np.isnan(measurements['BI']) and measurements['BI'] > 0:
        for measure_name in MEASUREMENT_PAIRS.keys():
            if measure_name != 'BI' and measure_name in measurements:
                norm_name = f"{measure_name}_norm"  # e.g., "HL_norm" = HL/BI
                measurements[norm_name] = measurements[measure_name] / measurements['BI']
    
    # Add keypoint confidence scores (average)
    scores = [kp['score'] for kp in keypoints]
    measurements['mean_confidence'] = np.mean(scores) if scores else np.nan
    measurements['min_confidence'] = np.min(scores) if scores else np.nan
    
    return measurements


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute phenotypic measurements from keypoint predictions')
    parser.add_argument(
        'input_json',
        help='Path to consolidated JSON file with predicted keypoints')
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV file path (default: same as input with .csv extension)')
    parser.add_argument(
        '--format',
        choices=['csv', 'json', 'excel'],
        default='csv',
        help='Output format')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load keypoints JSON
    print(f'Loading keypoints from {args.input_json}...')
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    # Handle both single image dict and list of images
    if isinstance(data, dict):
        # Single image format: {"image": "...", "keypoints": [...]}
        keypoints_list = [data]
        print(f'Found 1 image (single-image format)')
    elif isinstance(data, list):
        # Batch format: [{"image": "...", "keypoints": [...]}, ...]
        keypoints_list = data
        print(f'Found {len(keypoints_list)} images (batch format)')
    else:
        raise ValueError(f'Unexpected JSON format: expected dict or list, got {type(data)}')
    
    # Compute measurements for all images
    all_measurements = []
    for idx, kp_data in enumerate(keypoints_list):
        try:
            measurements = compute_measurements_for_image(kp_data)
            all_measurements.append(measurements)
        except Exception as e:
            print(f'Warning: Error processing image {idx}: {e}')
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_measurements)
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.input_json)
        args.output = input_path.parent / f"{input_path.stem}_measurements.{args.format}"
    
    # Save results
    print(f'\nSaving {len(df)} measurements to {args.output}...')
    
    if args.format == 'csv':
        df.to_csv(args.output, index=False)
    elif args.format == 'json':
        df.to_json(args.output, orient='records', indent=2)
    elif args.format == 'excel':
        df.to_excel(args.output, index=False)
    
    # Print summary statistics
    print('\n' + '='*80)
    print('SUMMARY STATISTICS (in pixels)')
    print('='*80)
    
    # Show measurement columns (exclude metadata)
    measure_cols = [col for col in df.columns 
                   if col not in ['image', 'mean_confidence', 'min_confidence']
                   and not col.endswith('_norm')]
    
    if measure_cols:
        summary = df[measure_cols].describe()
        print(summary)
        
        print('\n' + '='*80)
        print('SL-NORMALIZED MEASUREMENTS (unitless)')
        print('='*80)
        norm_cols = [col for col in df.columns if col.endswith('_norm')]
        if norm_cols:
            summary_norm = df[norm_cols].describe()
            print(summary_norm)
    
    print(f'\n✅ Phenotypic measurements saved to: {args.output}')
    print(f'   Total images processed: {len(df)}')
    print(f'   Mean keypoint confidence: {df["mean_confidence"].mean():.3f}')


if __name__ == '__main__':
    main()
