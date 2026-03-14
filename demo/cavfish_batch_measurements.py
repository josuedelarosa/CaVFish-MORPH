#!/usr/bin/env python
"""
Build a final phenotypic CSV from batch inference JSON outputs.

This script is designed for the CavFish batch inference layout where each
dataset folder contains a merged JSON file named:

  all_keypoints_predicted_<dataset-tag>.json

Main features:
- Aggregates all prediction JSONs under a root folder (e.g. phenoloss/)
- Resolves fish type labels from annotation JSON files and CVAT XML tags
- Supports two masking modes:
  - class-fixed: use predefined fish-type keypoint removals
  - learned: infer removals from annotation visibility/tag data
- Removes non-present keypoints by fish type (sets x/y to 0)
- Computes phenotypic measurements per image
- Exports a simplified CSV schema for publication

Usage example:
  python demo/cavfish_batch_measurements.py \
      --pred-root /data/Datasets/Fish/CavFish-Dataset/phenoloss \
      --dataset-root /data/Datasets/Fish/CavFish-Dataset \
      --output-csv /data/Datasets/Fish/CavFish-Dataset/phenoloss/final_measurements_with_fish_type.csv
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


# =====================
# CONSTANTS
# =====================

DEFAULT_DATASET_ROOT = "/data/Datasets/Fish/CavFish-Dataset"
DEFAULT_PRED_ROOT = f"{DEFAULT_DATASET_ROOT}/phenoloss"

DEFAULT_ANN_JSONS = [
    "fish20kpt_all_train_2nd-run.json",
    "fish20kpt_all_val_2nd-run.json",
]

DEFAULT_NEW_CATALOGS_XML_DIR = "new_catalogs"

N_KEYPOINTS = 20

# 0-based keypoint IDs to remove by fish type.
# Verified from fish20kpt_all_train_2nd-run + fish20kpt_all_val_2nd-run.
FIXED_INVALID_KEYPOINTS_0_BASED = {
    "Compressed body": set(),
    "Depressed body": {13},
    "Compressed body_without caudal fin": {8, 9, 17},
    # Low-sample classes kept for completeness.
    "Fusiform": {8, 9, 14, 15, 17, 18, 19},
    "Rounded body": {4, 5, 12, 13, 14, 15, 17, 18, 19},
}

MEASUREMENT_PAIRS = {
    "BI": (0, 1),
    "Bd": (2, 3),
    "Hd": (4, 5),
    "CPd": (6, 7),
    "CFd": (8, 9),
    "Ed": (10, 11),
    "Eh": (12, 3),
    "JI": (0, 13),
    "PFI": (14, 15),
    "PFi": (14, 3),
    "HL": (0, 16),
    "DL": (2, 17),
    "AL": (18, 19),
}


# =====================
# ARGUMENTS
# =====================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate prediction JSONs, apply fish-type keypoint masking, and "
            "export final phenotype CSV."
        )
    )

    parser.add_argument(
        "--pred-root",
        type=str,
        default=DEFAULT_PRED_ROOT,
        help=(
            "Root folder containing per-dataset inference outputs "
            f"(default: {DEFAULT_PRED_ROOT})"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "CavFish dataset root containing annotation JSONs and new_catalogs XMLs "
            f"(default: {DEFAULT_DATASET_ROOT})"
        ),
    )

    parser.add_argument(
        "--annotation-jsons",
        type=str,
        nargs="*",
        default=DEFAULT_ANN_JSONS,
        help=(
            "Annotation JSON files relative to dataset-root used to resolve labels "
            "and keypoint visibility stats."
        ),
    )

    parser.add_argument(
        "--xml-dir",
        type=str,
        default=DEFAULT_NEW_CATALOGS_XML_DIR,
        help=(
            "Directory (relative to dataset-root) with CVAT XML files for new catalogs "
            f"(default: {DEFAULT_NEW_CATALOGS_XML_DIR})"
        ),
    )

    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.05,
        help=(
            "Keypoint is considered non-present for a fish type if visibility ratio "
            "is below this threshold (default: 0.05). Used only in learned mode."
        ),
    )

    parser.add_argument(
        "--mask-mode",
        type=str,
        choices=["class-fixed", "learned"],
        default="class-fixed",
        help=(
            "Masking strategy. class-fixed uses predefined removals by fish type. "
            "learned infers removals from visibility statistics. "
            "(default: class-fixed)"
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Output CSV path. If omitted, uses "
            "<pred-root>/final_measurements_with_fish_type.csv"
        ),
    )

    return parser.parse_args()


# =====================
# HELPERS
# =====================


def canon_path(path_str: str) -> str:
    path_str = (path_str or "").strip().replace("\\", "/")
    while path_str.startswith("./"):
        path_str = path_str[2:]
    return path_str


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _safe_load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_prediction_jsons(pred_root: Path) -> List[Path]:
    """Prefer merged files, fallback to per-image files if needed."""
    merged = sorted(pred_root.rglob("all_keypoints_predicted_*.json"))
    if merged:
        return merged
    return sorted(pred_root.rglob("*_keypoints.json"))


def load_predictions(pred_json_paths: List[Path]) -> Dict[str, dict]:
    """Load and deduplicate predictions by canonical image path."""
    by_image: Dict[str, dict] = {}

    for path in pred_json_paths:
        try:
            payload = _safe_load_json(path)
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
            continue

        if isinstance(payload, dict):
            items = [payload]
        elif isinstance(payload, list):
            items = payload
        else:
            print(f"[WARN] Unsupported JSON format in {path}")
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            image = canon_path(item.get("image", ""))
            if not image:
                continue
            by_image[image] = item

    return by_image


def load_annotation_json_stats(
    ann_json_paths: List[Path],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """
    Returns:
      label_by_relpath: image relative path -> fish label
      label_total:      label -> {"total": int}
      label_visible:    label -> {kp_id_str: visible_count}
    """
    label_by_relpath: Dict[str, str] = {}
    label_total: Dict[str, Dict[str, int]] = {}
    label_visible: Dict[str, Dict[str, int]] = {}

    for ann_path in ann_json_paths:
        if not ann_path.exists():
            print(f"[WARN] Annotation JSON not found: {ann_path}")
            continue

        try:
            data = _safe_load_json(ann_path)
        except Exception as exc:
            print(f"[WARN] Failed to load annotation JSON {ann_path}: {exc}")
            continue

        images = data.get("images", [])
        annotations = data.get("annotations", [])

        image_meta = {}
        for img in images:
            img_id = img.get("id")
            rel = canon_path(img.get("file_name", ""))
            label = img.get("label")
            image_meta[img_id] = (rel, label)
            if rel and label:
                label_by_relpath[rel] = label

        for ann in annotations:
            img_id = ann.get("image_id")
            rel, label = image_meta.get(img_id, (None, None))
            if not rel or not label:
                continue

            kps = ann.get("keypoints", [])
            if not isinstance(kps, list) or len(kps) < 3:
                continue

            if label not in label_total:
                label_total[label] = {"total": 0}
                label_visible[label] = {str(i): 0 for i in range(N_KEYPOINTS)}

            label_total[label]["total"] += 1
            max_k = min(N_KEYPOINTS, len(kps) // 3)
            for kp_id in range(max_k):
                visibility = kps[3 * kp_id + 2]
                if visibility and float(visibility) > 0:
                    label_visible[label][str(kp_id)] += 1

    return label_by_relpath, label_total, label_visible


def load_new_catalog_xml_stats(
    xml_paths: List[Path],
    label_by_relpath: Dict[str, str],
    label_total: Dict[str, Dict[str, int]],
    label_visible: Dict[str, Dict[str, int]],
    update_visibility: bool,
) -> None:
    """
    Update label mappings and optional visibility stats from CVAT XML files.

    XML image names are usually basenames, so each is mapped as:
      <catalog-name>/<image-name>
    """
    known_shape_labels = {
        "Compressed body",
        "Depressed body",
        "Compressed body_without caudal fin",
        "Fusiform",
        "Rounded body",
        "Compressed body_ventral_mouth",
    }

    for xml_path in xml_paths:
        catalog_name = xml_path.stem
        try:
            root = ET.parse(xml_path).getroot()
        except Exception as exc:
            print(f"[WARN] Could not parse XML {xml_path}: {exc}")
            continue

        for image in root.findall("image"):
            image_name = image.get("name", "")
            rel = canon_path(f"{catalog_name}/{image_name}")

            fish_label = None
            for tag in image.findall("tag"):
                candidate = tag.get("label")
                if candidate in known_shape_labels:
                    fish_label = candidate
                    break

            if not rel or not fish_label:
                continue

            label_by_relpath[rel] = fish_label

            if not update_visibility:
                continue

            if fish_label not in label_total:
                label_total[fish_label] = {"total": 0}
                label_visible[fish_label] = {str(i): 0 for i in range(N_KEYPOINTS)}

            label_total[fish_label]["total"] += 1

            present_kps: Set[int] = set()
            for point in image.findall("points"):
                kp_name = point.get("label", "")
                if kp_name.isdigit():
                    # XML uses 1..20, predictions use 0..19
                    kp_id = int(kp_name) - 1
                    if 0 <= kp_id < N_KEYPOINTS:
                        present_kps.add(kp_id)

            for kp_id in present_kps:
                label_visible[fish_label][str(kp_id)] += 1


def build_label_lookup(label_by_relpath: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      by_relpath: canonical relative path -> label
      by_basename_unique: basename -> label only if unambiguous
    """
    basename_to_labels: Dict[str, Set[str]] = {}
    for rel, label in label_by_relpath.items():
        base = os.path.basename(rel)
        basename_to_labels.setdefault(base, set()).add(label)

    by_basename_unique = {
        base: list(labels)[0]
        for base, labels in basename_to_labels.items()
        if len(labels) == 1
    }

    return label_by_relpath, by_basename_unique


def compute_invalid_keypoints_by_type(
    label_total: Dict[str, Dict[str, int]],
    label_visible: Dict[str, Dict[str, int]],
    threshold: float,
) -> Dict[str, Set[int]]:
    invalid_by_type: Dict[str, Set[int]] = {}

    for label, total_dict in label_total.items():
        total = int(total_dict.get("total", 0))
        if total <= 0:
            invalid_by_type[label] = set()
            continue

        invalid_kps: Set[int] = set()
        for kp_id in range(N_KEYPOINTS):
            visible = int(label_visible.get(label, {}).get(str(kp_id), 0))
            ratio = visible / float(total)
            if ratio < threshold:
                invalid_kps.add(kp_id)

        invalid_by_type[label] = invalid_kps

    return invalid_by_type


def split_catalog_and_image(image_relpath: str) -> Tuple[str, str]:
    catalog = canon_path(os.path.dirname(image_relpath))
    image_name = os.path.basename(image_relpath) if image_relpath else "unknown"
    if not image_name:
        image_name = image_relpath or "unknown"
    return catalog, image_name


def measurement_row_from_prediction(
    pred: dict,
    fish_type: str,
    invalid_kps: Set[int],
) -> dict:
    image_relpath = canon_path(pred.get("image", "unknown"))
    catalog, image_name = split_catalog_and_image(image_relpath)
    keypoints = pred.get("keypoints", [])

    coords: Dict[int, Tuple[float, float]] = {}

    for kp in keypoints:
        try:
            kp_id = int(kp.get("name"))
        except Exception:
            continue
        if kp_id < 0 or kp_id >= N_KEYPOINTS:
            continue

        x = float(kp.get("x", 0.0))
        y = float(kp.get("y", 0.0))

        # Fish-type keypoint removal: force to zero for non-present keypoints.
        if kp_id in invalid_kps:
            x, y = 0.0, 0.0

        coords[kp_id] = (x, y)

    row: Dict[str, float | str] = {
        "catalog": catalog,
        "image": image_name,
        "fish_type": fish_type,
    }

    for kp_id in range(N_KEYPOINTS):
        x, y = coords.get(kp_id, (0.0, 0.0))
        row[f"kp{kp_id}_x"] = x
        row[f"kp{kp_id}_y"] = y

    for measurement, (i, j) in MEASUREMENT_PAIRS.items():
        valid = (i not in invalid_kps) and (j not in invalid_kps)
        if valid and i in coords and j in coords:
            row[measurement] = euclidean_distance(coords[i], coords[j])
        else:
            row[measurement] = 0.0

    return row


def main() -> None:
    args = parse_args()

    pred_root = Path(args.pred_root)
    dataset_root = Path(args.dataset_root)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else pred_root / "final_measurements_with_fish_type.csv"
    )

    print("=" * 80)
    print("Building final CavFish phenotype CSV")
    print(f"Pred root         : {pred_root}")
    print(f"Dataset root      : {dataset_root}")
    print(f"Mask mode         : {args.mask_mode}")
    print(f"Presence threshold: {args.presence_threshold}")
    print(f"Output CSV        : {output_csv}")
    print("=" * 80)

    pred_json_paths = discover_prediction_jsons(pred_root)
    print(f"Prediction JSON files found: {len(pred_json_paths)}")
    if not pred_json_paths:
        raise FileNotFoundError(f"No prediction JSON files found under: {pred_root}")

    pred_by_image = load_predictions(pred_json_paths)
    print(f"Unique predicted images: {len(pred_by_image)}")

    ann_json_paths = [dataset_root / path for path in args.annotation_jsons]
    label_by_relpath, label_total, label_visible = load_annotation_json_stats(ann_json_paths)

    learned_mode = args.mask_mode == "learned"

    xml_dir = dataset_root / args.xml_dir
    xml_paths = sorted(xml_dir.glob("*.xml")) if xml_dir.exists() else []
    if xml_paths:
        load_new_catalog_xml_stats(
            xml_paths=xml_paths,
            label_by_relpath=label_by_relpath,
            label_total=label_total,
            label_visible=label_visible,
            update_visibility=learned_mode,
        )

    print(f"Label map entries (relative path): {len(label_by_relpath)}")
    print(f"Fish types discovered: {sorted(label_total.keys())}")

    by_relpath, by_basename_unique = build_label_lookup(label_by_relpath)

    if learned_mode:
        invalid_by_type = compute_invalid_keypoints_by_type(
            label_total=label_total,
            label_visible=label_visible,
            threshold=args.presence_threshold,
        )
        for fish_type in sorted(invalid_by_type.keys()):
            print(
                f"[INFO] {fish_type}: invalid keypoints (by threshold) = "
                f"{sorted(invalid_by_type[fish_type])}"
            )
    else:
        invalid_by_type = {
            fish_type: set(keypoints)
            for fish_type, keypoints in FIXED_INVALID_KEYPOINTS_0_BASED.items()
        }
        for fish_type in sorted(invalid_by_type.keys()):
            print(
                f"[INFO] {fish_type}: invalid keypoints (class-fixed) = "
                f"{sorted(invalid_by_type[fish_type])}"
            )

    rows = []
    unknown_count = 0

    for image_path, pred in pred_by_image.items():
        fish_type = by_relpath.get(image_path)
        if fish_type is None:
            fish_type = by_basename_unique.get(os.path.basename(image_path), "unknown")

        if fish_type == "unknown":
            unknown_count += 1

        invalid_kps = invalid_by_type.get(fish_type, set())
        rows.append(measurement_row_from_prediction(pred, fish_type, invalid_kps))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["fish_type", "catalog", "image"], kind="stable").reset_index(
            drop=True
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print("Done")
    print(f"Rows written      : {len(df)}")
    print(f"Unknown fish_type : {unknown_count}")
    print(f"Output CSV        : {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
