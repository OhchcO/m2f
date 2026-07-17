# -*- coding: utf-8 -*-
"""Register MFR multi-view data as independent single-image instance samples."""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


MFR_FEATURE_NAMES = [
    "chamfer",
    "through_hole",
    "triangular_passage",
    "rectangular_passage",
    "6sides_passage",
    "triangular_through_slot",
    "rectangular_through_slot",
    "circular_through_slot",
    "rectangular_through_step",
    "2sides_through_step",
    "slanted_through_step",
    "Oring",
    "blind_hole",
    "triangular_pocket",
    "rectangular_pocket",
    "6sides_pocket",
    "circular_end_pocket",
    "rectangular_blind_slot",
    "v_circular_end_blind_slot",
    "h_circular_end_blind_slot",
    "triangular_blind_step",
    "circular_blind_step",
    "rectangular_blind_step",
    "round",
]


def _dataset_basename(root):
    return os.path.basename(os.path.normpath(root))


def _join(split_root, rel_or_abs_path):
    if os.path.isabs(rel_or_abs_path):
        return rel_or_abs_path
    return os.path.normpath(os.path.join(split_root, rel_or_abs_path.replace("\\", "/")))


def _encode_binary_mask(mask):
    encoded = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(encoded.get("counts"), bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def load_mfr_singleview_json(models_json, split_root):
    models_json = Path(models_json)
    split_root = str(split_root)
    with models_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_dicts = []
    image_id = 0
    anno_id = 1

    for model in data.get("models", []):
        features = []
        for feature in model.get("features", []):
            category_id = int(feature["category_id"])
            if category_id >= len(MFR_FEATURE_NAMES):
                continue
            features.append(
                {
                    "instance_id": int(feature["instance_id"]),
                    "category_id": category_id,
                    "face_ids": [int(face_id) for face_id in feature["face_ids"]],
                }
            )

        for view in sorted(model.get("views", []), key=lambda item: item["view_id"]):
            image_path = _join(split_root, view["image"])
            face_id_map_path = _join(split_root, view["face_id_map"])
            if not os.path.isfile(image_path) or not os.path.isfile(face_id_map_path):
                continue

            with Image.open(image_path) as image:
                width, height = image.size

            face_id_map = np.load(face_id_map_path)
            annotations = []
            for feature in features:
                mask = np.isin(face_id_map, feature["face_ids"])
                if not mask.any():
                    continue

                rle = _encode_binary_mask(mask)
                bbox = mask_util.toBbox(rle).tolist()
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                annotations.append(
                    {
                        "id": anno_id,
                        "iscrowd": 0,
                        "category_id": feature["category_id"],
                        "bbox": [float(v) for v in bbox],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": rle,
                    }
                )
                anno_id += 1

            dataset_dicts.append(
                {
                    "file_name": image_path,
                    "image_id": image_id,
                    "height": height,
                    "width": width,
                    "model_id": model.get("model_id", ""),
                    "view_id": int(view.get("view_id", 0)),
                    "annotations": annotations,
                }
            )
            image_id += 1

    return dataset_dicts


def register_mfr_singleview(root=None, prefix=None):
    root = root or os.getenv("MFR_SINGLEVIEW_DATASET", "/hy-tmp/datasets/MFRInstSegM2F_2100")
    dataset_base = os.getenv("MFR_DATASET_NAME") or os.getenv("MFR_SINGLEVIEW_DATASET_NAME") or _dataset_basename(root)
    prefix = prefix or f"{dataset_base}_singleview"
    if not os.path.isdir(root):
        print(f"[WARNING] MFR single-view dataset directory not found: {root}")
        return

    for split in ("train", "val"):
        split_root = os.path.join(root, split)
        models_json = os.path.join(split_root, "models.json")
        dataset_name = f"{prefix}_{split}"

        if dataset_name in DatasetCatalog.list():
            print(f"[SKIP] Dataset already registered: {dataset_name}")
            continue
        if not os.path.isfile(models_json):
            print(f"[WARNING] MFR single-view models.json not found: {models_json}")
            continue

        DatasetCatalog.register(
            dataset_name,
            lambda models_json=models_json, split_root=split_root: load_mfr_singleview_json(models_json, split_root),
        )
        MetadataCatalog.get(dataset_name).set(
            models_json=models_json,
            image_root=split_root,
            evaluator_type="coco",
            thing_classes=MFR_FEATURE_NAMES,
            thing_dataset_id_to_contiguous_id={idx: idx for idx in range(len(MFR_FEATURE_NAMES))},
        )
        print(f"[OK] Registered MFR single-view dataset: {dataset_name}")

    # Backward-compatible aliases for older configs/scripts.
    legacy_prefix = "mfr_singleview"
    if prefix != legacy_prefix:
        for split in ("train", "val"):
            alias_name = f"{legacy_prefix}_{split}"
            target_name = f"{prefix}_{split}"
            if alias_name in DatasetCatalog.list() or target_name not in DatasetCatalog.list():
                continue
            split_root = os.path.join(root, split)
            models_json = os.path.join(split_root, "models.json")
            DatasetCatalog.register(
                alias_name,
                lambda models_json=models_json, split_root=split_root: load_mfr_singleview_json(models_json, split_root),
            )
            alias_metadata = MetadataCatalog.get(target_name).as_dict()
            alias_metadata.pop("name", None)
            MetadataCatalog.get(alias_name).set(**alias_metadata)
            print(f"[OK] Registered MFR single-view alias: {alias_name} -> {target_name}")


register_mfr_singleview()
