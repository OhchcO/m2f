# -*- coding: utf-8 -*-
"""Register the AAG 24-class machining feature dataset."""

import json
import os

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


AAG_24CLASS_CATEGORIES = [
    {"id": 0, "name": "chamfer"},
    {"id": 1, "name": "through_hole"},
    {"id": 2, "name": "triangular_passage"},
    {"id": 3, "name": "rectangular_passage"},
    {"id": 4, "name": "6sides_passage"},
    {"id": 5, "name": "triangular_through_slot"},
    {"id": 6, "name": "rectangular_through_slot"},
    {"id": 7, "name": "circular_through_slot"},
    {"id": 8, "name": "rectangular_through_step"},
    {"id": 9, "name": "2sides_through_step"},
    {"id": 10, "name": "slanted_through_step"},
    {"id": 11, "name": "Oring"},
    {"id": 12, "name": "blind_hole"},
    {"id": 13, "name": "triangular_pocket"},
    {"id": 14, "name": "rectangular_pocket"},
    {"id": 15, "name": "6sides_pocket"},
    {"id": 16, "name": "circular_end_pocket"},
    {"id": 17, "name": "rectangular_blind_slot"},
    {"id": 18, "name": "v_circular_end_blind_slot"},
    {"id": 19, "name": "h_circular_end_blind_slot"},
    {"id": 20, "name": "triangular_blind_step"},
    {"id": 21, "name": "circular_blind_step"},
    {"id": 22, "name": "rectangular_blind_step"},
    {"id": 23, "name": "round"},
]


def _dataset_root():
    datasets_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(datasets_dir))
    return os.path.join(project_root, "temp_data", "dataset_aag_24class")


def _load_class_map(class_map_path):
    with open(class_map_path, "r", encoding="utf-8") as f:
        raw_class_map = json.load(f)

    class_map = {}
    for item in raw_class_map.values():
        model_name = item["model_name"]
        gray_value = int(item["gray_value"])
        class_id = int(item["class_id"])
        class_map.setdefault(model_name, {})[gray_value] = class_id
    return class_map


def _mask_to_rle(binary_mask):
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _mask_to_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def _load_split(image_root, mask_root, class_map_path):
    class_map = _load_class_map(class_map_path)
    image_files = sorted([
        f for f in os.listdir(image_root)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])

    dataset_dicts = []
    valid_class_ids = {category["id"] for category in AAG_24CLASS_CATEGORIES}

    for image_id, image_file in enumerate(image_files):
        image_path = os.path.join(image_root, image_file)
        mask_path = os.path.join(mask_root, image_file)
        if not os.path.isfile(mask_path):
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        mask = np.array(Image.open(mask_path))
        model_name = image_file.rsplit("_", 1)[0]
        image_class_map = class_map.get(model_name, {})

        annotations = []
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[(instance_ids >= 0) & (instance_ids < 255)]

        for instance_id in instance_ids:
            instance_id = int(instance_id)
            class_id = image_class_map.get(instance_id)
            if class_id not in valid_class_ids:
                continue

            binary_mask = mask == instance_id
            area = int(binary_mask.sum())
            if area < 10:
                continue

            annotations.append({
                "bbox": _mask_to_bbox(binary_mask),
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": class_id,
                "segmentation": _mask_to_rle(binary_mask),
                "area": area,
                "iscrowd": 0,
            })

        dataset_dicts.append({
            "file_name": image_path,
            "height": height,
            "width": width,
            "image_id": image_id,
            "annotations": annotations,
        })

    return dataset_dicts


def register_dataset_aag_24class(root=None, prefix="temp_data_aag_24class"):
    root = root or _dataset_root()
    if not os.path.isdir(root):
        print(f"[WARNING] AAG 24-class dataset directory not found: {root}")
        return

    for split in ("train", "val"):
        image_root = os.path.join(root, split, "encoded_views")
        mask_root = os.path.join(root, split, "masks")
        class_map_path = os.path.join(root, split, "class_map.json")
        dataset_name = f"{prefix}_{split}"

        if dataset_name in DatasetCatalog.list():
            print(f"[SKIP] Dataset already registered: {dataset_name}")
            continue
        if not os.path.isdir(image_root):
            print(f"[WARNING] AAG 24-class image directory not found: {image_root}")
            continue
        if not os.path.isdir(mask_root):
            print(f"[WARNING] AAG 24-class mask directory not found: {mask_root}")
            continue
        if not os.path.isfile(class_map_path):
            print(f"[WARNING] AAG 24-class class_map file not found: {class_map_path}")
            continue

        DatasetCatalog.register(
            dataset_name,
            lambda image_root=image_root, mask_root=mask_root, class_map_path=class_map_path:
                _load_split(image_root, mask_root, class_map_path),
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=image_root,
            mask_root=mask_root,
            class_map_path=class_map_path,
            evaluator_type="coco",
            thing_classes=[category["name"] for category in AAG_24CLASS_CATEGORIES],
            thing_dataset_id_to_contiguous_id={
                category["id"]: index for index, category in enumerate(AAG_24CLASS_CATEGORIES)
            },
        )
        print(f"[OK] Registered AAG 24-class dataset: {dataset_name}")


register_dataset_aag_24class()
