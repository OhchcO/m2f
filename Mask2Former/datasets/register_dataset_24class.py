# -*- coding: utf-8 -*-
"""Register the 24-class machining feature dataset."""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json


DATASET_24CLASS_CATEGORIES = [
    {"id": 0, "name": "through_hole"},
    {"id": 1, "name": "triangular_passage"},
    {"id": 2, "name": "rectangular_passage"},
    {"id": 3, "name": "6sides_passage"},
    {"id": 4, "name": "triangular_through_slot"},
    {"id": 5, "name": "rectangular_through_slot"},
    {"id": 6, "name": "circular_through_slot"},
    {"id": 7, "name": "rectangular_through_step"},
    {"id": 8, "name": "2sides_through_step"},
    {"id": 9, "name": "slanted_through_step"},
    {"id": 10, "name": "Oring"},
    {"id": 11, "name": "blind_hole"},
    {"id": 12, "name": "triangular_pocket"},
    {"id": 13, "name": "rectangular_pocket"},
    {"id": 14, "name": "6sides_pocket"},
    {"id": 15, "name": "circular_end_pocket"},
    {"id": 16, "name": "rectangular_blind_slot"},
    {"id": 17, "name": "v_circular_end_blind_slot"},
    {"id": 18, "name": "h_circular_end_blind_slot"},
    {"id": 19, "name": "triangular_blind_step"},
    {"id": 20, "name": "circular_blind_step"},
    {"id": 21, "name": "rectangular_blind_step"},
    {"id": 22, "name": "round"},
]


def _dataset_root():
    datasets_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(datasets_dir))
    return os.path.join(project_root, "temp_data", "dataset_24class")


def _load_split(image_root, json_file):
    dataset_dicts = load_coco_json(json_file, image_root, dataset_name=None)
    valid_ids = {category["id"] for category in DATASET_24CLASS_CATEGORIES}
    for record in dataset_dicts:
        record["annotations"] = [
            anno for anno in record.get("annotations", [])
            if anno.get("category_id") in valid_ids
        ]
    return dataset_dicts


def register_dataset_24class(root=None, prefix="temp_data_24class"):
    root = root or _dataset_root()
    if not os.path.isdir(root):
        print(f"[WARNING] 24-class dataset directory not found: {root}")
        return

    for split in ("train", "val"):
        image_root = os.path.join(root, split, "encoded_views")
        json_file = os.path.join(root, split, "instances.json")
        dataset_name = f"{prefix}_{split}"

        if dataset_name in DatasetCatalog.list():
            print(f"[SKIP] Dataset already registered: {dataset_name}")
            continue
        if not os.path.isdir(image_root):
            print(f"[WARNING] 24-class image directory not found: {image_root}")
            continue
        if not os.path.isfile(json_file):
            print(f"[WARNING] 24-class annotation file not found: {json_file}")
            continue

        DatasetCatalog.register(
            dataset_name,
            lambda image_root=image_root, json_file=json_file: _load_split(image_root, json_file),
        )
        MetadataCatalog.get(dataset_name).set(
            json_file=json_file,
            image_root=image_root,
            evaluator_type="coco",
            thing_classes=[category["name"] for category in DATASET_24CLASS_CATEGORIES],
            thing_dataset_id_to_contiguous_id={
                category["id"]: index for index, category in enumerate(DATASET_24CLASS_CATEGORIES)
            },
        )
        print(f"[OK] Registered 24-class dataset: {dataset_name}")


register_dataset_24class()
