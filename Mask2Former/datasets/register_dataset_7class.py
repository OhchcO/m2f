# -*- coding: utf-8 -*-
"""Register the 7-class machining feature dataset."""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json


DATASET_7CLASS_CATEGORIES = [
    {"id": 0, "name": "hole"},
    {"id": 1, "name": "closed_pocket"},
    {"id": 2, "name": "closed_slot"},
    {"id": 3, "name": "open_pocket"},
    {"id": 4, "name": "open_slot"},
    {"id": 5, "name": "wide_slot"},
    {"id": 6, "name": "oring_slot"},
]


def _dataset_root():
    datasets_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(datasets_dir))
    return os.path.join(project_root, "temp_data", "dataset_7class")


def _load_split(image_root, json_file):
    dataset_dicts = load_coco_json(json_file, image_root, dataset_name=None)
    valid_ids = {category["id"] for category in DATASET_7CLASS_CATEGORIES}

    for record in dataset_dicts:
        fixed_annotations = []
        for anno in record.get("annotations", []):
            fixed_category_id = anno.get("category_id") + 1
            if fixed_category_id not in valid_ids:
                continue
            anno = anno.copy()
            anno["category_id"] = fixed_category_id
            fixed_annotations.append(anno)
        record["annotations"] = fixed_annotations

    return dataset_dicts


def register_dataset_7class(root=None, prefix="temp_data_7class"):
    root = root or _dataset_root()
    if not os.path.isdir(root):
        print(f"[WARNING] 7-class dataset directory not found: {root}")
        return

    for split in ("train", "val"):
        image_root = os.path.join(root, split, "encoded_views")
        json_file = os.path.join(root, split, "instances.json")
        dataset_name = f"{prefix}_{split}"

        if dataset_name in DatasetCatalog.list():
            print(f"[SKIP] Dataset already registered: {dataset_name}")
            continue
        if not os.path.isdir(image_root):
            print(f"[WARNING] 7-class image directory not found: {image_root}")
            continue
        if not os.path.isfile(json_file):
            print(f"[WARNING] 7-class annotation file not found: {json_file}")
            continue

        DatasetCatalog.register(
            dataset_name,
            lambda image_root=image_root, json_file=json_file: _load_split(image_root, json_file),
        )
        MetadataCatalog.get(dataset_name).set(
            source_json_file=json_file,
            image_root=image_root,
            evaluator_type="coco",
            thing_classes=[category["name"] for category in DATASET_7CLASS_CATEGORIES],
            thing_dataset_id_to_contiguous_id={
                category["id"]: index for index, category in enumerate(DATASET_7CLASS_CATEGORIES)
            },
        )
        print(f"[OK] Registered 7-class dataset: {dataset_name}")


register_dataset_7class()
