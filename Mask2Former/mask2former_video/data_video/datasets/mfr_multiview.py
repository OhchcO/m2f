import json
import logging
import os
from pathlib import Path

from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

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


def _join_dataset_path(split_root, rel_or_abs_path):
    if os.path.isabs(rel_or_abs_path):
        return rel_or_abs_path
    return os.path.normpath(os.path.join(split_root, rel_or_abs_path.replace("\\", "/")))


def load_mfr_multiview_json(models_json, split_root):
    models_json = Path(models_json)
    split_root = str(split_root)
    with models_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_dicts = []
    for video_idx, model in enumerate(data.get("models", [])):
        views = sorted(model.get("views", []), key=lambda item: item["view_id"])
        file_names = [_join_dataset_path(split_root, view["image"]) for view in views]
        face_id_map_names = [_join_dataset_path(split_root, view["face_id_map"]) for view in views]
        camera_directions = []
        for view in views:
            direction = view.get("camera", {}).get("direction")
            if not isinstance(direction, list) or len(direction) != 3:
                raise ValueError(f"Missing camera.direction for model {model.get('model_id')} view {view.get('view_id')}")
            camera_directions.append([float(value) for value in direction])

        if not file_names:
            continue

        with Image.open(file_names[0]) as image:
            width, height = image.size

        features = []
        annotations = []
        for feature in model.get("features", []):
            category_id = int(feature["category_id"])
            if category_id >= len(MFR_FEATURE_NAMES):
                # Stock/background is not trained as a foreground instance.
                continue
            item = {
                "id": int(feature["instance_id"]),
                "category_id": category_id,
                "iscrowd": 0,
                "face_ids": [int(face_id) for face_id in feature["face_ids"]],
                "name": feature.get("name", ""),
            }
            features.append(item)

        frame_annotations = [
            [{"id": f["id"], "category_id": f["category_id"], "iscrowd": 0} for f in features]
            for _ in views
        ]
        annotations.extend(frame_annotations)

        dataset_dicts.append(
            {
                "file_names": file_names,
                "face_id_map_names": face_id_map_names,
                "camera_directions": camera_directions,
                "height": height,
                "width": width,
                "length": len(views),
                "video_id": video_idx,
                "model_id": model["model_id"],
                "features": features,
                "annotations": annotations,
            }
        )

    logger.info("Loaded %d MFR multi-view models from %s", len(dataset_dicts), models_json)
    return dataset_dicts


def register_mfr_multiview_instances(name, metadata, models_json, split_root):
    DatasetCatalog.register(name, lambda: load_mfr_multiview_json(models_json, split_root))
    MetadataCatalog.get(name).set(
        models_json=models_json,
        image_root=split_root,
        evaluator_type="mfr_multiview",
        **metadata,
    )


def get_mfr_multiview_instances_meta():
    return {
        "thing_classes": MFR_FEATURE_NAMES,
        "thing_dataset_id_to_contiguous_id": {idx: idx for idx in range(len(MFR_FEATURE_NAMES))},
    }
