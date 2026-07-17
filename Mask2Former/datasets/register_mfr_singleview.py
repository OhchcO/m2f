# -*- coding: utf-8 -*-
"""Register MFR multi-view data as independent single-image instance samples."""

import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
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

SINGLEVIEW_CACHE_VERSION = 1


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


def _cache_file_for(models_json):
    return models_json.with_name(f".{models_json.stem}_singleview_cache_v{SINGLEVIEW_CACHE_VERSION}.pkl")


def _cache_meta(models_json):
    stat = models_json.stat()
    return {
        "version": SINGLEVIEW_CACHE_VERSION,
        "models_json": str(models_json),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _load_cache(cache_file, expected_meta):
    if not cache_file.is_file():
        return None
    try:
        with cache_file.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"[WARNING] Failed to read single-view cache {cache_file}: {exc}", flush=True)
        return None

    if payload.get("meta") != expected_meta:
        return None
    return payload.get("dataset_dicts")


def _save_cache(cache_file, expected_meta, dataset_dicts):
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    try:
        with tmp_file.open("wb") as f:
            pickle.dump({"meta": expected_meta, "dataset_dicts": dataset_dicts}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_file, cache_file)
        print(f"[OK] Saved MFR single-view cache: {cache_file}", flush=True)
    except Exception as exc:
        print(f"[WARNING] Failed to save single-view cache {cache_file}: {exc}", flush=True)
        try:
            tmp_file.unlink(missing_ok=True)
        except TypeError:
            if tmp_file.exists():
                tmp_file.unlink()


def _model_to_records(model, split_root):
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

    records = []
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
                    "iscrowd": 0,
                    "category_id": feature["category_id"],
                    "bbox": [float(v) for v in bbox],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": rle,
                }
            )

        records.append(
            {
                "file_name": image_path,
                "height": height,
                "width": width,
                "model_id": model.get("model_id", ""),
                "view_id": int(view.get("view_id", 0)),
                "annotations": annotations,
            }
        )
    return records


def load_mfr_singleview_json(models_json, split_root):
    models_json = Path(models_json)
    split_root = str(split_root)
    use_cache = os.getenv("MFR_SINGLEVIEW_CACHE", "0") == "1"
    expected_meta = _cache_meta(models_json)
    cache_file = _cache_file_for(models_json)
    if use_cache:
        cached = _load_cache(cache_file, expected_meta)
        if cached is not None:
            print(f"[OK] Loaded MFR single-view cache: {cache_file} ({len(cached)} images)", flush=True)
            return cached

    with models_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_dicts = []
    image_id = 0
    anno_id = 1
    models = data.get("models", [])
    log_period = int(os.getenv("MFR_SINGLEVIEW_LOG_PERIOD", "50"))
    build_workers = int(os.getenv("MFR_SINGLEVIEW_BUILD_WORKERS", "8"))
    start_time = time.time()
    print(
        f"[MFR single-view] Building dataset from {models_json}: {len(models)} models. "
        f"This can take a while because masks are generated from face_id_maps. workers={build_workers}",
        flush=True,
    )

    def consume(iterator):
        nonlocal image_id, anno_id
        for model_idx, records in enumerate(iterator, start=1):
            for record in records:
                record["image_id"] = image_id
                image_id += 1
                for annotation in record["annotations"]:
                    annotation["id"] = anno_id
                    anno_id += 1
                dataset_dicts.append(record)

            if log_period > 0 and (model_idx == 1 or model_idx % log_period == 0 or model_idx == len(models)):
                elapsed = time.time() - start_time
                print(
                    f"[MFR single-view] {models_json.parent.name}: "
                    f"{model_idx}/{len(models)} models, {image_id} images, {anno_id - 1} annotations, "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

    if build_workers > 1 and len(models) > 1:
        with ThreadPoolExecutor(max_workers=build_workers) as executor:
            consume(executor.map(lambda model: _model_to_records(model, split_root), models))
    else:
        consume(_model_to_records(model, split_root) for model in models)

    if use_cache:
        _save_cache(cache_file, expected_meta, dataset_dicts)
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
