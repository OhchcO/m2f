"""
Evaluate MFR multi-view VideoMaskFormer checkpoints.

This script reads the new multi-view dataset:
  val/models.json
  val/encoded_views/*.png
  val/face_id_maps/*.npy

For each CAD model, it runs the 14 views together through VideoMaskFormer,
projects predicted query masks back to face ids using face_id_maps, and reports
face-level classification metrics.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK2FORMER_DIR = os.path.join(PROJECT_DIR, "Mask2Former")
if MASK2FORMER_DIR not in sys.path:
    sys.path.insert(0, MASK2FORMER_DIR)

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config  # noqa: F401

sys.path.insert(0, os.path.dirname(__file__))
from eval_area_metrics import (  # noqa: E402
    compute_metrics,
    print_report,
    remap_results_for_coarse_eval,
    save_eval_outputs,
)

MFR_CLASS_NAMES = {
    0: "chamfer",
    1: "through_hole",
    2: "triangular_passage",
    3: "rectangular_passage",
    4: "6sides_passage",
    5: "triangular_through_slot",
    6: "rectangular_through_slot",
    7: "circular_through_slot",
    8: "rectangular_through_step",
    9: "2sides_through_step",
    10: "slanted_through_step",
    11: "Oring",
    12: "blind_hole",
    13: "triangular_pocket",
    14: "rectangular_pocket",
    15: "6sides_pocket",
    16: "circular_end_pocket",
    17: "rectangular_blind_slot",
    18: "v_circular_end_blind_slot",
    19: "h_circular_end_blind_slot",
    20: "triangular_blind_step",
    21: "circular_blind_step",
    22: "rectangular_blind_step",
    23: "round",
}

MFR_COARSE_CLASS_NAMES = {
    0: "chamfer",
    1: "hole",
    2: "closed_pocket",
    3: "closed_slot",
    4: "open_pocket",
    5: "open_slot",
    6: "wide_slot",
    7: "oring_slot",
}

MFR_FINE_TO_COARSE_CLASS = {
    0: 0,   # chamfer -> chamfer
    1: 1,   # through_hole -> hole
    2: 4,   # triangular_passage -> open_pocket
    3: 4,   # rectangular_passage -> open_pocket
    4: 4,   # 6sides_passage -> open_pocket
    5: 5,   # triangular_through_slot -> open_slot
    6: 5,   # rectangular_through_slot -> open_slot
    7: 5,   # circular_through_slot -> open_slot
    8: 6,   # rectangular_through_step -> wide_slot
    9: 6,   # 2sides_through_step -> wide_slot
    10: 6,  # slanted_through_step -> wide_slot
    11: 7,  # Oring -> oring_slot
    12: 1,  # blind_hole -> hole
    13: 2,  # triangular_pocket -> closed_pocket
    14: 2,  # rectangular_pocket -> closed_pocket
    15: 2,  # 6sides_pocket -> closed_pocket
    16: 2,  # circular_end_pocket -> closed_pocket
    17: 3,  # rectangular_blind_slot -> closed_slot
    18: 3,  # v_circular_end_blind_slot -> closed_slot
    19: 3,  # h_circular_end_blind_slot -> closed_slot
    20: 2,  # triangular_blind_step -> closed_pocket
    21: 2,  # circular_blind_step -> closed_pocket
    22: 2,  # rectangular_blind_step -> closed_pocket
    23: 1,  # round -> hole
}


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.DATASETS.TRAIN = ("mfr_multiview_train",)
    cfg.DATASETS.TEST = ("mfr_multiview_val",)
    cfg.INPUT.DATASET_MAPPER_NAME = "mfr_multiview"
    cfg.INPUT.SAMPLING_FRAME_NUM = args.num_views
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.freeze()

    # Ensure metadata exists even when builtin auto-registration has not run.
    MetadataCatalog.get("mfr_multiview_train").set(thing_classes=list(MFR_CLASS_NAMES.values()))
    MetadataCatalog.get("mfr_multiview_val").set(thing_classes=list(MFR_CLASS_NAMES.values()))
    return cfg


def load_models(val_dir):
    models_path = os.path.join(val_dir, "models.json")
    with open(models_path, "r", encoding="utf-8") as f:
        return json.load(f)["models"]


def resolve_path(root, path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path.replace("\\", "/")))


def read_video_inputs(model_record, val_dir, input_format):
    images = []
    face_id_maps = []
    camera_directions = []
    views = sorted(model_record["views"], key=lambda item: item["view_id"])
    for view in views:
        image_path = resolve_path(val_dir, view["image"])
        face_map_path = resolve_path(val_dir, view["face_id_map"])
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if input_format == "RGB" else image_bgr
        images.append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).float())
        face_id_maps.append(np.load(face_map_path).astype(np.int32))
        direction = view.get("camera", {}).get("direction")
        if not isinstance(direction, list) or len(direction) != 3:
            raise ValueError(f"Missing camera.direction for {model_record['model_id']} view {view['view_id']}")
        camera_directions.append(torch.tensor(direction, dtype=torch.float32))
    return images, face_id_maps, camera_directions


def build_gt_faces(model_record, face_id_maps):
    face_visible_area = defaultdict(int)
    for face_id_map in face_id_maps:
        face_ids, counts = np.unique(face_id_map, return_counts=True)
        for face_id, count in zip(face_ids, counts):
            face_id = int(face_id)
            if face_id >= 0:
                face_visible_area[face_id] += int(count)

    gt_faces = []
    for feature in model_record["features"]:
        category_id = int(feature["category_id"])
        if category_id >= len(MFR_CLASS_NAMES):
            continue
        for face_id in feature["face_ids"]:
            face_id = int(face_id)
            area = face_visible_area.get(face_id, 0)
            if area <= 0:
                continue
            gt_faces.append(
                {
                    "instance_id": face_id,
                    "feature_instance_id": int(feature["instance_id"]),
                    "gt_class": category_id,
                    "area": area,
                }
            )
    return gt_faces


def project_predictions_to_faces(output, face_id_maps, score_threshold, mask_threshold):
    face_votes = defaultdict(Counter)
    pred_scores = {}
    pred_face_sets = defaultdict(set)

    for query_idx, (score, label, mask_tensor) in enumerate(
        zip(output["pred_scores"], output["pred_labels"], output["pred_masks"]),
        start=1,
    ):
        score = float(score)
        if score < score_threshold:
            continue
        label = int(label)
        mask_video = mask_tensor.numpy()
        for view_idx, face_id_map in enumerate(face_id_maps):
            if view_idx >= mask_video.shape[0]:
                break
            pred_mask = mask_video[view_idx] > mask_threshold
            if pred_mask.shape != face_id_map.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (face_id_map.shape[1], face_id_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            face_ids = face_id_map[pred_mask]
            face_ids = face_ids[face_ids >= 0]
            if face_ids.size == 0:
                continue
            unique_face_ids, counts = np.unique(face_ids, return_counts=True)
            for face_id, count in zip(unique_face_ids, counts):
                face_id = int(face_id)
                face_votes[face_id][(query_idx, label)] += int(count)
                pred_face_sets[query_idx].add(face_id)
                pred_scores[query_idx] = score

    face_predictions = {}
    for face_id, votes in face_votes.items():
        (query_idx, label), pixels = votes.most_common(1)[0]
        total_pixels = sum(votes.values())
        face_predictions[face_id] = {
            "query_id": query_idx,
            "label_id": label,
            "score": pred_scores.get(query_idx, 0.0),
            "vote_pixels": pixels,
            "vote_ratio": pixels / total_pixels if total_pixels else 0.0,
            "pred_face_set": pred_face_sets.get(query_idx, set()),
        }
    return face_predictions


def evaluate_model(model_record, output, face_id_maps, score_threshold, mask_threshold):
    gt_faces = build_gt_faces(model_record, face_id_maps)
    face_predictions = project_predictions_to_faces(output, face_id_maps, score_threshold, mask_threshold)

    results = []
    for gt in gt_faces:
        face_id = int(gt["instance_id"])
        pred = face_predictions.get(face_id)
        gt_feature = {
            int(fid)
            for feature in model_record["features"]
            if int(feature["instance_id"]) == int(gt["feature_instance_id"])
            for fid in feature["face_ids"]
        }
        if pred is None:
            results.append(
                {
                    "model_id": model_record["model_id"],
                    "instance_id": face_id,
                    "feature_instance_id": gt["feature_instance_id"],
                    "gt_class": gt["gt_class"],
                    "pred_class": -1,
                    "is_correct": False,
                    "area": gt["area"],
                    "face_iou": 0.0,
                    "vote_ratio": 0.0,
                    "score": 0.0,
                }
            )
            continue

        pred_face_set = pred["pred_face_set"]
        union = gt_feature | pred_face_set
        inter = gt_feature & pred_face_set
        face_iou = len(inter) / len(union) if union else 0.0
        pred_class = int(pred["label_id"])
        results.append(
            {
                "model_id": model_record["model_id"],
                "instance_id": face_id,
                "feature_instance_id": gt["feature_instance_id"],
                "gt_class": gt["gt_class"],
                "pred_class": pred_class,
                "is_correct": pred_class == gt["gt_class"],
                "area": gt["area"],
                "face_iou": face_iou,
                "vote_ratio": pred["vote_ratio"],
                "score": pred["score"],
            }
        )
    return results


def save_prediction_summary(output_dir, model_record, output, face_predictions):
    pred_dir = os.path.join(output_dir, "pred_details")
    os.makedirs(pred_dir, exist_ok=True)
    payload = {
        "model_id": model_record["model_id"],
        "queries": [
            {"id": i + 1, "score": float(s), "label_id": int(l)}
            for i, (s, l) in enumerate(zip(output["pred_scores"], output["pred_labels"]))
        ],
        "face_predictions": {
            str(face_id): {
                "query_id": int(pred["query_id"]),
                "label_id": int(pred["label_id"]),
                "score": float(pred["score"]),
                "vote_ratio": float(pred["vote_ratio"]),
            }
            for face_id, pred in face_predictions.items()
        },
    }
    path = os.path.join(pred_dir, f"{model_record['model_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MFR multi-view VideoMaskFormer")
    parser.add_argument("--val_dir", default="/data/m2f/temp_data/multiview_feature_dataset/val")
    parser.add_argument("--config_file", default="/data/m2f/Mask2Former/configs/mfr_multiview/video_maskformer2_R50_bs1_14view.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output_dir", default="/data/m2f/temp_data/eval_mfr_multiview")
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--min_size_test", type=int, default=256)
    parser.add_argument("--num_views", type=int, default=14)
    parser.add_argument("--eval_class_mode", choices=["fine", "coarse", "both"], default="both")
    parser.add_argument("--max_models", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = setup_cfg(args)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(args.weights)

    models = load_models(args.val_dir)
    if args.max_models:
        models = models[: args.max_models]

    all_results = []
    with torch.no_grad():
        for model_record in tqdm(models, desc="Evaluating MFR multiview"):
            images, face_id_maps, camera_directions = read_video_inputs(model_record, args.val_dir, cfg.INPUT.FORMAT)
            height, width = images[0].shape[-2:]
            batched_inputs = [
                {
                    "image": images,
                    "face_id_maps": [torch.from_numpy(face_id_map).long() for face_id_map in face_id_maps],
                    "camera_directions": camera_directions,
                    "height": height,
                    "width": width,
                    "model_id": model_record["model_id"],
                }
            ]
            output = model(batched_inputs)
            face_predictions = project_predictions_to_faces(
                output, face_id_maps, args.score_threshold, args.mask_threshold
            )
            save_prediction_summary(args.output_dir, model_record, output, face_predictions)
            all_results.extend(
                evaluate_model(
                    model_record,
                    output,
                    face_id_maps,
                    args.score_threshold,
                    args.mask_threshold,
                )
            )

    saved_paths = []
    if args.eval_class_mode in ("fine", "both"):
        fine_metrics = compute_metrics(all_results, MFR_CLASS_NAMES)
        print_report(fine_metrics, MFR_CLASS_NAMES, "MFR multiview fine face-level report")
        suffix = "fine" if args.eval_class_mode == "both" else ""
        saved_paths.append(save_eval_outputs(args.output_dir, fine_metrics, MFR_CLASS_NAMES, all_results, suffix))

    if args.eval_class_mode in ("coarse", "both"):
        coarse_results = remap_results_for_coarse_eval(all_results, MFR_FINE_TO_COARSE_CLASS)
        coarse_metrics = compute_metrics(coarse_results, MFR_COARSE_CLASS_NAMES)
        print_report(coarse_metrics, MFR_COARSE_CLASS_NAMES, "MFR multiview coarse face-level report")
        suffix = "coarse" if args.eval_class_mode == "both" else ""
        saved_paths.append(save_eval_outputs(args.output_dir, coarse_metrics, MFR_COARSE_CLASS_NAMES, coarse_results, suffix))

    print("\n结果已保存:")
    for metrics_path, csv_path, detail_path in saved_paths:
        print(f"  汇总指标: {metrics_path}")
        print(f"  混淆矩阵: {csv_path}")
        print(f"  详细结果: {detail_path}")
    print(f"  预测详情: {os.path.join(args.output_dir, 'pred_details')}")


if __name__ == "__main__":
    main()
