"""
Convert balanced CAD labels into model-level machining feature instances.

Input layout:
  data/
    labels/<model_id>.json
    steps/<model_id>.step

Each label JSON is expected to contain:
  [[model_id, {"seg": {"face_id": class_id, ...}, "inst": [[0/1, ...], ...]}]]

Output:
  feature_instances.json
  train_models.json
  val_models.json
  split.json

The generated labels are intentionally 3D/model-level:
  feature instance -> class -> face_ids

2D masks for each rendered view can be derived later from a face_id_map:
  mask = np.isin(face_id_map, feature["face_ids"])
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


FEAT_NAMES = [
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
    "stock",
]

IGNORE_ID = 255
STOCK_ID = 24


def load_label(label_path):
    with label_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        raise ValueError("empty label json")

    # Current data stores one item: [model_id, {"seg": ..., "inst": ...}]
    model_id, payload = raw[0]
    seg = payload.get("seg", {})
    inst = payload.get("inst", [])
    if not isinstance(seg, dict) or not isinstance(inst, list):
        raise ValueError("expected payload keys 'seg' dict and 'inst' list")

    face_labels = {int(face_id): int(class_id) for face_id, class_id in seg.items()}
    num_faces = max(face_labels.keys(), default=-1) + 1
    return str(model_id), face_labels, inst, num_faces


def majority_class(face_ids, face_labels, include_stock):
    labels = []
    for face_id in face_ids:
        label = face_labels.get(face_id, IGNORE_ID)
        if label == IGNORE_ID:
            continue
        if not include_stock and label == STOCK_ID:
            continue
        labels.append(label)

    if not labels:
        return None

    counts = Counter(labels)
    # Stable tie break: most common first, then smaller class id.
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def extract_features(face_labels, inst_matrix, num_faces, include_stock=False):
    seen_masks = set()
    features = []

    for row in inst_matrix:
        if len(row) < num_faces:
            row = list(row) + [0] * (num_faces - len(row))
        mask = tuple(1 if int(v) != 0 else 0 for v in row[:num_faces])
        if not any(mask) or mask in seen_masks:
            continue
        seen_masks.add(mask)

        face_ids = [idx for idx, value in enumerate(mask) if value]
        class_id = majority_class(face_ids, face_labels, include_stock=include_stock)
        if class_id is None:
            continue
        if class_id < 0 or class_id >= len(FEAT_NAMES):
            raise ValueError(f"unknown class id {class_id}")

        features.append(
            {
                "instance_id": len(features) + 1,
                "name": f"{FEAT_NAMES[class_id]}_{len(features) + 1:04d}",
                "category_id": class_id,
                "original_category_id": class_id,
                "category_name": FEAT_NAMES[class_id],
                "face_ids": face_ids,
            }
        )

    return features


def build_categories(include_stock):
    max_class = STOCK_ID if include_stock else STOCK_ID - 1
    return [
        {"id": class_id, "original_id": class_id, "name": FEAT_NAMES[class_id]}
        for class_id in range(max_class + 1)
    ]


def convert_dataset(input_dir, output_dir, num_train, num_val, seed, include_stock):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    label_dir = input_dir / "labels"
    step_dir = input_dir / "steps"

    label_paths = sorted(label_dir.glob("*.json"))
    step_paths = {path.stem: path for path in step_dir.glob("*.step")}

    models = []
    skipped_no_step = []
    skipped_bad_label = []

    for label_path in label_paths:
        step_path = step_paths.get(label_path.stem)
        if step_path is None:
            skipped_no_step.append(label_path.stem)
            continue

        try:
            model_id, face_labels, inst_matrix, num_faces = load_label(label_path)
            features = extract_features(
                face_labels,
                inst_matrix,
                num_faces,
                include_stock=include_stock,
            )
        except Exception as exc:
            skipped_bad_label.append({"model_id": label_path.stem, "error": str(exc)})
            continue

        models.append(
            {
                "model_id": model_id,
                "step_path": str(step_path),
                "label_path": str(label_path),
                "num_faces": num_faces,
                "features": features,
            }
        )

    rng = random.Random(seed)
    indices = list(range(len(models)))
    rng.shuffle(indices)
    if num_train is None:
        num_train = max(0, len(models) - num_val)
    total_use = min(len(models), num_train + num_val)
    train_indices = indices[: min(num_train, total_use)]
    val_indices = indices[min(num_train, total_use) : total_use]

    train_ids = [models[i]["model_id"] for i in train_indices]
    val_ids = [models[i]["model_id"] for i in val_indices]

    dataset = {
        "source_dir": str(input_dir),
        "categories": build_categories(include_stock=include_stock),
        "models": models,
        "skipped": {
            "no_step": skipped_no_step,
            "bad_label": skipped_bad_label,
        },
    }
    split = {
        "seed": seed,
        "train": train_ids,
        "val": val_ids,
        "unused": [models[i]["model_id"] for i in indices[total_use:]],
    }

    by_id = {model["model_id"]: model for model in models}
    train_models = [by_id[model_id] for model_id in train_ids]
    val_models = [by_id[model_id] for model_id in val_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_instances.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "split.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "train_models.json").write_text(
        json.dumps({"categories": dataset["categories"], "models": train_models}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "val_models.json").write_text(
        json.dumps({"categories": dataset["categories"], "models": val_models}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    class_counts = Counter()
    feature_count = 0
    for model in models:
        for feature in model["features"]:
            class_counts[feature["category_name"]] += 1
            feature_count += 1

    return {
        "output_dir": str(output_dir),
        "labels": len(label_paths),
        "models": len(models),
        "features": feature_count,
        "train": len(train_models),
        "val": len(val_models),
        "skipped_no_step": len(skipped_no_step),
        "skipped_bad_label": len(skipped_bad_label),
        "class_counts": dict(sorted(class_counts.items())),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/data/m2f/temp_data/data")
    parser.add_argument("--output-dir", default="/data/m2f/temp_data/feature_instance_labels")
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-stock", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
        include_stock=args.include_stock,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
