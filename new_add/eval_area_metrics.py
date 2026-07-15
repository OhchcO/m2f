"""
面级别面积评估脚本。

直接从 masks/ 和 class_map.json 读取 GT，计算：
1. 面级别分类准确率
2. 面级别面积准确率（按面积加权）
3. 面级别 IoU

GT 数据格式：
- masks/xxx.png: 实例掩码，像素值=实例ID(0,1,2...)，255=背景/忽略
- class_map.json: {图片名: {实例ID(0,1,2...): 类别ID}}

注意：masks 中的实例ID 与 class_map 中的 key 一一对应，二者都从 0 开始。

用法：
python eval_area_metrics.py \
  --val_dir /path/to/balanced_dataset/val \
  --config_file /path/to/config.yaml \
  --opts MODEL.WEIGHTS /path/to/model.pkl

训练/推理使用 24 类，但按手动粗分类评估：
python eval_area_metrics.py \
  --val_dir /path/to/dataset_24class/val \
  --config_file /path/to/config.yaml \
  --weights /path/to/model_final.pth \
  --label_set 24 \
  --eval_class_mode both
"""
import argparse
import csv
import json
import os
import sys

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Mask2Former"))
from mask2former import add_maskformer2_config

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASS_NAMES, NUM_CLASSES

from ins_inference_encoded import setup_cfg, detectron2_to_unified_format

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_NAMES_7 = CLASS_NAMES
CLASS_NAMES_24 = {
    0: "through_hole",
    1: "triangular_passage",
    2: "rectangular_passage",
    3: "6sides_passage",
    4: "triangular_through_slot",
    5: "rectangular_through_slot",
    6: "circular_through_slot",
    7: "rectangular_through_step",
    8: "2sides_through_step",
    9: "slanted_through_step",
    10: "Oring",
    11: "blind_hole",
    12: "triangular_pocket",
    13: "rectangular_pocket",
    14: "6sides_pocket",
    15: "circular_end_pocket",
    16: "rectangular_blind_slot",
    17: "v_circular_end_blind_slot",
    18: "h_circular_end_blind_slot",
    19: "triangular_blind_step",
    20: "circular_blind_step",
    21: "rectangular_blind_step",
    22: "round",
}

# 手动粗分类映射。左侧是模型/GT 的细分类 ID，右侧是评估用粗分类 ID。
# 这里先给出一版默认映射，实际项目里可以直接改这个字典。
MANUAL_COARSE_CLASS_NAMES = {
    0: "hole",
    1: "closed_pocket",
    2: "closed_slot",
    3: "open_pocket",
    4: "open_slot",
    5: "wide_slot",
    6: "oring_slot",
}

MANUAL_FINE_TO_COARSE_CLASS = {
    0: 0,   # through_hole -> hole
    1: 3,   # triangular_passage -> open_pocket
    2: 3,   # rectangular_passage -> open_pocket
    3: 3,   # 6sides_passage -> open_pocket
    4: 4,   # triangular_through_slot -> open_slot
    5: 4,   # rectangular_through_slot -> open_slot
    6: 4,   # circular_through_slot -> open_slot
    7: 5,   # rectangular_through_step -> wide_slot
    8: 5,   # 2sides_through_step -> wide_slot
    9: 5,   # slanted_through_step -> wide_slot
    10: 6,  # Oring -> oring_slot
    11: 0,  # blind_hole -> hole
    12: 1,  # triangular_pocket -> closed_pocket
    13: 1,  # rectangular_pocket -> closed_pocket
    14: 1,  # 6sides_pocket -> closed_pocket
    15: 1,  # circular_end_pocket -> closed_pocket
    16: 2,  # rectangular_blind_slot -> closed_slot
    17: 2,  # v_circular_end_blind_slot -> closed_slot
    18: 2,  # h_circular_end_blind_slot -> closed_slot
    19: 1,  # triangular_blind_step -> closed_pocket
    20: 1,  # circular_blind_step -> closed_pocket
    21: 1,  # rectangular_blind_step -> closed_pocket
    22: 0,  # round -> hole
}

COLOR_PALETTE = [
    np.array([255, 0, 0], dtype=np.uint8),
    np.array([0, 255, 0], dtype=np.uint8),
    np.array([0, 0, 255], dtype=np.uint8),
    np.array([255, 255, 0], dtype=np.uint8),
    np.array([255, 0, 255], dtype=np.uint8),
    np.array([0, 255, 255], dtype=np.uint8),
    np.array([128, 0, 255], dtype=np.uint8),
    np.array([255, 128, 0], dtype=np.uint8),
]


def build_class_colors(class_names):
    return {
        cls_id: COLOR_PALETTE[i % len(COLOR_PALETTE)]
        for i, cls_id in enumerate(class_names.keys())
    }


def colorize_class_mask(class_mask, class_names):
    class_colors = build_class_colors(class_names)
    colored = np.full((*class_mask.shape, 3), 255, dtype=np.uint8)
    for class_id, color in class_colors.items():
        colored[class_mask == class_id] = color
    return colored


def build_legend_patches(class_names):
    class_colors = build_class_colors(class_names)
    patches = []
    for class_id, class_name in class_names.items():
        color = class_colors[class_id] / 255.0
        patches.append(mpatches.Patch(color=color, label=f"{class_id}={class_name}"))
    return patches


def build_gt_class_mask(gt_mask, class_map):
    """从GT实例掩码和class_map构建类别掩码。"""
    class_mask = np.full_like(gt_mask, 255, dtype=np.uint8)
    instance_ids = np.unique(gt_mask)
    instance_ids = instance_ids[(instance_ids >= 0) & (instance_ids < 255)]
    for inst_id in instance_ids:
        seq_num = str(int(inst_id))
        if seq_num in class_map:
            class_mask[gt_mask == inst_id] = class_map[seq_num]
    return class_mask


def build_pred_class_mask(raw_segmentation, segments_info):
    """从预测结果构建像素级类别掩码（投票前）。"""
    class_mask = np.full_like(raw_segmentation, 255, dtype=np.uint8)
    for seg in segments_info:
        inst_id = seg.get("id", 0)
        cls_id = seg.get("label_id", 0)
        class_mask[raw_segmentation == inst_id] = cls_id
    return class_mask


def build_voted_pred_mask(gt_mask, raw_segmentation, segments_info):
    """
    按GT面边界投票，生成投票后的预测类别掩码。
    每个GT面内的像素统一赋值为投票选出的预测类别。
    """
    info_by_id = {int(s["id"]): s for s in segments_info}
    voted_mask = np.full_like(gt_mask, 255, dtype=np.uint8)
    instance_ids = np.unique(gt_mask)
    instance_ids = instance_ids[(instance_ids >= 0) & (instance_ids < 255)]

    for inst_id in instance_ids:
        face_mask = (gt_mask == inst_id)
        face_pred_ids = raw_segmentation[face_mask]
        face_pred_nonzero = face_pred_ids[face_pred_ids != 0]

        if len(face_pred_nonzero) == 0:
            continue

        unique_ids, counts = np.unique(face_pred_nonzero, return_counts=True)
        max_idx = int(np.argmax(counts))
        pred_inst_id = int(unique_ids[max_idx])

        seg = info_by_id.get(pred_inst_id, {})
        pred_class = seg.get("label_id", -1)
        voted_mask[face_mask] = pred_class

    return voted_mask


def visualize_comparison(image_rgb, gt_class_mask, raw_pred_class_mask,
                         voted_pred_class_mask, img_file, output_dir, class_names):
    """生成4列对比可视化：原图 | GT | Raw Pred | Voted Pred + Overlay。"""
    class_colors = build_class_colors(class_names)
    gt_color = colorize_class_mask(gt_class_mask, class_names)
    raw_color = colorize_class_mask(raw_pred_class_mask, class_names)
    voted_color = colorize_class_mask(voted_pred_class_mask, class_names)
    legend_patches = build_legend_patches(class_names)

    # Overlay: 原图转灰底，投票后的特征按类别染色叠加
    gray = np.mean(image_rgb, axis=2).astype(np.uint8)
    overlay = np.stack([gray, gray, gray], axis=2)
    # 预测出的特征区域按类别染色
    pred_valid = voted_pred_class_mask != 255
    for cls_id, color in class_colors.items():
        cls_mask = pred_valid & (voted_pred_class_mask == cls_id)
        overlay[cls_mask] = color

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt_color)
    axes[1].set_title("GT")
    axes[1].axis("off")
    axes[1].legend(handles=legend_patches, loc="upper right", fontsize=7)

    axes[2].imshow(raw_color)
    axes[2].set_title("Raw Pred (pixel)")
    axes[2].axis("off")
    axes[2].legend(handles=legend_patches, loc="upper right", fontsize=7)

    axes[3].imshow(voted_color)
    axes[3].set_title("After Voting (face)")
    axes[3].axis("off")
    axes[3].legend(handles=legend_patches, loc="upper right", fontsize=7)

    axes[4].imshow(overlay)
    axes[4].set_title("Features Overlay")
    axes[4].axis("off")
    overlay_legend = [mpatches.Patch(color=[0.63, 0.63, 0.63], label="Model")]
    for cls_id, cls_name in class_names.items():
        color = class_colors[cls_id] / 255.0
        overlay_legend.append(mpatches.Patch(color=color, label=cls_name))
    axes[4].legend(handles=overlay_legend, loc="upper right", fontsize=6)

    plt.suptitle(img_file, fontsize=12)
    plt.tight_layout()
    vis_path = os.path.join(output_dir, img_file.rsplit(".", 1)[0] + "_vis.png")
    plt.savefig(vis_path, dpi=100, bbox_inches="tight")
    plt.close()
    return vis_path


def build_gt_from_mask_and_classmap(mask, class_map):
    """
    从 masks 和 class_map.json 构建 GT 面信息。

    Args:
        mask: 实例掩码 (H, W), 像素值=实例ID(0,1,2...), 255=背景/忽略
        class_map: {实例ID(字符串): 类别ID}

    Returns:
        gt_faces: list of dict
    """
    gt_faces = []
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[(instance_ids >= 0) & (instance_ids < 255)]

    for inst_id in instance_ids:
        inst_id_int = int(inst_id)
        seq_num = str(inst_id_int)

        if seq_num not in class_map:
            continue

        category_id = class_map[seq_num]
        instance_mask = (mask == inst_id_int)
        area = int(instance_mask.sum())

        if area < 10:
            continue

        gt_faces.append({
            "instance_id": inst_id_int,
            "category_id": category_id,
            "mask": instance_mask,
            "area": area,
        })

    return gt_faces


def evaluate_single(gt_faces, pred_mask, pred_classes):
    """
    评估单张图的面级别结果。

    Args:
        gt_faces: list of dict, GT 面信息
        pred_mask: (H, W) 预测实例ID图 (0=背景)
        pred_classes: list of {"id": instance_id, "class": class_id, "score": score}

    Returns:
        results: list of dict
    """
    pred_info = {p["id"]: p for p in pred_classes}

    results = []

    for gt in gt_faces:
        gt_mask = gt["mask"]
        gt_class = gt["category_id"]
        gt_area = gt["area"]

        # 在该面区域内，统计各预测实例的像素数
        face_pred_ids = pred_mask[gt_mask]
        face_pred_nonzero = face_pred_ids[face_pred_ids != 0]

        if len(face_pred_nonzero) == 0:
            # 面内无预测
            results.append({
                "instance_id": gt["instance_id"],
                "gt_class": gt_class,
                "pred_class": -1,
                "is_correct": False,
                "area": gt_area,
                "face_iou": 0.0,
                "vote_ratio": 0.0,
                "score": 0.0,
            })
            continue

        # 投票：找出占比最大的预测实例
        unique_ids, counts = np.unique(face_pred_nonzero, return_counts=True)
        max_idx = int(np.argmax(counts))
        pred_id = int(unique_ids[max_idx])
        vote_ratio = float(counts[max_idx] / gt_area)

        # DEBUG: 打印调试信息（只打印第一张图）
        if not hasattr(evaluate_single, '_debug_done'):
            print(f"\n[DEBUG] pred_info keys: {list(pred_info.keys())[:5]}...")
            print(f"[DEBUG] pred_id={pred_id}, in pred_info: {pred_id in pred_info}")
            print(f"[DEBUG] pred_info_item: {pred_info.get(pred_id, 'NOT FOUND')}")
            evaluate_single._debug_done = True

        pred_info_item = pred_info.get(pred_id, {})
        pred_class = pred_info_item.get("label_id", -1)
        score = pred_info_item.get("score", 0.0)

        is_correct = (pred_class == gt_class)

        # 计算面级别的 IoU
        if pred_id > 0:
            pred_bin = (pred_mask == pred_id)
            intersection = int(np.sum(gt_mask & pred_bin))
            union = int(np.sum(gt_mask | pred_bin))
            face_iou = intersection / union if union > 0 else 0.0
        else:
            face_iou = 0.0

        results.append({
            "instance_id": gt["instance_id"],
            "gt_class": gt_class,
            "pred_class": pred_class,
            "is_correct": is_correct,
            "area": gt_area,
            "face_iou": face_iou,
            "vote_ratio": vote_ratio,
            "score": score,
        })

    return results


def infer_fine_class_names(cfg, label_set):
    if label_set == "7":
        return CLASS_NAMES_7
    if label_set == "24":
        return CLASS_NAMES_24

    num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    if num_classes == 23:
        return CLASS_NAMES_24
    if num_classes == 7:
        return CLASS_NAMES_7

    return {i: f"class_{i}" for i in range(num_classes)}


def remap_results_for_coarse_eval(all_results, class_mapping):
    remapped = []
    for result in all_results:
        gt_class = class_mapping.get(result["gt_class"])
        if gt_class is None:
            continue

        pred_class = class_mapping.get(result["pred_class"], -1)
        mapped = result.copy()
        mapped["raw_gt_class"] = result["gt_class"]
        mapped["raw_pred_class"] = result["pred_class"]
        mapped["gt_class"] = gt_class
        mapped["pred_class"] = pred_class
        mapped["is_correct"] = pred_class == gt_class
        remapped.append(mapped)

    return remapped


def compute_metrics(all_results, class_names):
    """汇总所有图片的评估结果。"""
    num_classes = len(class_names)
    total_faces = len(all_results)
    correct_faces = sum(1 for r in all_results if r["is_correct"])
    face_acc = correct_faces / total_faces if total_faces > 0 else 0.0

    total_area = sum(r["area"] for r in all_results)
    correct_area = sum(r["area"] for r in all_results if r["is_correct"])
    area_acc = correct_area / total_area if total_area > 0 else 0.0

    face_ious = [r["face_iou"] for r in all_results]
    mean_face_iou = np.mean(face_ious) if face_ious else 0.0

    # 各类别指标
    class_metrics = {}
    for cls_id in range(num_classes):
        cls_results = [r for r in all_results if r["gt_class"] == cls_id]
        cls_correct = [r for r in cls_results if r["is_correct"]]
        cls_area = sum(r["area"] for r in cls_results)
        cls_correct_area = sum(r["area"] for r in cls_correct)

        class_metrics[cls_id] = {
            "name": class_names.get(cls_id, f"class_{cls_id}"),
            "total_faces": len(cls_results),
            "correct_faces": len(cls_correct),
            "accuracy": len(cls_correct) / len(cls_results) if cls_results else 0.0,
            "total_area": cls_area,
            "correct_area": cls_correct_area,
            "area_accuracy": cls_correct_area / cls_area if cls_area > 0 else 0.0,
            "mean_iou": np.mean([r["face_iou"] for r in cls_results]) if cls_results else 0.0,
        }

    # 混淆矩阵
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for r in all_results:
        gt = r["gt_class"]
        pred = r["pred_class"]
        if 0 <= gt < num_classes and 0 <= pred < num_classes:
            confusion[gt][pred] += 1

    return {
        "total_faces": total_faces,
        "correct_faces": correct_faces,
        "face_accuracy": face_acc,
        "total_area": total_area,
        "correct_area": correct_area,
        "area_accuracy": area_acc,
        "mean_face_iou": float(mean_face_iou),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion.tolist(),
    }


def print_report(metrics, class_names, title="面级别面积评估报告"):
    """打印评估报告。"""
    num_classes = len(class_names)
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"\n总体指标:")
    print(f"  总面数:          {metrics['total_faces']}")
    print(f"  正确面数:        {metrics['correct_faces']}")
    print(f"  面级别准确率:    {metrics['face_accuracy']:.4f} ({metrics['face_accuracy']*100:.2f}%)")
    print(f"  总面积:          {metrics['total_area']}")
    print(f"  面积加权准确率:  {metrics['area_accuracy']:.4f} ({metrics['area_accuracy']*100:.2f}%)")
    print(f"  平均面 IoU:      {metrics['mean_face_iou']:.4f}")

    print(f"\n{'类别':<12} {'面数':>6} {'准确率':>10} {'面积':>10} {'面积准确率':>12} {'平均IoU':>10}")
    print("-" * 70)
    for cls_id in range(num_classes):
        cm = metrics["class_metrics"][cls_id]
        print(f"  {cm['name']:<10} {cm['total_faces']:>6} {cm['accuracy']:>10.4f} "
              f"{cm['total_area']:>10} {cm['area_accuracy']:>12.4f} {cm['mean_iou']:>10.4f}")

    # 混淆矩阵
    print(f"\n混淆矩阵 (行=GT, 列=Pred):")
    header = "".join(f"{class_names.get(i, f'C{i}'):>8}" for i in range(num_classes))
    print(f"{'':>12}{header}")
    for gt_cls in range(num_classes):
        row = "".join(f"{metrics['confusion_matrix'][gt_cls][p]:>8}" for p in range(num_classes))
        print(f"  {class_names.get(gt_cls, f'C{gt_cls}'):<10}{row}")

    print("\n" + "=" * 70)


def save_eval_outputs(output_dir, metrics, class_names, all_results, suffix=""):
    suffix_part = f"_{suffix}" if suffix else ""

    metrics_path = os.path.join(output_dir, f"eval_results{suffix_part}.json")
    save_metrics = {
        "total_faces": metrics["total_faces"],
        "correct_faces": metrics["correct_faces"],
        "face_accuracy": metrics["face_accuracy"],
        "total_area": metrics["total_area"],
        "correct_area": metrics["correct_area"],
        "area_accuracy": metrics["area_accuracy"],
        "mean_face_iou": metrics["mean_face_iou"],
        "class_metrics": {str(k): v for k, v in metrics["class_metrics"].items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(save_metrics, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, f"confusion_matrix{suffix_part}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["GT\\Pred"] + [class_names.get(i, f"C{i}") for i in range(len(class_names))]
        writer.writerow(header)
        for gt_cls in range(len(class_names)):
            row = [class_names.get(gt_cls, f"C{gt_cls}")]
            row += [metrics["confusion_matrix"][gt_cls][p] for p in range(len(class_names))]
            writer.writerow(row)

    detail_path = os.path.join(output_dir, f"eval_details{suffix_part}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    return metrics_path, csv_path, detail_path


def main():
    parser = argparse.ArgumentParser(description="面级别面积评估")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="验证集目录 (balanced_dataset/val/)")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Mask2Former detectron2 配置文件")
    parser.add_argument("--output_dir", type=str, default="/data/project_m2f/temp_data/eval_output",
                        help="评估结果输出目录")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="实例置信度阈值")
    parser.add_argument("--weights", type=str, default=None,
                        help="模型权重路径 (覆盖配置文件中的 MODEL.WEIGHTS)")
    parser.add_argument("--label_set", choices=["auto", "7", "24"], default="auto",
                        help="细分类标签集。auto 会根据配置中的 NUM_CLASSES 判断")
    parser.add_argument("--eval_class_mode", choices=["fine", "coarse", "both"], default="fine",
                        help="fine=按模型原始类别评估，coarse=按手动映射后类别评估，both=同时输出")

    args, unknown = parser.parse_known_args()

    # 读取 class_map.json
    class_map_path = os.path.join(args.val_dir, "class_map.json")
    print(f"加载 class_map: {class_map_path}")
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    # class_map 格式: {图片名: {序号: 类别ID}}
    # 需要展平为 {(图片名, 序号): 类别ID}
    flat_class_map = {}
    for img_name, instances in class_map.items():
        for seq_str, cat_id in instances.items():
            flat_class_map[(img_name, seq_str)] = cat_id

    print(f"  图片数: {len(class_map)}")

    # 配置模型
    cfg = setup_cfg(args.config_file)
    if args.weights:
        cfg.defrost()
        cfg.MODEL.WEIGHTS = args.weights
        cfg.freeze()
    fine_class_names = infer_fine_class_names(cfg, args.label_set)
    coarse_class_names = MANUAL_COARSE_CLASS_NAMES
    fine_to_coarse = MANUAL_FINE_TO_COARSE_CLASS

    print(f"模型权重: {cfg.MODEL.WEIGHTS}")
    print(f"细分类类别数: {len(fine_class_names)}")
    if args.eval_class_mode in ("coarse", "both"):
        print(f"粗分类类别数: {len(coarse_class_names)}")
        print(f"粗分类映射: {fine_to_coarse}")
    predictor = DefaultPredictor(cfg)

    # 图像和掩码目录
    image_dir = os.path.join(args.val_dir, "encoded_views")
    mask_dir = os.path.join(args.val_dir, "masks")

    # 收集验证集图片
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if len(image_files) == 0:
        print(f"[ERROR] 未找到图片: {image_dir}")
        return

    print(f"验证集图片数: {len(image_files)}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建推理结果保存目录
    pred_mask_dir = os.path.join(args.output_dir, "pred_masks")
    pred_detail_dir = os.path.join(args.output_dir, "pred_details")
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(pred_mask_dir, exist_ok=True)
    os.makedirs(pred_detail_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 逐图评估
    all_results = []

    for img_file in tqdm(image_files, desc="评估进度"):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)

        # 读取图像
        image_np = cv2.imread(img_path)
        if image_np is None:
            continue

        # 读取 GT 掩码
        if not os.path.exists(mask_path):
            continue
        gt_mask = np.array(Image.open(mask_path))

        # 获取该图的类别映射
        if img_file not in class_map:
            continue
        img_class_map = class_map[img_file]

        # 构建 GT 面信息
        gt_faces = build_gt_from_mask_and_classmap(gt_mask, img_class_map)
        if len(gt_faces) == 0:
            continue

        # Mask2Former 推理
        outputs = predictor(image_np)
        raw_segmentation, segments_info = detectron2_to_unified_format(outputs, args.threshold)

        # 保存预测掩码图 (每个实例用不同颜色)
        pred_mask_vis = np.zeros((*raw_segmentation.shape, 3), dtype=np.uint8)
        for seg in segments_info:
            cls_id = seg.get("label_id", 0)
            inst_id = seg.get("id", 0)
            mask_bin = (raw_segmentation == inst_id)
            # 用类别ID生成颜色
            color = [(cls_id * 40 + 80) % 256, (cls_id * 70 + 40) % 256, (cls_id * 110 + 120) % 256]
            pred_mask_vis[mask_bin] = color
        pred_mask_path = os.path.join(pred_mask_dir, img_file)
        cv2.imwrite(pred_mask_path, pred_mask_vis)

        # 保存该图的推理详情
        pred_detail = {
            "image": img_file,
            "segments": [
                {
                    "id": seg.get("id"),
                    "label_id": seg.get("label_id"),
                    "label_name": fine_class_names.get(seg.get("label_id", -1), "unknown"),
                    "score": round(seg.get("score", 0), 4),
                    "area": int((raw_segmentation == seg.get("id", 0)).sum()),
                }
                for seg in segments_info
            ],
        }
        detail_path = os.path.join(pred_detail_dir, img_file.rsplit(".", 1)[0] + ".json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(pred_detail, f, ensure_ascii=False, indent=2)

        # 评估
        results = evaluate_single(gt_faces, raw_segmentation, segments_info)

        # 生成可视化：原图 | GT | Raw Pred | Voted Pred | Overlay
        gt_class_mask = build_gt_class_mask(gt_mask, img_class_map)
        raw_pred_class_mask = build_pred_class_mask(raw_segmentation, segments_info)
        voted_pred_class_mask = build_voted_pred_mask(gt_mask, raw_segmentation, segments_info)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        visualize_comparison(image_rgb, gt_class_mask, raw_pred_class_mask,
                             voted_pred_class_mask, img_file, vis_dir, fine_class_names)

        # 记录图片名
        for r in results:
            r["image"] = img_file

        all_results.extend(results)

    # 汇总和保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    saved_paths = []
    if args.eval_class_mode in ("fine", "both"):
        fine_metrics = compute_metrics(all_results, fine_class_names)
        print_report(fine_metrics, fine_class_names, "细分类面级别面积评估报告")
        suffix = "fine" if args.eval_class_mode == "both" else ""
        saved_paths.append(save_eval_outputs(
            args.output_dir, fine_metrics, fine_class_names, all_results, suffix
        ))

    if args.eval_class_mode in ("coarse", "both"):
        coarse_results = remap_results_for_coarse_eval(all_results, fine_to_coarse)
        coarse_metrics = compute_metrics(coarse_results, coarse_class_names)
        print_report(coarse_metrics, coarse_class_names, "粗分类面级别面积评估报告")
        suffix = "coarse" if args.eval_class_mode == "both" else ""
        saved_paths.append(save_eval_outputs(
            args.output_dir, coarse_metrics, coarse_class_names, coarse_results, suffix
        ))

    print(f"\n结果已保存:")
    for metrics_path, csv_path, detail_path in saved_paths:
        print(f"  汇总指标: {metrics_path}")
        print(f"  混淆矩阵: {csv_path}")
        print(f"  详细结果: {detail_path}")
    print(f"  预测掩码: {pred_mask_dir}")
    print(f"  推理详情: {pred_detail_dir}")
    print(f"  可视化图: {vis_dir}")


if __name__ == "__main__":
    main()
