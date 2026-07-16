r"""
面后处理模块：从编码图提取面掩码 + 投票确定分割类别。

R 通道：面几何类型（范围匹配）
G + B 通道：面 ID（G*256 + B）
"""
import os
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from inference_config import (
    CLASS_NAMES,
    ENCODER_TYPE_GAP,
    ENCODER_TYPE_R_BASE,
    ENCODER_TYPE_NAMES,
)


# ---------------------------------------------------------------------------
# R 通道 → 面的几何类型（范围匹配，从 config 生成）
# ---------------------------------------------------------------------------
def _build_r_ranges():
    """构建 R 值范围 → 面类型名称 的映射"""
    ranges = []
    for type_id, r_base in sorted(ENCODER_TYPE_R_BASE.items()):
        r_max = r_base + ENCODER_TYPE_GAP - 1
        ranges.append((r_base, r_max, ENCODER_TYPE_NAMES[type_id]))
    return ranges


_R_RANGES = _build_r_ranges()


def get_face_type_name(r_value):
    """根据 R 值返回面类型名称（范围匹配）"""
    for r_min, r_max, name in _R_RANGES:
        if r_min <= r_value <= r_max:
            return name
    return f"未知({r_value})"


# 兼容旧接口
FACE_TYPE_MAP = {v: ENCODER_TYPE_NAMES[k] for k, v in ENCODER_TYPE_R_BASE.items()}
FACE_TYPE_R_MAP = ENCODER_TYPE_R_BASE


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def mask_to_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


# ---------------------------------------------------------------------------
# 从编码图提取面掩码
#   R   = 面几何类型（20/70/120/170/220）
#   G+B = 面ID
# ---------------------------------------------------------------------------
def extract_faces_from_encoded_image(image_rgb, min_area=10, gb_to_fid=None):
    """
    从编码图 RGB 三通道中提取面信息。

    Args:
        image_rgb: (H, W, 3) numpy uint8 数组
        min_area: 最小面面积（像素），小于此值的忽略
        gb_to_fid: GB 值 → face_id 反向映射（可选）

    Returns:
        face_masks: dict, {face_id: {"mask": bool_array, "face_type": int, "face_type_name": str, "area": int}}
    """
    if isinstance(image_rgb, torch.Tensor):
        arr = image_rgb.cpu().numpy()
    else:
        arr = image_rgb

    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = arr.transpose(1, 2, 0)

    h, w = arr.shape[:2]
    r_channel = arr[:, :, 0].astype(np.uint8)
    g_channel = arr[:, :, 1].astype(np.uint16)
    b_channel = arr[:, :, 2].astype(np.uint16)

    face_id_map = (g_channel << 8) | b_channel

    face_masks = {}
    face_type_counts = defaultdict(int)
    unique_ids = np.unique(face_id_map)

    for face_id_raw in unique_ids:
        face_mask = face_id_map == face_id_raw
        area = int(face_mask.sum())
        if area < min_area:
            continue

        # 该面内 R 通道主导值 = 面几何类型
        face_r_values = r_channel[face_mask]
        r_unique, r_counts = np.unique(face_r_values, return_counts=True)
        dominant_face_type = int(r_unique[np.argmax(r_counts)])

        # 解码 GB 值 → 原始 face_id
        raw_gb = int(face_id_raw)
        if gb_to_fid is not None:
            face_id = gb_to_fid.get(raw_gb, raw_gb)
        else:
            face_id = raw_gb
        face_masks[face_id] = {
            "mask": face_mask,
            "face_type": dominant_face_type,
            "face_type_name": get_face_type_name(dominant_face_type),
            "area": area,
        }
        face_type_counts[dominant_face_type] += 1

    print(f"从编码图提取到 {len(face_masks)} 个面 (min_area={min_area})")
    for ft, cnt in sorted(face_type_counts.items()):
        print(f"  {get_face_type_name(ft)}: {cnt} 个面")
    return face_masks


# ---------------------------------------------------------------------------
# 面后处理：按面边界裁剪 + 投票确定类别
#   分割类别完全由 Mask2Former 决定，R 通道只记录面几何类型
# ---------------------------------------------------------------------------
def postprocess_with_encoded_faces(raw_segmentation, segments_info, face_masks, min_ratio=0.5):
    """
    Args:
        raw_segmentation: Mask2Former 输出的实例 ID 图 (H, W)
        segments_info: Mask2Former 预测的实例信息列表 [{"id": ..., "label_id": ..., "score": ...}]
        face_masks: extract_faces_from_encoded_image 返回的面掩码
        min_ratio: 投票阈值，面内主实例占比低于此值则忽略该面
    """
    info_by_raw_id = {int(s["id"]): s for s in segments_info}
    instance_mask = np.zeros_like(raw_segmentation, dtype=np.uint16)
    class_mask = np.full_like(raw_segmentation, 255, dtype=np.uint8)  # 255=背景
    class_map = {}

    empty_faces = 0
    low_ratio_faces = 0
    new_instance_id = 1

    for face_id, face_info in face_masks.items():
        face_mask_bool = face_info["mask"]
        face_type = face_info["face_type"]
        face_type_name = face_info["face_type_name"]

        face_pixels = raw_segmentation[face_mask_bool]
        face_pixels_nonzero = face_pixels[face_pixels != 0]

        if len(face_pixels_nonzero) == 0:
            empty_faces += 1
            continue

        unique_ids, counts = np.unique(face_pixels_nonzero, return_counts=True)
        max_idx = int(np.argmax(counts))
        raw_instance_id = int(unique_ids[max_idx])
        ratio = float(counts[max_idx] / face_mask_bool.sum())

        if ratio < min_ratio:
            low_ratio_faces += 1
            continue

        segment = info_by_raw_id.get(raw_instance_id)
        if segment is None:
            continue

        pred_class = int(segment["label_id"])
        score = float(segment.get("score", 0.0))

        instance_mask[face_mask_bool] = new_instance_id
        class_mask[face_mask_bool] = pred_class
        class_map[str(new_instance_id)] = {
            "class_id": pred_class,
            "class_name": CLASS_NAMES.get(pred_class, f"class_{pred_class}"),
            "score": score,
            "area": face_info["area"],
            "bbox": mask_to_bbox(face_mask_bool),
            "face_id": int(face_id),
            "face_type": face_type,
            "face_type_name": face_type_name,
            "vote_ratio": round(ratio, 4),
        }
        new_instance_id += 1

    print(f"后处理完成: {len(face_masks)} 个面 → {new_instance_id - 1} 个有效实例")
    print(f"  面内无预测: {empty_faces} 个面")
    print(f"  投票占比不足({min_ratio}): {low_ratio_faces} 个面")
    return instance_mask, class_mask, class_map
