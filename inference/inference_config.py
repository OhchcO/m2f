# -*- coding: utf-8 -*-
"""Inference-side constants for MFR multi-view recognition.

The Detectron2 YAML still owns the model architecture and NUM_CLASSES.  This
file only owns display names, render encoding constants, and deterministic
colors used by the UI/postprocess code.
"""

NUM_CLASSES = 24

CLASS_NAMES = {
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

TYPE_GAP = 51
TYPE_R_BASE = {
    0: 0,
    1: 51,
    2: 102,
    3: 153,
    4: 204,
}

TYPE_NAMES_EN = {
    0: "Plane",
    1: "Cylinder",
    2: "Cone",
    3: "Sphere",
    4: "Other",
}

TYPE_NAMES_CN = {
    0: "平面",
    1: "圆柱面",
    2: "圆锥面",
    3: "球面",
    4: "其他面",
}

ENCODER_TYPE_GAP = TYPE_GAP
ENCODER_TYPE_R_BASE = TYPE_R_BASE
ENCODER_TYPE_NAMES = TYPE_NAMES_CN

OUTPUT_WIDTH = 512
OUTPUT_HEIGHT = 512


def class_name(class_id):
    return CLASS_NAMES.get(int(class_id), f"class_{int(class_id)}")


def class_color(class_id):
    class_id = int(class_id)
    if class_id < 0:
        return (205, 205, 205)

    # Deterministic high-contrast palette generated from the class id.
    hue = (class_id * 137) % 360
    saturation = 0.72
    value = 0.92
    chroma = value * saturation
    x = chroma * (1 - abs((hue / 60) % 2 - 1))
    m = value - chroma
    if hue < 60:
        r, g, b = chroma, x, 0
    elif hue < 120:
        r, g, b = x, chroma, 0
    elif hue < 180:
        r, g, b = 0, chroma, x
    elif hue < 240:
        r, g, b = 0, x, chroma
    elif hue < 300:
        r, g, b = x, 0, chroma
    else:
        r, g, b = chroma, 0, x
    return (
        int(round((r + m) * 255)),
        int(round((g + m) * 255)),
        int(round((b + m) * 255)),
    )


CLASS_COLORS_3D = {class_id: class_color(class_id) for class_id in CLASS_NAMES}
CLASS_COLORS = {class_id: list(color) for class_id, color in CLASS_COLORS_3D.items()}
