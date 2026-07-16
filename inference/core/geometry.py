"""共享几何工具函数，从 label_tool_instance_study-1.py 和 back_project_to_step.py 提取。"""

import math
from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
)


# ==================== 视角方向 ====================


def get_dodecahedron_view_directions():
    """正十二面体的12个面法向方向（基于黄金比例）。"""
    phi = (1 + math.sqrt(5)) / 2
    length = math.sqrt(1 + phi**2)
    return [
        (0, 1 / length, phi / length),
        (0, -1 / length, phi / length),
        (0, 1 / length, -phi / length),
        (0, -1 / length, -phi / length),
        (1 / length, phi / length, 0),
        (-1 / length, phi / length, 0),
        (1 / length, -phi / length, 0),
        (-1 / length, -phi / length, 0),
        (phi / length, 0, 1 / length),
        (-phi / length, 0, 1 / length),
        (phi / length, 0, -1 / length),
        (-phi / length, 0, -1 / length),
    ]


def get_cube_view_directions():
    """正六面体（立方体）的6个面法向方向：前、后、左、右、上、下"""
    return [
        (0, 0, 1),    # 前 (front)
        (0, 0, -1),   # 后 (back)
        (-1, 0, 0),   # 左 (left)
        (1, 0, 0),    # 右 (right)
        (0, 1, 0),    # 上 (top)
        (0, -1, 0),   # 下 (bottom)
    ]


def get_cube_14_view_directions():
    """正六面体14视角：6个面法向 + 8个顶点对角线方向"""
    face_dirs = get_cube_view_directions()
    # 8个顶点方向：(±1, ±1, ±1) 归一化
    diag = 1.0 / math.sqrt(3)
    vertex_dirs = [
        ( diag,  diag,  diag),   # 右上前
        ( diag,  diag, -diag),   # 右上后
        ( diag, -diag,  diag),   # 右下前
        ( diag, -diag, -diag),   # 右下后
        (-diag,  diag,  diag),   # 左上前
        (-diag,  diag, -diag),   # 左上后
        (-diag, -diag,  diag),   # 左下前
        (-diag, -diag, -diag),   # 左下后
    ]
    return face_dirs + vertex_dirs


def get_24_face_normal_directions():
    """正二十四面体（四方三八面体 / tetrakis hexahedron）的24个面法向方向"""
    # 对偶体为截角八面体，24个面法向 = (0, ±1, ±2) 的所有坐标排列
    dirs = []
    for zero_axis in range(3):
        nonzero_axes = [axis for axis in range(3) if axis != zero_axis]
        for one_axis, two_axis in [nonzero_axes, nonzero_axes[::-1]]:
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    vec = [0.0, 0.0, 0.0]
                    vec[one_axis] = s1 * 1.0
                    vec[two_axis] = s2 * 2.0
                    x, y, z = vec
                    length = math.sqrt(x * x + y * y + z * z)
                    dirs.append((x / length, y / length, z / length))
    return dirs


# ==================== 相机姿态 ====================


def get_viewup(direction):
    """Z轴锁定版本的 up 向量计算。

    当视线方向接近Z轴时返回 (0, 1, 0)，否则通过投影生成合适的 up 向量。
    """
    if abs(direction[2]) > 0.99:
        return (0, 1, 0)
    dot = direction[2]
    vx = -dot * direction[0]
    vy = -dot * direction[1]
    vz = 1 - dot * direction[2]
    length = math.sqrt(vx**2 + vy**2 + vz**2)
    return (vx / length, vy / length, vz / length)


# ==================== 平行投影缩放 ====================


def get_parallel_scale(bounds, center, direction, viewup, aspect_ratio=1.0, margin=1.10):
    """根据模型包围盒计算平行投影的缩放比例。"""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = [
        (xmin, ymin, zmin), (xmin, ymin, zmax),
        (xmin, ymax, zmin), (xmin, ymax, zmax),
        (xmax, ymin, zmin), (xmax, ymin, zmax),
        (xmax, ymax, zmin), (xmax, ymax, zmax),
    ]

    right = (
        direction[1] * viewup[2] - direction[2] * viewup[1],
        direction[2] * viewup[0] - direction[0] * viewup[2],
        direction[0] * viewup[1] - direction[1] * viewup[0],
    )
    right_len = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
    right = (right[0] / right_len, right[1] / right_len, right[2] / right_len)

    max_u = 0.0
    max_v = 0.0
    for corner in corners:
        rel = (
            corner[0] - center[0],
            corner[1] - center[1],
            corner[2] - center[2],
        )
        u = abs(rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2])
        v = abs(rel[0] * viewup[0] + rel[1] * viewup[1] + rel[2] * viewup[2])
        max_u = max(max_u, u)
        max_v = max(max_v, v)

    return max(max_v, max_u / max(aspect_ratio, 1e-6)) * margin


# ==================== 颜色工具 ====================


def rgb_to_float(r, g=None, b=None):
    """将 RGB 整数 (0-255) 转换为浮点元组 (0.0-1.0)。

    支持两种调用方式:
        rgb_to_float(r, g, b)
        rgb_to_float((r, g, b))
    """
    if g is None and b is None and hasattr(r, '__len__') and len(r) == 3:
        r, g, b = r
    return (r / 255.0, g / 255.0, b / 255.0)


# ==================== 表面类型映射 ====================


# GeomAbs 枚举 → 可读表面类型名称
SURFACE_TYPE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_SurfaceOfRevolution: "revolution",
}

# GeomAbs 枚举 → color_encoder TYPE_NAMES (0=Plane,1=Cylinder,2=Cone,3=Sphere,4=Other)
GEOABS_TO_ENCODER_TYPE = {
    GeomAbs_Plane: 0,
    GeomAbs_Cylinder: 1,
    GeomAbs_Cone: 2,
    GeomAbs_Sphere: 3,
}
