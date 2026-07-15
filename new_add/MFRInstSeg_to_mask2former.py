"""
data/ -> multi-view feature-instance dataset.

Input:
  data/
    labels/<model_id>.json
    steps/<model_id>.step

Output:
  output/{train,val}/
    encoded_views/     14-view RGB inputs
    face_id_maps/      per-pixel face id maps, saved as .npy; background is -1
    camera_views.json  camera/view metadata keyed by image name
    models.json        model-level labels + 14 view records

Masks are not saved here. During training, generate each feature mask with:
  mask = np.isin(face_id_map, feature["face_ids"])
"""
import os
import sys
import json
import math
import time
import tempfile
import multiprocessing
import traceback
import shutil
import argparse
import posixpath
from collections import Counter
import numpy as np

# VTK/numpy 兼容性: numpy 1.24+ 移除了 np.bool 等别名，但旧版 VTK 仍依赖
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float_
if not hasattr(np, 'complex'):
    np.complex = np.complex_
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'str'):
    np.str = np.str_

from pathlib import Path

import cv2
import pyvista as pv
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.TopLoc import TopLoc_Location

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 配置
# ============================================================
INPUT_DIR = "/data/m2f/temp_data/data"
OUTPUT_DIR = "/data/m2f/temp_data/multiview_feature_dataset"
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 786
RANDOM_SEED = 42
NUM_TRAIN = 190
NUM_VAL = 10
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# 24 machining classes + stock/background.
IGNORE_ID = 255
STOCK_ID = 24
IMAGE_EXT = (".png", ".jpg", ".bmp")
FACE_ID_BACKGROUND = -1

FEAT_NAMES = [
    'chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
    'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
    'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring', 'blind_hole',
    'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
    'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
    'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round', 'stock'
]


# ============================================================
# 工具函数
# ============================================================
def hsv_to_rgb(h, s, v):
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:    r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def compute_unique_colors(num_faces):
    if num_faces == 0:
        return []
    golden_angle = 137.508
    return [hsv_to_rgb(i * golden_angle, 0.85, 0.95) for i in range(num_faces)]


def face_id_to_rgb(face_id):
    """Encode face_id as RGB. Uses face_id + 1 so black can remain background if needed."""
    encoded = face_id + 1
    if encoded <= 0 or encoded > 0xFFFFFF:
        raise ValueError(f"face_id out of encodable range: {face_id}")
    r = (encoded >> 16) & 255
    g = (encoded >> 8) & 255
    b = encoded & 255
    return r, g, b


def decode_face_id_image(rgb_image):
    """Decode RGB face-id render into a dense int32 map. White background becomes -1."""
    if rgb_image.shape[-1] > 3:
        rgb_image = rgb_image[:, :, :3]
    rgb = rgb_image.astype(np.int32)
    encoded = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
    face_id_map = encoded - 1

    # The script renders with a white background. It decodes to 0xFFFFFF - 1,
    # so explicitly mark it as background.
    white_bg = (rgb[:, :, 0] == 255) & (rgb[:, :, 1] == 255) & (rgb[:, :, 2] == 255)
    face_id_map[white_bg] = FACE_ID_BACKGROUND

    return face_id_map.astype(np.int32)


def majority_class(face_ids, face_labels, include_stock=False):
    labels = []
    for face_id in face_ids:
        if face_id >= len(face_labels):
            continue
        label = int(face_labels[face_id])
        if label == IGNORE_ID:
            continue
        if not include_stock and label == STOCK_ID:
            continue
        labels.append(label)
    if not labels:
        return None
    return sorted(Counter(labels).items(), key=lambda item: (-item[1], item[0]))[0][0]


def extract_feature_instances(face_labels, inst_matrix, include_stock=False):
    """Convert repeated 0/1 inst rows into unique model-level feature instances."""
    num_faces = len(face_labels)
    seen = set()
    features = []
    for row in inst_matrix:
        if len(row) < num_faces:
            row = list(row) + [0] * (num_faces - len(row))
        mask = tuple(1 if int(v) != 0 else 0 for v in row[:num_faces])
        if not any(mask) or mask in seen:
            continue
        seen.add(mask)

        face_ids = [idx for idx, value in enumerate(mask) if value]
        class_id = majority_class(face_ids, face_labels, include_stock=include_stock)
        if class_id is None:
            continue
        if class_id < 0 or class_id >= len(FEAT_NAMES):
            raise ValueError(f"unknown class id {class_id}")

        features.append({
            "instance_id": len(features) + 1,
            "name": f"{FEAT_NAMES[class_id]}_{len(features) + 1:04d}",
            "category_id": int(class_id),
            "original_category_id": int(class_id),
            "category_name": FEAT_NAMES[class_id],
            "face_ids": face_ids,
        })
    return features


def get_viewup(direction):
    if abs(direction[2]) > 0.99:
        return (0, 1, 0)
    dot = direction[2]
    vx = -dot * direction[0]
    vy = -dot * direction[1]
    vz = 1 - dot * direction[2]
    length = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return (vx / length, vy / length, vz / length)


def get_parallel_scale(bounds, center, direction, viewup, aspect_ratio=1.0, margin=1.10):
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
    right_len = math.sqrt(right[0] ** 2 + right[1] ** 2 + right[2] ** 2)
    right = (right[0] / right_len, right[1] / right_len, right[2] / right_len)
    max_u = 0.0
    max_v = 0.0
    for corner in corners:
        rel = (corner[0] - center[0], corner[1] - center[1], corner[2] - center[2])
        u = abs(rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2])
        v = abs(rel[0] * viewup[0] + rel[1] * viewup[1] + rel[2] * viewup[2])
        max_u = max(max_u, u)
        max_v = max(max_v, v)
    return max(max_v, max_u / max(aspect_ratio, 1e-6)) * margin


def safe_screenshot(plotter, filepath):
    d = os.path.dirname(filepath)
    if d:
        os.makedirs(d, exist_ok=True)
    plotter.screenshot(filepath)


def get_cube_14_view_directions():
    face_dirs = [
        (0, 0, 1), (0, 0, -1), (-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0),
    ]
    diag = 1.0 / math.sqrt(3)
    vertex_dirs = [
        (diag, diag, diag), (diag, diag, -diag), (diag, -diag, diag), (diag, -diag, -diag),
        (-diag, diag, diag), (-diag, diag, -diag), (-diag, -diag, diag), (-diag, -diag, -diag),
    ]
    return face_dirs + vertex_dirs


# ============================================================
# 单模型处理
# ============================================================
def process_single_model(label_path, step_path, face_labels, features, out_dir):
    model_name = os.path.splitext(os.path.basename(label_path))[0]

    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        print(f"  [跳过] {model_name}: 无法读取 STEP")
        return None
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.1)

    ocp_faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None and tri.NbTriangles() > 0:
            ocp_faces.append(face)
        explorer.Next()

    if len(ocp_faces) != len(face_labels):
        print(f"  [跳过] {model_name}: 面数不匹配 ({len(ocp_faces)} vs {len(face_labels)})")
        return None

    num_faces = len(ocp_faces)

    writer = StlAPI_Writer()
    face_meshes = []
    for face in ocp_faces:
        fd_f, temp_stl_f = tempfile.mkstemp(suffix=".stl")
        os.close(fd_f)
        writer.Write(face, temp_stl_f)
        pv_mesh = pv.read(temp_stl_f)
        face_meshes.append(pv_mesh)
        os.remove(temp_stl_f)

    unique_colors = compute_unique_colors(num_faces)

    encoded_render_items = []
    face_id_render_items = []
    for fi, mesh in enumerate(face_meshes):
        r, g, b = unique_colors[fi]
        encoded_render_items.append((mesh, (r / 255.0, g / 255.0, b / 255.0)))
        ir, ig, ib = face_id_to_rgb(fi)
        face_id_render_items.append((mesh, (ir / 255.0, ig / 255.0, ib / 255.0)))

    all_pts = np.vstack([m.points for m in face_meshes])
    bmin, bmax = all_pts.min(axis=0), all_pts.max(axis=0)
    bounds = (bmin[0], bmax[0], bmin[1], bmax[1], bmin[2], bmax[2])
    center = ((bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2)
    max_dim = max(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
    dist = max_dim * 3 if max_dim > 0.001 else 3.0

    directions = get_cube_14_view_directions()
    num_views = len(directions)

    encoded_dir = os.path.join(out_dir, "encoded_views")
    face_map_dir = os.path.join(out_dir, "face_id_maps")
    os.makedirs(encoded_dir, exist_ok=True)
    os.makedirs(face_map_dir, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))
    plotter.disable_anti_aliasing()
    plotter.camera.SetParallelProjection(True)
    plotter.add_mesh(face_meshes[0], color='white')
    plotter.render()
    _ = plotter.screenshot(None)
    plotter.clear()

    camera_records = {}
    view_records = []
    aspect_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT

    for view_id, direction in enumerate(directions):
        viewup = get_viewup(direction)
        cam_pos = (
            center[0] + direction[0] * dist,
            center[1] + direction[1] * dist,
            center[2] + direction[2] * dist,
        )
        parallel_scale = get_parallel_scale(bounds, center, direction, viewup, aspect_ratio, margin=1.10)
        img_name = f"{model_name}_{view_id:06d}.png"
        face_map_name = f"{model_name}_{view_id:06d}.npy"

        # Unique 染色图
        plotter.clear()
        plotter.set_background('white')
        for mesh, color in encoded_render_items:
            plotter.add_mesh(mesh, color=color, smooth_shading=False, lighting=False)
        plotter.camera_position = [cam_pos, center, viewup]
        plotter.camera.SetParallelScale(max(parallel_scale, 0.01))
        plotter.render()
        unique_path = os.path.join(encoded_dir, img_name)
        safe_screenshot(plotter, unique_path)

        # Face-id map render. This is not a training target; it is the bridge
        # from 2D pixels back to STEP face ids.
        plotter.clear()
        plotter.set_background('white')
        for mesh, color in face_id_render_items:
            plotter.add_mesh(mesh, color=color, smooth_shading=False, lighting=False)
        plotter.camera_position = [cam_pos, center, viewup]
        plotter.camera.SetParallelScale(max(parallel_scale, 0.01))
        plotter.render()
        face_id_rgb = plotter.screenshot(None)
        face_id_map = decode_face_id_image(face_id_rgb)
        face_id_map[(face_id_map < 0) | (face_id_map >= num_faces)] = FACE_ID_BACKGROUND
        face_map_path = os.path.join(face_map_dir, face_map_name)
        np.save(face_map_path, face_id_map)

        camera = {
            "direction": [round(v, 6) for v in direction],
            "camera_position": [round(v, 4) for v in cam_pos],
            "focal_point": [round(v, 4) for v in center],
            "viewup": [round(v, 4) for v in viewup],
            "parallel_scale": round(float(max(parallel_scale, 0.01)), 4),
        }
        camera_records[img_name] = camera
        view_records.append({
            "view_id": view_id,
            "image": posixpath.join("encoded_views", img_name),
            "face_id_map": posixpath.join("face_id_maps", face_map_name),
            "camera": camera,
        })

    plotter.close()
    return {
        "model_id": model_name,
        "step_path": step_path,
        "label_path": label_path,
        "num_faces": num_faces,
        "features": features,
        "views": view_records,
    }, camera_records


# ============================================================
# 多进程 worker
# ============================================================
def _worker(args):
    label_path, step_path, face_labels, features, out_dir, idx, total = args
    t0 = time.time()
    try:
        result = process_single_model(label_path, step_path, face_labels, features, out_dir)
        elapsed = time.time() - t0
        model_name = os.path.splitext(os.path.basename(label_path))[0]
        return (True, model_name, result, elapsed, idx, total)
    except Exception:
        elapsed = time.time() - t0
        model_name = os.path.splitext(os.path.basename(label_path))[0]
        traceback.print_exc()
        return (False, model_name, None, elapsed, idx, total)


def make_progress(total, desc):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="model", dynamic_ncols=True)


def update_progress(progress, model_name, elapsed, completed, failed, total):
    processed = completed + failed
    if tqdm is None:
        eta = elapsed / processed * (total - processed) if processed > 0 else 0
        print(f"  [{processed}/{total}] {model_name} {elapsed:.1f}s, ETA {eta:.0f}s")
        return

    if progress is not None:
        progress.update(1)
        progress.set_postfix({
            "ok": completed,
            "fail": failed,
            "last": model_name,
            "sec": f"{elapsed:.1f}",
        })


# ============================================================
# 主流程
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-train", type=int, default=NUM_TRAIN)
    parser.add_argument("--num-val", type=int, default=NUM_VAL)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--include-stock", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("data/ -> 14-view feature-instance dataset")
    print(f"输入: {args.input_dir}")
    print(f"输出: {args.output_dir}")
    print(f"划分: train={args.num_train}, val={args.num_val}")
    print("=" * 60)

    # ---- 读取数据目录 ----
    label_dir = os.path.join(args.input_dir, "labels")
    step_dir = os.path.join(args.input_dir, "steps")
    label_files = sorted(Path(label_dir).glob("*.json"))
    step_files = {p.stem: str(p) for p in Path(step_dir).glob("*.step")}
    print(f"标签文件: {len(label_files)}, STEP 文件: {len(step_files)}")

    # ---- 预处理: 读取标签、过滤 ----
    valid_models = []
    skipped_no_step = 0
    skipped_face_mismatch = 0

    for lf in label_files:
        model_name = lf.stem
        step_path = step_files.get(model_name)
        if step_path is None:
            skipped_no_step += 1
            continue

        with open(lf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            continue

        item = data[0]
        seg = item[1].get("seg", {})
        inst = item[1].get("inst", [])

        num_faces = len(seg)

        # 7类 (0-6)，255=ignore，直接使用
        face_labels = []
        for fi in range(num_faces):
            lbl = seg.get(str(fi), IGNORE_ID)
            face_labels.append(int(lbl))

        features = extract_feature_instances(face_labels, inst, include_stock=args.include_stock)

        valid_models.append((str(lf), step_path, face_labels, features))

    print(f"有效模型: {len(valid_models)}")
    print(f"  跳过(无 STEP):     {skipped_no_step}")

    if len(valid_models) < args.num_train + args.num_val:
        print(f"错误: 有效模型({len(valid_models)})不足 {args.num_train}+{args.num_val}={args.num_train+args.num_val}")
        return

    # ---- 随机打乱 + 划分 (固定种子) ----
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(valid_models))
    total_use = args.num_train + args.num_val

    split_indices = {
        "train": indices[:args.num_train].tolist(),
        "val": indices[args.num_train:total_use].tolist(),
    }
    n_unused = len(valid_models) - total_use
    print(f"划分: train {args.num_train} | val {args.num_val} | 未使用 {n_unused}")
    print()

    # ---- 按 split 分别处理 ----
    for split_name in ("train", "val"):
        split_models = [valid_models[i] for i in split_indices[split_name]]
        split_out = os.path.join(args.output_dir, split_name)
        encoded_dir = os.path.join(split_out, "encoded_views")
        face_map_dir = os.path.join(split_out, "face_id_maps")
        os.makedirs(encoded_dir, exist_ok=True)
        os.makedirs(face_map_dir, exist_ok=True)

        total = len(split_models)
        print(f"\n[{split_name}] 开始 {total} 模型, {args.num_workers} 进程")
        t_start = time.time()

        tasks = []
        for idx, (label_path, step_path, face_labels, features) in enumerate(split_models):
            tasks.append((label_path, step_path, face_labels, features, split_out, idx, total))

        all_camera = {}
        all_models = []
        completed = 0
        failed = 0

        if args.num_workers <= 1:
            progress = make_progress(total, desc=f"{split_name}")
            for task in tasks:
                ok, model_name, result, elapsed, idx, total_t = _worker(task)
                if ok and result is not None:
                    model_entry, cam_entries = result
                    all_models.append(model_entry)
                    all_camera.update(cam_entries)
                    completed += 1
                else:
                    failed += 1
                update_progress(progress, model_name, elapsed, completed, failed, total)
            if progress is not None:
                progress.close()
        else:
            chunksize = 1
            with multiprocessing.Pool(processes=args.num_workers) as pool:
                results_iter = pool.imap_unordered(_worker, tasks, chunksize=chunksize)
                progress = make_progress(total, desc=f"{split_name}")
                for ok, model_name, result, elapsed, idx, total_t in results_iter:
                    if ok and result is not None:
                        model_entry, cam_entries = result
                        all_models.append(model_entry)
                        all_camera.update(cam_entries)
                        completed += 1
                    else:
                        failed += 1
                    update_progress(progress, model_name, elapsed, completed, failed, total)
                if progress is not None:
                    progress.close()

        total_time = time.time() - t_start
        print(f"\n[{split_name}] 完成: {total_time:.1f}s ({total_time/60:.1f}min), 成功 {completed}, 失败 {failed}")

        all_models = sorted(all_models, key=lambda item: item["model_id"])
        models_path = os.path.join(split_out, "models.json")
        camera_path = os.path.join(split_out, "camera_views.json")
        with open(models_path, "w", encoding="utf-8") as f:
            json.dump({"models": all_models}, f, indent=2, ensure_ascii=False)
        with open(camera_path, "w", encoding="utf-8") as f:
            json.dump(all_camera, f, indent=2, ensure_ascii=False)

        n_encoded = len([f for f in os.listdir(encoded_dir) if f.lower().endswith(IMAGE_EXT)])
        n_face_maps = len([f for f in os.listdir(face_map_dir) if f.lower().endswith(".npy")])
        print(f"  encoded_views: {n_encoded} 张")
        print(f"  face_id_maps:  {n_face_maps} 个")
        print(f"  models:        {len(all_models)} 条")
        print(f"  camera_views:  {len(all_camera)} 条")

    print()
    print("=" * 60)
    print(f"全部完成: {args.output_dir}")
    print(f"  train: {args.num_train} 模型 × 14视角 = {args.num_train * 14} 张图")
    print(f"  val:   {args.num_val} 模型 × 14视角 = {args.num_val * 14} 张图")
    total_imgs = (args.num_train + args.num_val) * 14
    print(f"  总计: {total_imgs} 张 encoded_views, {total_imgs} 个 face_id_maps")


if __name__ == "__main__":
    main()
