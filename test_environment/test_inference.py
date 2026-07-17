import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MASK2FORMER_DIR = PROJECT_DIR / "Mask2Former"

sys.path.insert(0, str(MASK2FORMER_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from mask2former import add_maskformer2_config


def setup_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(MASK2FORMER_DIR / "configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml"))
    cfg.MODEL.WEIGHTS = str(SCRIPT_DIR / "model_final_94dc52.pkl")
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    # 加载配置和模型
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("coco_2017_val_panoptic")

    # 读取图片
    image_path = SCRIPT_DIR / "test.png"
    if not os.path.exists(image_path):
        print(f"[ERROR] 找不到图片: {image_path}")
        sys.exit(1)

    im = cv2.imread(str(image_path))
    print(f"[OK] 图片加载成功: {image_path} ({im.shape[1]}x{im.shape[0]})")

    # 推理
    import time
    start = time.time()
    outputs = predictor(im)
    print(f"[OK] 推理完成，耗时: {time.time() - start:.2f}s")

    # 1. 全景分割
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.0)
    panoptic_result = v.draw_panoptic_seg(
        outputs["panoptic_seg"][0].to("cpu"),
        outputs["panoptic_seg"][1]
    ).get_image()
    panoptic_path = SCRIPT_DIR / "output_panoptic.jpg"
    cv2.imwrite(str(panoptic_path), panoptic_result[:, :, ::-1])
    print(f"[OK] 全景分割已保存: {panoptic_path}")

    # 2. 实例分割
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.0)
    instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
    instance_path = SCRIPT_DIR / "output_instance.jpg"
    cv2.imwrite(str(instance_path), instance_result[:, :, ::-1])
    print(f"[OK] 实例分割已保存: {instance_path}")

    # 3. 语义分割
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.0)
    semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
    semantic_path = SCRIPT_DIR / "output_semantic.jpg"
    cv2.imwrite(str(semantic_path), semantic_result[:, :, ::-1])
    print(f"[OK] 语义分割已保存: {semantic_path}")
    print()
    print("=== 推理测试通过！环境配置正常 ===")
