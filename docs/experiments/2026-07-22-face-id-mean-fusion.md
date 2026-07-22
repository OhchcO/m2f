# Face-ID 跨视图均值融合

**状态：待训练**

## 动机

当前 VideoMaskFormer 将 14 个渲染视图的特征一起交给共享 Query 做 cross-attention，但没有被显式告知不同视图中哪些区域对应同一个三维 CAD 面。模型只能依赖二维外观、帧序号和实例监督自行猜测跨视图对应，因此多视图效果可能低于单视图。

数据集和推理输入均可从 STEP 渲染得到 `face_id_map`：每个像素包含其所属 CAD 面的确定 ID。这提供了精确的跨视图面级对应，不需要由网络猜测。

## 假设与成功标准

**假设：** 将同一模型、同一 `face_id` 在全部可见视图中的特征聚合，再回写给各视图对应区域，能为遮挡、视角变化或局部外观不清晰的面补充证据，从而提高面级分割指标。

**成功标准：** 在相同验证集、评估阈值和训练预算下，融合模型的面级细粒度指标高于原 VideoMaskFormer 多视图基线。也应与当前最佳单视图模型比较。

## 修改内容

数据流：

```text
14 张 RGB + 14 张 face_id_map
          ↓
共享 Backbone + Pixel Decoder
          ↓
FaceFeatureFusion（新增）
          ↓
原 VideoMaskFormer Decoder
```

对每个特征尺度，设 `F[b,t,x,y]` 为 batch 中模型 `b`、视图 `t` 的 token，且该 token 的面 ID 为 `f`：

\[
z_{b,f}=\operatorname{Mean}\{F[b,t,x,y] \mid face\_id[b,t,x,y]=f\}
\]

将均值残差回写：

\[
F'[b,t,x,y]=F[b,t,x,y]+\gamma z_{b,f}
\]

`face_id=-1` 的背景及 padding 不参与聚合。对 3 个 `multi_scale_features` 和 1 个 `mask_features` 分别执行融合。

`gamma` 为每个特征尺度独立的可学习标量，初始化为 `0.0`。因此训练第一个 step 的模型行为等价于原模型；训练自行决定是否使用融合信息。

涉及文件：

- `Mask2Former/mask2former_video/modeling/face_feature_fusion.py`
- `Mask2Former/mask2former_video/data_video/dataset_mapper.py`
- `Mask2Former/mask2former_video/video_maskformer_model.py`
- `Mask2Former/mask2former/modeling/meta_arch/mask_former_head.py`
- `Mask2Former/configs/mfr_multiview/video_maskformer2_R50_bs1_14view.yaml`
- `new_add/finetune_mfr_multiview_face_mean.sh`

## 训练设置

快速验证脚本：

```bash
cd /data/m2f

BASE_WEIGHTS=/path/to/original_multiview/model_final.pth \
DATASET_DIR=/hy-tmp/datasets/MFRInstSegM2F_2100 \
OUTPUT_DIR=/hy-tmp/mfr_multiview_face_mean_20k \
CUDA_VISIBLE_DEVICES=0 \
./new_add/finetune_mfr_multiview_face_mean.sh
```

默认快速验证参数：

| 参数 | 值 |
| --- | --- |
| 输入 | 14 视图，512×512 |
| 初始化 | 原多视图 checkpoint；不使用 `--resume` |
| `MAX_ITER` | 20,000 |
| `BASE_LR` | `1e-5` |
| 学习率衰减 | 15,000 iter |
| batch | 1 个模型 × 14 视图 |
| 融合 | `MODEL.FACE_FUSION.ENABLED=True`，`INIT_GAMMA=0.0` |

## 对照组

| 组别 | checkpoint / 设置 | 面级细粒度指标 | 备注 |
| --- | --- | --- | --- |
| A | 当前最佳单视图模型 | 待填 | 14 个独立视图后融合 |
| B | 原 VideoMaskFormer 多视图模型 | 待填 | 无 Face-ID 特征融合 |
| C | Face-ID Mean Fusion 20k | 待填 | 本实验 |

评估使用 `new_add/eval_mfr_multiview.py`，且三个组必须保持相同验证集、`score_threshold`、`mask_threshold` 与面级回投规则。

## 结果

待训练后填写：

- checkpoint 路径：
- 训练完成 iter：
- 验证集细粒度指标：
- 验证集粗粒度指标：
- 各尺度 `gamma` 最终值：
- 典型成功案例：
- 典型失败案例：

## 结论与下一步

待结果确定后填写：

- 若 C > B：验证精确 CAD 面级跨视图对应有效；下一步比较均值池化与视角/内容加权池化。
- 若 C ≤ B：检查面级均值是否过度平滑；可尝试仅在较低分辨率层融合、降低/限制 `gamma`，或分析标签与实例定义。
- 若 C > B 但仍 < A：分析 VideoMaskFormer 的 Query 匹配、损失和跨视图实例表达是否仍限制性能。
