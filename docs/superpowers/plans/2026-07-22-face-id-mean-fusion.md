# Face-ID Mean Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse Pixel Decoder features belonging to the same visible CAD face across the 14 rendered views before VideoMaskFormer decoding.

**Architecture:** The dataset mapper retains the augmented per-pixel face IDs and the video model pads and forwards them to the segmentation head. A new residual feature-fusion module pools every valid `(batch, face_id)` group across all frames, broadcasts its mean back to member tokens, and is gated by a zero-initialized scale so the initial model equals the baseline.

**Tech Stack:** PyTorch, Detectron2, Mask2Former, pytest.

## Global Constraints

- Use `-1` as the background/padding face ID; it must never affect or receive fusion.
- Preserve feature tensor shapes and all existing decoder/loss interfaces.
- Keep the feature-fusion gate initialized to `0.0`.
- Run the same fusion on the three multi-scale features and mask features.

---

### Task 1: Face-level feature fusion primitive

**Files:**
- Create: `Mask2Former/mask2former_video/modeling/face_feature_fusion.py`
- Test: `Mask2Former/tests/test_face_feature_fusion.py`

**Interfaces:**
- Produces: `FaceFeatureFusion.forward(mask_features, multi_scale_features, face_id_maps, num_frames)` returning shape-preserving fused tensors.

- [x] Write tests covering same-face cross-frame mean broadcast, background exclusion, and zero gate identity.
- [x] Run them and confirm they initially fail because the module is absent.
- [x] Implement vectorized per-model `(batch, face_id)` mean pooling with `index_add_`, residual writeback, and zero-initialized gates.
- [x] Run the tests and confirm they pass.

### Task 2: Data and model plumbing

**Files:**
- Modify: `Mask2Former/mask2former_video/data_video/dataset_mapper.py`
- Modify: `Mask2Former/mask2former_video/video_maskformer_model.py`
- Modify: `Mask2Former/mask2former/modeling/meta_arch/mask_former_head.py`
- Modify: `Mask2Former/mask2former/config.py`
- Test: `Mask2Former/tests/test_face_feature_fusion.py`

**Interfaces:**
- Mapper produces `face_id_maps: list[Tensor[H,W]]` aligned to `image`.
- Video model pads the maps with `-1` and calls `sem_seg_head(features, face_id_maps=..., num_frames=...)`.
- Segmentation head calls `FaceFeatureFusion` when `MODEL.FACE_FUSION.ENABLED`.

- [x] Implement mapper retention, `-1` padding, config defaults, and optional segmentation-head fusion.
- [x] Run targeted feature tests, static compilation, import checks, and a CPU model-forward smoke test.

### Task 3: Evaluation integration and verification

**Files:**
- Modify: `new_add/eval_mfr_multiview.py`
- Modify: `Mask2Former/configs/mfr_multiview/video_maskformer2_R50_bs1_14view.yaml`

**Interfaces:**
- Evaluation supplies its already-loaded face maps as `face_id_maps` in the model input.

- [x] Enable mean face fusion in the MFR multiview configuration.
- [x] Pass face maps to the evaluator model call.
- [x] Run unit tests, compile changed Python files, and run CPU config/model smoke checks.
