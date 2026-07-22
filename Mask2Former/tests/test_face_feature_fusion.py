import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mask2former_video/modeling/face_feature_fusion.py"
)


def load_face_fusion_module():
    spec = importlib.util.spec_from_file_location("face_feature_fusion", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_face_mean_is_broadcast_across_video_frames():
    module = load_face_fusion_module()
    fusion = module.FaceFeatureFusion(feature_channels=[1], init_gamma=1.0)

    # Two frames of one model. Face 7 contributes values 2 and 6, so both
    # visible tokens should receive their shared mean, 4, through the residual.
    feature = torch.tensor([[[[2.0, 10.0]]], [[[6.0, 20.0]]]])
    face_ids = torch.tensor([[[7, -1]], [[7, -1]]])

    fused_mask, fused_scales = fusion(feature, [], face_ids, num_frames=2)

    assert fused_scales == []
    assert torch.equal(fused_mask, torch.tensor([[[[6.0, 10.0]]], [[[10.0, 20.0]]]]))


def test_background_is_unchanged_and_zero_gate_is_identity():
    module = load_face_fusion_module()
    fusion = module.FaceFeatureFusion(feature_channels=[1], init_gamma=0.0)
    feature = torch.tensor([[[[2.0, 10.0]]], [[[6.0, 20.0]]]])
    face_ids = torch.tensor([[[7, -1]], [[7, -1]]])

    fused_mask, _ = fusion(feature, [], face_ids, num_frames=2)

    assert torch.equal(fused_mask, feature)
