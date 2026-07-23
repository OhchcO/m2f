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


def test_mask_features_can_be_excluded_while_multiscale_features_fuse():
    module = load_face_fusion_module()
    fusion = module.FaceFeatureFusion(
        feature_channels=[1, 1], init_gamma=1.0, fuse_mask_features=False
    )
    mask_feature = torch.tensor([[[[2.0]]], [[[6.0]]]])
    multi_scale_feature = torch.tensor([[[[2.0]]], [[[6.0]]]])
    face_ids = torch.tensor([[[7]], [[7]]])

    fused_mask, fused_scales = fusion(
        mask_feature, [multi_scale_feature], face_ids, num_frames=2
    )

    assert torch.equal(fused_mask, mask_feature)
    assert torch.equal(
        fused_scales[0], torch.tensor([[[[6.0]]], [[[10.0]]]])
    )


def test_content_attention_prefers_the_more_informative_view_feature():
    module = load_face_fusion_module()
    fusion = module.FaceFeatureFusion(
        feature_channels=[1], init_gamma=1.0, aggregation="content_attention"
    )
    scorer = fusion.attention_scorers[0]
    with torch.no_grad():
        scorer[0].weight.fill_(1.0)
        scorer[0].bias.zero_()
        scorer[2].weight.fill_(1.0)
        scorer[2].bias.zero_()

    # The two frames show one face with features 1 and 3. Content attention
    # assigns a larger weight to the stronger second-view feature, therefore
    # their shared message is greater than the arithmetic mean (2).
    feature = torch.tensor([[[[1.0]]], [[[3.0]]]])
    face_ids = torch.tensor([[[7]], [[7]]])
    fused_mask, _ = fusion(feature, [], face_ids, num_frames=2)

    assert fused_mask[0, 0, 0, 0] > 3.0
    assert fused_mask[1, 0, 0, 0] > 5.0
