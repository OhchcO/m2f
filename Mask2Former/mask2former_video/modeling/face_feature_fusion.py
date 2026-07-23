"""Feature fusion over exact CAD-face correspondences across video frames."""

from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FaceFeatureFusion(nn.Module):
    """Broadcast each CAD face's cross-view mean feature back to its tokens.

    ``face_id_maps`` contains one integer CAD face id per input pixel.  A face
    id is only meaningful within a single video, so pooling groups use both
    the batch index and face id.  ``-1`` denotes background or padded pixels
    and is left untouched.
    """

    def __init__(
        self,
        feature_channels: List[int],
        init_gamma: float = 0.0,
        fuse_mask_features: bool = True,
        aggregation: str = "mean",
    ):
        super().__init__()
        if aggregation not in {"mean", "content_attention"}:
            raise ValueError(f"unsupported face feature aggregation: {aggregation}")
        # The values are deliberately scalar gates: gamma=0 preserves the
        # baseline exactly while training can enable each resolution separately.
        self.gammas = nn.Parameter(torch.full((len(feature_channels),), init_gamma))
        self.fuse_mask_features = fuse_mask_features
        self.aggregation = aggregation
        self.attention_scorers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels, max(channels // 4, 32)),
                    nn.ReLU(),
                    nn.Linear(max(channels // 4, 32), 1),
                )
                for channels in feature_channels
            ]
        )
        # Uniform view weighting at initialization makes the attention variant
        # initially equivalent to mean pooling once its residual gate opens.
        for scorer in self.attention_scorers:
            nn.init.zeros_(scorer[-1].weight)
            nn.init.zeros_(scorer[-1].bias)

    @staticmethod
    def _resize_face_ids(face_id_maps: Tensor, height: int, width: int) -> Tensor:
        return F.interpolate(
            face_id_maps.unsqueeze(1).float(),
            size=(height, width),
            mode="nearest",
        ).squeeze(1).long()

    def _fuse_one(
        self,
        features: Tensor,
        face_id_maps: Tensor,
        num_frames: int,
        gamma: Tensor,
        attention_scorer: nn.Module,
    ) -> Tensor:
        bt, channels, height, width = features.shape
        if bt % num_frames != 0:
            raise ValueError(f"feature batch {bt} is not divisible by num_frames={num_frames}")
        if face_id_maps.shape[0] != bt:
            raise ValueError("face_id_maps and features must have the same flattened video batch size")

        face_ids = self._resize_face_ids(face_id_maps, height, width)
        batch_size = bt // num_frames
        face_ids = face_ids.view(batch_size, num_frames, height, width)
        token_features = features.view(batch_size, num_frames, channels, height, width)
        token_features = token_features.permute(0, 1, 3, 4, 2).reshape(-1, channels)

        flat_face_ids = face_ids.reshape(-1)
        valid = flat_face_ids >= 0
        if not valid.any():
            return features

        # ``unique`` avoids assumptions about the maximum face id and combines
        # only frames that belong to the same video in the batch.
        frame_batch_ids = torch.arange(batch_size, device=features.device)
        frame_batch_ids = frame_batch_ids[:, None, None, None].expand(-1, num_frames, height, width).reshape(-1)
        frame_indices = torch.arange(num_frames, device=features.device)
        frame_indices = frame_indices[None, :, None, None].expand(batch_size, -1, height, width).reshape(-1)
        valid_face_ids = flat_face_ids[valid]
        face_keys = torch.stack((frame_batch_ids[valid], valid_face_ids), dim=1)
        _, face_inverse = torch.unique(face_keys, dim=0, return_inverse=True)

        valid_features = token_features[valid]
        if self.aggregation == "mean":
            face_sums = valid_features.new_zeros((int(face_inverse.max()) + 1, channels))
            face_sums.index_add_(0, face_inverse, valid_features)
            face_counts = torch.bincount(face_inverse, minlength=face_sums.shape[0]).to(valid_features.dtype).unsqueeze(1)
            face_features = face_sums / face_counts.clamp_min_(1)
            shared_features = face_features[face_inverse]
        else:
            # Pool each (model, view, face) region first. Attention then selects
            # among views rather than letting a large projected area dominate.
            view_keys = torch.stack(
                (frame_batch_ids[valid], frame_indices[valid], valid_face_ids), dim=1
            )
            unique_view_keys, view_inverse = torch.unique(view_keys, dim=0, return_inverse=True)
            view_sums = valid_features.new_zeros((int(view_inverse.max()) + 1, channels))
            view_sums.index_add_(0, view_inverse, valid_features)
            view_counts = torch.bincount(view_inverse, minlength=view_sums.shape[0]).to(valid_features.dtype).unsqueeze(1)
            view_features = view_sums / view_counts.clamp_min_(1)

            unique_face_keys, view_to_face = torch.unique(
                unique_view_keys[:, (0, 2)], dim=0, return_inverse=True
            )
            logits = attention_scorer(view_features).squeeze(1)
            face_max = logits.new_full((unique_face_keys.shape[0],), float("-inf"))
            face_max.scatter_reduce_(0, view_to_face, logits, reduce="amax", include_self=True)
            unnormalized = torch.exp(logits - face_max[view_to_face])
            face_denominators = logits.new_zeros((unique_face_keys.shape[0],))
            face_denominators.index_add_(0, view_to_face, unnormalized)
            view_weights = unnormalized / face_denominators[view_to_face].clamp_min(1e-12)

            weighted_face_features = view_features.new_zeros((unique_face_keys.shape[0], channels))
            weighted_face_features.index_add_(0, view_to_face, view_features * view_weights.unsqueeze(1))
            # Map every token's (model, face) group to its weighted shared feature.
            _, token_to_face = torch.unique(face_keys, dim=0, return_inverse=True)
            shared_features = weighted_face_features[token_to_face]

        fused_tokens = token_features.clone()
        fused_tokens[valid] = valid_features + gamma * shared_features
        return fused_tokens.view(batch_size, num_frames, height, width, channels).permute(0, 1, 4, 2, 3).reshape_as(features)

    def forward(
        self,
        mask_features: Tensor,
        multi_scale_features: List[Tensor],
        face_id_maps: Tensor,
        num_frames: int,
    ) -> Tuple[Tensor, List[Tensor]]:
        fused_mask = (
            self._fuse_one(mask_features, face_id_maps, num_frames, self.gammas[0], self.attention_scorers[0])
            if self.fuse_mask_features
            else mask_features
        )
        fused_scales = [
            self._fuse_one(
                feature, face_id_maps, num_frames, self.gammas[index], self.attention_scorers[index]
            )
            for index, feature in enumerate(multi_scale_features, start=1)
        ]
        return fused_mask, fused_scales
