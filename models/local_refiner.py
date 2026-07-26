import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMFusion(nn.Module):
    def __init__(self, feat_dim, text_dim):
        super().__init__()

        self.gamma = nn.Linear(
            text_dim,
            feat_dim
        )
        self.beta = nn.Linear(
            text_dim,
            feat_dim
        )

    def forward(self, visual_features, text_features):
        gamma = self.gamma(text_features)
        beta = self.beta(text_features)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return (
            visual_features * (1.0 + gamma)
            + beta
        )


class LocalBackbone(nn.Module):
    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                feat_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, patches):
        return self.layers(patches)


class LocalRefiner(nn.Module):
    def __init__(
        self,
        text_dim=768,
        feat_dim=256,
        hm_size=64,
    ):
        super().__init__()

        self.backbone = LocalBackbone(
            in_channels=3,
            feat_dim=feat_dim,
        )

        self.fusion = FiLMFusion(
            feat_dim=feat_dim,
            text_dim=text_dim,
        )

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(
                feat_dim,
                feat_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                feat_dim,
                1,
                kernel_size=1,
            ),
        )

        self.hm_size = hm_size

    def forward(self, patches, text_features):
        features = self.backbone(patches)
        features = self.fusion(
            features,
            text_features,
        )

        heatmaps = self.heatmap_head(features)

        heatmaps = F.interpolate(
            heatmaps,
            size=(self.hm_size, self.hm_size),
            mode="bilinear",
            align_corners=False,
        )

        return heatmaps


def crop_and_resize(
    images,
    centers,
    crop_sizes,
    output_size=128,
):
    """
    Differentiable crop in the 224x224 person coordinate
    system. Regions outside the image are zero padded.
    """
    if images.ndim != 4:
        raise ValueError(
            "images must have shape [N, C, H, W]"
        )

    n, _, height, width = images.shape

    if centers.shape != (n, 2):
        raise ValueError(
            "centers must have shape [N, 2]"
        )

    crop_sizes = crop_sizes.reshape(n)

    relative = torch.linspace(
        -0.5,
        0.5,
        output_size,
        device=images.device,
        dtype=images.dtype,
    )

    grid_y, grid_x = torch.meshgrid(
        relative,
        relative,
        indexing="ij",
    )

    pixel_x = (
        centers[:, 0, None, None]
        + grid_x[None]
        * crop_sizes[:, None, None]
    )

    pixel_y = (
        centers[:, 1, None, None]
        + grid_y[None]
        * crop_sizes[:, None, None]
    )

    normalized_x = (
        2.0 * pixel_x / max(width - 1, 1)
        - 1.0
    )
    normalized_y = (
        2.0 * pixel_y / max(height - 1, 1)
        - 1.0
    )

    sampling_grid = torch.stack(
        [normalized_x, normalized_y],
        dim=-1,
    )

    patches = F.grid_sample(
        images,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    half_size = crop_sizes / 2.0
    crop_boxes = torch.stack(
        [
            centers[:, 0] - half_size,
            centers[:, 1] - half_size,
            centers[:, 0] + half_size,
            centers[:, 1] + half_size,
        ],
        dim=-1,
    )

    return patches, crop_boxes


def build_gaussian_heatmaps(
    target_xy,
    crop_boxes,
    heatmap_size=64,
    sigma=2.0,
):
    x1, y1, x2, y2 = crop_boxes.unbind(dim=-1)

    crop_width = (x2 - x1).clamp_min(1.0)
    crop_height = (y2 - y1).clamp_min(1.0)

    local_x = (
        (target_xy[:, 0] - x1)
        / crop_width
        * (heatmap_size - 1)
    )
    local_y = (
        (target_xy[:, 1] - y1)
        / crop_height
        * (heatmap_size - 1)
    )

    axis = torch.arange(
        heatmap_size,
        device=target_xy.device,
        dtype=target_xy.dtype,
    )

    grid_y, grid_x = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    squared_distance = (
        (
            grid_x[None]
            - local_x[:, None, None]
        ) ** 2
        + (
            grid_y[None]
            - local_y[:, None, None]
        ) ** 2
    )

    heatmaps = torch.exp(
        -squared_distance
        / (2.0 * sigma ** 2)
    )

    return heatmaps.unsqueeze(1)


def heatmaps_to_global(
    heatmaps,
    crop_boxes,
):
    batch_size, _, height, width = heatmaps.shape

    flat_indices = heatmaps[:, 0].flatten(1).argmax(
        dim=1
    )

    local_y = torch.div(
        flat_indices,
        width,
        rounding_mode="floor",
    ).to(heatmaps.dtype)

    local_x = (
        flat_indices % width
    ).to(heatmaps.dtype)

    x1, y1, x2, y2 = crop_boxes.unbind(dim=-1)

    global_x = (
        x1
        + local_x
        / max(width - 1, 1)
        * (x2 - x1)
    )

    global_y = (
        y1
        + local_y
        / max(height - 1, 1)
        * (y2 - y1)
    )

    return torch.stack(
        [global_x, global_y],
        dim=-1,
    )