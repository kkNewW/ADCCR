import torch


def make_gaussian_heatmap(size, center, sigma=2.0):
    y = torch.arange(size).float().unsqueeze(1)
    x = torch.arange(size).float().unsqueeze(0)
    cx, cy = center
    heatmap = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return heatmap


def soft_argmax_2d(heatmap):
    """
    heatmap: [B, 1, H, W]
    return: [B, 2]
    """
    B, _, H, W = heatmap.shape
    prob = torch.softmax(heatmap.view(B, -1), dim=-1)

    xs = torch.arange(W, device=heatmap.device).float()
    ys = torch.arange(H, device=heatmap.device).float()

    x_map = xs.unsqueeze(0).repeat(H, 1).reshape(-1)
    y_map = ys.unsqueeze(1).repeat(1, W).reshape(-1)

    x = torch.sum(prob * x_map.unsqueeze(0), dim=1)
    y = torch.sum(prob * y_map.unsqueeze(0), dim=1)
    return torch.stack([x, y], dim=-1)
