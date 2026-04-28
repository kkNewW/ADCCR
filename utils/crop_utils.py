import torch
import torch.nn.functional as F


def clamp_xy(x, y, w, h):
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return x, y


def crop_patch(image, center_xy, crop_size):
    """
    image: [C, H, W]
    center_xy: (x, y) in pixel space of current transformed image
    """
    C, H, W = image.shape
    cx = int(center_xy[0])
    cy = int(center_xy[1])

    half = crop_size // 2
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    patch = image[:, y1:y2, x1:x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        patch = F.pad(
            patch,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0
        )

    crop_box = (x1 - pad_left, y1 - pad_top, x1 - pad_left + crop_size, y1 - pad_top + crop_size)
    return patch, crop_box


def global_to_local(gt_xy, crop_box, hm_size):
    x1, y1, x2, y2 = crop_box
    gx, gy = gt_xy
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)
    lx = (gx - x1) / crop_w * hm_size
    ly = (gy - y1) / crop_h * hm_size
    return lx, ly


def local_to_global(local_xy, crop_box, hm_size):
    x1, y1, x2, y2 = crop_box
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)
    gx = x1 + local_xy[0] / hm_size * crop_w
    gy = y1 + local_xy[1] / hm_size * crop_h
    return gx, gy
