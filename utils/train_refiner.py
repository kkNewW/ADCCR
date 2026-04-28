import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.coco import COCOLocalRefineDataset
from models.local_refiner import LocalRefiner
from utils.crop_utils import crop_patch, global_to_local
from utils.refine_utils import make_gaussian_heatmap

from transformers import AutoTokenizer


class KeypointTextEncoder(nn.Module):
    def __init__(self, kpt_names, dim=768):
        super().__init__()
        self.name_to_idx = {k: i for i, k in enumerate(kpt_names)}
        self.embedding = nn.Embedding(len(kpt_names), dim)

    def forward(self, kpt_names):
        idx = torch.tensor([self.name_to_idx[k] for k in kpt_names], dtype=torch.long, device=self.embedding.weight.device)
        return self.embedding(idx)


def add_noise(xy, sigma=6.0, size=224):
    noise = torch.randn_like(xy) * sigma
    xy = xy + noise
    xy[:, 0] = xy[:, 0].clamp(0, size - 1)
    xy[:, 1] = xy[:, 1].clamp(0, size - 1)
    return xy


def collate_fn(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    kpt_names = [x["kpt_name"] for x in batch]
    descriptions = [x["description"] for x in batch]
    target_xy_224 = torch.stack([x["target_xy_224"] for x in batch], dim=0)
    crop_sizes = [x["crop_size"] for x in batch]
    return {
        "images": images,
        "kpt_names": kpt_names,
        "descriptions": descriptions,
        "target_xy_224": target_xy_224,
        "crop_sizes": crop_sizes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_dynamic_desc", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = COCOLocalRefineDataset(
        data_path=args.data_path,
        tokenizer=None,
        multimodal_cfg=dict(
            image_folder=args.image_folder,
            data_augmentation=True,
            image_size=224,
            crop_size=224,
            conv_format="keypoint",
            use_dynamic_desc=args.use_dynamic_desc,
            desc_mode="dynamic",
        ),
        is_train=True
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    device = "cuda"
    model = LocalRefiner(text_dim=768, feat_dim=256, hm_size=32).to(device)
    text_encoder = KeypointTextEncoder(dataset.multimodal_cfg.get("kpt_names", [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ])).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(text_encoder.parameters()),
        lr=args.lr
    )

    for epoch in range(args.epochs):
        model.train()
        text_encoder.train()
        total_loss = 0.0

        for batch in loader:
            images = batch["images"].to(device)
            kpt_names = batch["kpt_names"]
            gt_xy = batch["target_xy_224"].to(device)
            crop_sizes = batch["crop_sizes"]

            noisy_xy = add_noise(gt_xy, sigma=6.0, size=224)
            patches = []
            gt_hms = []

            for i in range(images.shape[0]):
                patch, crop_box = crop_patch(images[i], noisy_xy[i].tolist(), crop_sizes[i])
                patches.append(patch)

                lx, ly = global_to_local(gt_xy[i].tolist(), crop_box, hm_size=32)
                gt_hm = make_gaussian_heatmap(32, (lx, ly), sigma=2.0)
                gt_hms.append(gt_hm)

            patches = torch.stack(patches, dim=0).to(device)
            gt_hms = torch.stack(gt_hms, dim=0).unsqueeze(1).to(device)

            text_feat = text_encoder(kpt_names)
            pred_hm = model(patches, text_feat)

            loss = ((pred_hm - gt_hms) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, loss={total_loss / len(loader):.6f}")
        torch.save(
            {
                "model": model.state_dict(),
                "text_encoder": text_encoder.state_dict(),
            },
            os.path.join(args.output_dir, f"refiner_epoch_{epoch+1}.pth")
        )


if __name__ == "__main__":
    main()
