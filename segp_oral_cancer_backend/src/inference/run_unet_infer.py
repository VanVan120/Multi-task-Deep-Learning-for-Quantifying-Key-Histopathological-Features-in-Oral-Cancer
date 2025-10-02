from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.data.seg_pair_dataset import SegPairDataset
from src.models.unet import UNet


def save_overlay(rgb: np.ndarray, mask_pred: np.ndarray, out_path: Path, alpha: float = 0.4) -> None:
    overlay = rgb.astype(np.float32)
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    mask = mask_pred.astype(bool)[..., None]
    blended = np.where(mask, (1 - alpha) * overlay + alpha * red, overlay)
    Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(out_path)


def build_unet(base_channels: int) -> UNet:
    try:
        return UNet(n_classes=2, in_ch=3, base=base_channels)
    except TypeError:
        return UNet(in_channels=3, num_classes=2, base_channels=base_channels)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--split", default="test")
    parser.add_argument("--save-dir", default="outputs/unet_pred")
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold for tumour probability")
    parser.add_argument(
        "--save-probmaps",
        action="store_true",
        help="Also save tumour probability maps as grayscale PNGs",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    dataset = SegPairDataset(args.data_root, args.split, args.img_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    net = build_unet(args.base_ch).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    net.load_state_dict(state_dict)
    net.eval()

    out_root = Path(args.save_dir)
    out_masks = out_root / "masks"
    out_overlays = out_root / "overlays"
    out_masks.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)
    out_probmaps = out_root / "probmaps"
    if args.save_probmaps:
        out_probmaps.mkdir(parents=True, exist_ok=True)

    for images, _, paths in dataloader:
        images = images.to(device)
        logits = net(images)
        probs = torch.softmax(logits, dim=1)
        prob_tumour = probs[:, 1]
        prob_map = prob_tumour[0].cpu().numpy()
        pred_mask = (prob_map > args.thr).astype(np.uint8)

        path_str = paths[0] if isinstance(paths, (list, tuple)) else paths
        img_path = Path(path_str)
        mask_path = out_masks / f"{img_path.stem}.png"
        overlay_path = out_overlays / f"{img_path.stem}.jpg"

        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_path)
        if args.save_probmaps:
            prob_image = (prob_map * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(prob_image).save(out_probmaps / f"{img_path.stem}.png")
        with Image.open(img_path) as origin:
            rgb = np.array(origin.convert("RGB"))
        save_overlay(rgb, pred_mask, overlay_path)

    print(f"Saved masks to {out_masks} and overlays to {out_overlays}")


if __name__ == "__main__":
    main()
