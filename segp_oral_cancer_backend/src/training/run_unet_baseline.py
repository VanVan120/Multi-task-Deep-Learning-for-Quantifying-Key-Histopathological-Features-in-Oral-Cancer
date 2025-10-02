from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.seg_pair_dataset import SegPairDataset
from src.models.unet import UNet


def dice_iou_prob(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    """Compute soft Dice/IoU scores from class-1 probabilities and masks."""

    with torch.no_grad():
        probs = probs.clamp(0.0, 1.0)
        target = (target > 0.5).float()
        inter = (probs * target).sum(dim=(1, 2, 3))
        p_sum = probs.sum(dim=(1, 2, 3))
        t_sum = target.sum(dim=(1, 2, 3))
        union = p_sum + t_sum - inter
        dice = ((2 * inter + eps) / (p_sum + t_sum + eps)).mean().item()
        iou = ((inter + eps) / (union + eps)).mean().item()
        return dice, iou


def dice_loss_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft Dice loss for the foreground class probabilities."""

    probs = probs.clamp(0.0, 1.0)
    target = (target > 0.5).float()
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def build_unet(base_channels: int) -> UNet:
    try:
        return UNet(n_classes=2, in_ch=3, base=base_channels)
    except TypeError:
        return UNet(in_channels=3, num_classes=2, base_channels=base_channels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", default="runs/unet_baseline")
    parser.add_argument("--base-ch", type=int, default=32)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0
    pin_memory = device.type == "cuda"

    train_ds = SegPairDataset(args.data_root, "train", args.img_size, augment=True)
    val_ds = SegPairDataset(args.data_root, "val", args.img_size, augment=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    net = build_unet(args.base_ch).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    class_weight = torch.tensor([1.0, 3.0], device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weight)

    log_path = outdir / "log.csv"
    best_ckpt = outdir / "best.ckpt"
    best_dice = float("-inf")

    with log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou"])

        for epoch in range(1, args.epochs + 1):
            net.train()
            train_loss = 0.0
            for images, masks, _ in train_dl:
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                logits = net(images)
                mask_classes = (masks > 0.5).long().squeeze(1)
                loss_ce = criterion_ce(logits, mask_classes)
                probs_tumour = torch.softmax(logits, dim=1)[:, 1:2]
                loss_dice = dice_loss_from_probs(probs_tumour, masks)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss_epoch = train_loss / max(len(train_ds), 1)

            net.eval()
            val_loss_total = 0.0
            dices: list[float] = []
            ious: list[float] = []
            with torch.no_grad():
                for images, masks, _ in val_dl:
                    images = images.to(device)
                    masks = masks.to(device)
                    logits = net(images)
                    mask_classes = (masks > 0.5).long().squeeze(1)
                    loss_ce = criterion_ce(logits, mask_classes)
                    probs_tumour = torch.softmax(logits, dim=1)[:, 1:2]
                    loss_dice = dice_loss_from_probs(probs_tumour, masks)
                    loss = 0.5 * loss_ce + 0.5 * loss_dice
                    val_loss_total += loss.item() * images.size(0)
                    dice, iou = dice_iou_prob(probs_tumour, masks)
                    dices.append(dice)
                    ious.append(iou)

            val_count = max(len(val_ds), 1)
            val_loss = val_loss_total / val_count
            val_dice = sum(dices) / len(dices) if dices else 0.0
            val_iou = sum(ious) / len(ious) if ious else 0.0

            writer.writerow(
                [
                    epoch,
                    f"{train_loss_epoch:.4f}",
                    f"{val_loss:.4f}",
                    f"{val_dice:.4f}",
                    f"{val_iou:.4f}",
                ]
            )
            handle.flush()
            print(
                f"[Epoch {epoch}] train {train_loss_epoch:.4f} | val {val_loss:.4f} | "
                f"Dice {val_dice:.4f} | IoU {val_iou:.4f}"
            )

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(net.state_dict(), best_ckpt)

    print(f"Best model saved to: {best_ckpt}")


if __name__ == "__main__":
    main()
