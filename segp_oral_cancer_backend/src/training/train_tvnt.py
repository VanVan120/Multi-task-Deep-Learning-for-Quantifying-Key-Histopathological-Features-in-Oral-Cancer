import argparse, os, random
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score

from src.data.wsi_dataset import build_imagefolder_datasets
from src.models.resnet_tvnt import TVNTResNet

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    losses, preds_all, targets_all = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).squeeze().backward()
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
        losses.append(loss.item())
        preds_all.extend(torch.argmax(logits,1).detach().cpu().tolist())
        targets_all.extend(y.detach().cpu().tolist())
    f1 = f1_score(targets_all, preds_all, average="macro")
    return np.mean(losses), f1

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, preds, probs, targets = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        p = torch.softmax(logits, dim=1)
        preds.extend(torch.argmax(p,1).cpu().tolist())
        probs.extend(p[:,1].cpu().tolist())
        targets.extend(y.cpu().tolist())
    f1 = f1_score(targets, preds, average="macro")
    try:
        auc = roc_auc_score(targets, probs)
    except Exception:
        auc = float("nan")
    return np.mean(losses), f1, auc

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["log_dir"]); log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    train_ds, val_ds = build_imagefolder_datasets(
        cfg["data"]["train_dir"], cfg["data"]["val_dir"], cfg["data"]["image_size"]
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = TVNTResNet(cfg["model"]["name"], cfg["model"]["pretrained"], cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler() if (cfg["train"]["mixed_precision"] and device=="cuda") else None

    best_auc, patience, es_counter = -1, cfg["train"]["early_stop_patience"], 0
    epochs = cfg["train"]["epochs"]
    for epoch in range(1, epochs+1):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        va_loss, va_f1, va_auc = validate(model, val_loader, criterion, device)

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("f1/train", tr_f1, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("f1/val", va_f1, epoch)
        writer.add_scalar("auc/val", va_auc, epoch)

        print(f"[{epoch}/{epochs}] train_loss={tr_loss:.4f} f1={tr_f1:.4f} | val_loss={va_loss:.4f} f1={va_f1:.4f} auc={va_auc:.4f}")

        if va_auc > best_auc:
            best_auc = va_auc; es_counter = 0
            torch.save(model.state_dict(), cfg["checkpoint_path"])
            print(f"Saved best checkpoint -> {cfg['checkpoint_path']} (AUC={va_auc:.4f})")
        else:
            es_counter += 1
            if es_counter >= patience:
                print("Early stopping.")
                break

    writer.close()

if __name__ == "__main__":
    main()
