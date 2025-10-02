import argparse, yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from src.data.wsi_dataset import build_imagefolder_datasets
from src.models.resnet_tvnt import TVNTResNet

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, test_ds = build_imagefolder_datasets(cfg["data"]["val_dir"], cfg["data"]["test_dir"], cfg["data"]["image_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = TVNTResNet(cfg["model"]["name"], False, cfg["model"]["num_classes"]).to(device)
    state = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(state); model.eval()

    y_true, y_pred, y_prob = [], [], []
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(torch.argmax(prob,1).cpu().numpy().tolist())
        y_prob.extend(prob[:,1].cpu().numpy().tolist())

    print("F1 (macro):", f1_score(y_true, y_pred, average="macro"))
    try:
        print("AUC:", roc_auc_score(y_true, y_prob))
    except Exception:
        print("AUC: nan (needs both classes present)")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Report:\n", classification_report(y_true, y_pred, target_names=test_ds.classes))

if __name__ == "__main__":
    main()
