import io
import os
from typing import Dict

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms

from src.models.resnet_tvnt import TVNTResNet

app = FastAPI(title="SEGP Oral Cancer API", version="0.1.0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
CKPT_PATH = os.getenv("TVNT_CKPT", "models_ckpt/tvnt_resnet18.pt")

_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

_model = None

@app.on_event("startup")
def load_model():
    global _model
    _model = TVNTResNet(model_name="resnet18", pretrained=False, num_classes=2)
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location="cpu")
        _model.load_state_dict(state)
    _model.eval()
    _model.to(DEVICE)

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict/tvnt")
async def predict_tvnt(file: UploadFile = File(...)) -> Dict:
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = _model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().tolist()[0]
    return {"classes": ["non_tumor", "tumor"], "probs": probs}
