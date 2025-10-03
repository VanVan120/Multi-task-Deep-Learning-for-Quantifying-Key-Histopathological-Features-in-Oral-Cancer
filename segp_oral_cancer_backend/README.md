# SEGP Oral Cancer Backend

Minimal backend/model scaffold for oral-cancer digital pathology:
- TVNT baseline (tumor vs non-tumor) with ResNet18 (PyTorch)
- Evaluation (AUC, F1), TensorBoard logs
- FastAPI inference endpoint
- Basic WSI patch extractor (OpenSlide)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train (expects ImageFolder-like data)
# data/patches/train/{tumor,non_tumor}/*.png
# data/patches/val/{tumor,non_tumor}/*.png
python -m src.training.train_tvnt --config configs/tvnt.yaml

# Evaluate on test split
python -m src.training.evaluate_tvnt --config configs/tvnt.yaml

# Run API
uvicorn src.api.main:app --reload --port 8000
```

## U-Net Segmentation Pipeline

### 1. Prepare the dataset directories
```
python scripts/prepare_unet_dataset.py \
  --src "E:\path\to\raw_cases" \
  --dst "E:\Nottingham\Year2\Autumn\Software-Engineering-Group-Project\data\dataset_seg" \
  --dry-run
```
- `--src` must contain per-case folders with matching JPG images and PNG masks (mask names may include `[x=..,y=..,w=..,h=..]`).
- `--dst` is the root that will receive `train/val/test/images|masks`. Use `--train-cases/--val-cases/--test-cases` to control case-level splits, `--only-cases` to filter a subset, and `--by-case` to keep case subfolders.
- Remove `--dry-run` once the preview looks correct; a `report.csv` will summarize pairing issues.

### 2. Train the baseline U-Net (CPU-only by default)
```
python -m src.training.run_unet_baseline \
  --data-root "E:\Nottingham\Year2\Autumn\Software-Engineering-Group-Project\data\dataset_seg" \
  --epochs 40 --img-size 256 --batch-size 8 --lr 1e-3 \
  --outdir runs/unet_baseline
```
- The script expects the prepared layout (`<data-root>/<split>/images|masks`).
- Loss combines class-weighted cross-entropy (foreground weight 3.0) and soft Dice. Tune `--base-ch`, learning rate, or epochs as needed.
- Training logs append to `runs/unet_baseline/log.csv`; `best.ckpt` stores the best validation Dice model.

### 3. Run inference and export mask overlays
```
python -m src.inference.run_unet_infer \
  --data-root "E:\Nottingham\Year2\Autumn\Software-Engineering-Group-Project\data\dataset_seg" \
  --ckpt runs/unet_baseline/best.ckpt \
  --split val \
  --save-dir outputs/unet_val_pred \
  --thr 0.4 --save-probmaps
```
- The script loads case tiles from the selected split, writes binary masks to `save-dir/masks`, red overlays to `save-dir/overlays`, and optional probability heatmaps to `save-dir/probmaps`.
- Adjust `--thr` when masks are too sparse/dense; the probability maps help pick a sensible threshold.
- The command accepts checkpoints saved as either pure `state_dict` files or dictionaries containing `state_dict`.

## Config
See `configs/tvnt.yaml`. Change paths, hyperparams, and checkpoint output.

## Data
- If you have WSIs, use `scripts/extract_patches.py` to grid-sample patches.
- For a quick smoke test, copy any small images into the ImageFolder layout.

## Full Dataset (256px) Example
```
# Train with the converted dataset
python -m src.training.run_unet_baseline \
  --data-root "..\\Oral Cancer.v2i.yolov11\\dataset_seg_full_256png" \
  --epochs 10 --img-size 256 --batch-size 8 --lr 1e-3 \
  --outdir runs/unet_dataset256

# Run validation inference and export overlays
python -m src.inference.run_unet_infer \
  --data-root "..\\Oral Cancer.v2i.yolov11\\dataset_seg_full_256png" \
  --ckpt runs/unet_dataset256/best.ckpt \
  --split val \
  --save-dir outputs/unet_dataset256_val \
  --thr 0.4 --save-probmaps
```

## API
POST `/predict/tvnt` with `multipart/form-data` field `file` (image). Returns class probabilities.
