# U-Net Dataset Preparation

## Source Input Structure
- Raw cases live under folders such as `.../test 256px jpg/<case_id>/`.
- Each case directory contains paired image and mask files, e.g. `lesion_01.jpg` and `lesion_01 [x=10,y=25,w=120,h=90].png`.
- Mask filenames may include the bounding box suffix (`[x=..,y=..,w=..,h=..]`); the script strips this when matching to the corresponding JPG.

## Target Dataset Layout
```
data/dataset_seg/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```
- When `--by-case` is used, an extra level is inserted: `train/<case_id>/images|masks/` (and likewise for other splits).
- Masks are rewritten as binary PNGs where every non-zero pixel becomes 255.
- Train/val/test membership is resolved at the **case** level; list each case in exactly one split for reproducible results.
- A `report.csv` is emitted in the destination root summarising pairs, missing images, and missing masks per case.

## Example PowerShell Commands
```powershell
# Preview actions without writing any files
python .\scripts\prepare_unet_dataset.py \`
  --src "E:\Data\oral_cancer\test 256px jpg" \`
  --dst "E:\Data\oral_cancer\data\dataset_seg" \`
  --dry-run

# Full run with explicit split lists
python .\scripts\prepare_unet_dataset.py \`
  --src "E:\Data\oral_cancer\test 256px jpg" \`
  --dst "E:\Data\oral_cancer\data\dataset_seg" \`
  --train-cases "case001,case002,case003" \`
  --val-cases "case004" \`
  --test-cases "case005,case006"

# Case-organised export for easier inspection
python .\scripts\prepare_unet_dataset.py \`
  --src "E:\Data\oral_cancer\test 256px jpg" \`
  --dst "E:\Data\oral_cancer\data\dataset_seg" \`
  --train-cases "case001,case002" \`
  --val-cases "case003" \`
  --by-case
```
