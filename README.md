# 🫀 Aortic Valve Detection — 2025 AI CUP

This repository provides the full pipeline used for the **2025 AI CUP – Aortic Valve Detection Task**, including:

* **Data preprocessing**
* **YOLOv12 training**
* **DINO 4-scale & 5-scale training**
* **Multi-model ensembling using Weighted Box Fusion (WBF)**

This README offers a complete, step-by-step reproducibility guide.

# 📦 1. Environment Setup

## Install dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
# ---------- YOLOv12 ----------
ultralytics==8.3.34
opencv-python
PyYAML
numpy
tqdm

# ---------- DINO / DETR ----------
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.2
einops
pycocotools
transformers

# ---------- Optional: Weighted Boxes Fusion ----------
ensemble-boxes
```

### Recommended environment

| Component | Version                       |
| --------- | ----------------------------- |
| Python    | 3.10–3.11                     |
| CUDA      | 12.1 / 12.4                   |
| GPU       | NVIDIA RTX 4090 / 5090 tested |
| OS        | Ubuntu 22.04 / 24.04          |

---

# 📁 2. Data Preprocessing

Before training YOLO or DINO, run the preprocessing scripts to organize training/validation data into YOLO-format folders.

Scripts are located in:

```
preprocess/
    make_train.py
    make_val.py
```

## Generate training dataset

```bash
python preprocess/make_train.py
```

This script will:

* Parse the raw dataset
* Convert annotations into YOLO format
* Produce:

```
training_image/
training_label/
```

## Generate validation dataset

```bash
python preprocess/make_val.py
```

This creates:

```
val_image/
val_label/
```

## Notes

* YOLO and DINO **share the same preprocessed data**.
* If your dataset is already YOLO-ready, YOLO training can skip preprocessing via `--skip-organize`.
* Preprocessing ensures filename consistency, correct image–label pairing, and standardized annotation format.

# 🟦 3. YOLOv12 Training

Training script:

```
yolo/train_aortic_valve_local.py
```

Run:

```bash
python yolo/train_aortic_valve_local.py \
    --img-root ./training_image \
    --lbl-root ./training_label \
    --skip-organize \
    --model-size x \
    --epochs 500 \
    --batch-size 8 \
    --img-size 512 \
    --device 0 \
    --patience 50 \
    --save-period 10 \
    --cache \
    --workers 16 \
    --name YCtrain
```

### Key parameters

| Argument           | Description                   |
| ------------------ | ----------------------------- |
| `--img-root`       | Path to training images       |
| `--lbl-root`       | Path to YOLO label files      |
| `--skip-organize`  | Skip additional preprocessing |
| `--model-size x`   | Use YOLOv12x backbone         |
| `--epochs 500`     | Training epochs               |
| `--batch-size 8`   | Batch size                    |
| `--img-size 512`   | Input resolution              |
| `--patience 50`    | Early stopping                |
| `--save-period 10` | Save checkpoints periodically |

Output directory:

```
runs/detect/YCtrain/
```

---

# 🟧 4. DINO Training (4-scale & 5-scale)

DINO training is performed via notebooks instead of a python script.

Notebooks:

```
DINO/train_4scale.ipynb
DINO/train_5scale.ipynb
```

## How to train DINO models

1. Open the notebook in Jupyter / VS Code
2. Set dataset paths (training + validation)
3. Run cells sequentially
4. Export prediction txt files to `predict_txt/`

Example outputs:

```
predict_txt/predictions_4scale_15.txt
predict_txt/predictions_15.txt
```

Each file uses:

```
img_name class score x1 y1 x2 y2   # normalized coordinates
```

---

# 🔷 5. Weighted Box Fusion (WBF) Ensemble

After YOLOv12 and DINO inference, merge all prediction txt files using WBF.

Run:

```bash
python wbf_ensemble.py \
    --inputs \
        ./predict_txt/predictions_4scale_15.txt \
        ./predict_txt/images_34.txt \
        ./predict_txt/predictions_15.txt \
    --output ./predict_txt/images_56.txt \
    --img-width 512 \
    --img-height 512 \
    --iou-thr 0.5 \
    --conf-thr 0.15 \
    --skip-box-thr 0.01
```

### Parameters

| Argument                       | Description                             |
| ------------------------------ | --------------------------------------- |
| `--inputs`                     | List of input txt files                 |
| `--output`                     | Output ensemble txt                     |
| `--img-width` / `--img-height` | Original image resolution               |
| `--iou-thr`                    | IoU threshold for merging boxes         |
| `--conf-thr`                   | Remove boxes below confidence threshold |
| `--skip-box-thr`               | Ignore extremely low-confidence boxes   |

Final prediction file example:

```
predict_txt/images_56.txt
```

# 🔁 6. Complete Reproducibility Pipeline

### Step 1 — Preprocess data

```bash
python preprocess/make_train.py
python preprocess/make_val.py
```

### Step 2 — Train YOLOv12

Produces YOLO txt prediction files.

### Step 3 — Train DINO (4-scale & 5-scale)

Produces two additional prediction files.

### Step 4 — Run inference for each model

Store all txt results in `predict_txt/`.

### Step 5 — Apply WBF ensemble

Combines predictions into final output.

# 📂 7. Repository Structure

```
2025-AICUP_Aortic_Valve/
│
├── yolo/
│   └── train_aortic_valve_local.py
│
├── DINO/
│   ├── train_4scale.ipynb
│   └── train_5scale.ipynb
│
├── preprocess/
│   ├── make_train.py
│   └── make_val.py
│
├── predict_txt/
│   └── (YOLO, DINO, and ensemble outputs)
│
├── wbf_ensemble.py
│
└── README.md
```
