## Replication

### Environment Setup

The models were trained in Python 3.12 environments.

1.  **Download Data and Checkpoints:**
    ```bash
    bash scripts/download.sh
    ```

2.  **Prepare COCO Format Data:**
    ```bash
    python tools/prepare_valve_coco.py
    ```

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    if not on colab or kaggle, run 
    ```
    !pip install "setuptools<70" wheel Cython numpy
    !pip install -r requirements.txt --no-build-isolation
    ```

4.  **Compile CUDA Operators:**
    ```bash
    bash scripts/setup.sh
    ```

### Training

#### 4-scale Model (Kaggle)
This model was trained on Kaggle using T4 GPUs.

```bash
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --output_dir logs/4scalekaggle \
  -c config/DINO/DINO_valve_4scale.py \
  --coco_path data \
  --pretrain_model_path ckpt/checkpoint0033_4scale.pth \
  --finetune_ignore label_enc.weight class_embed \
  --options \
    dn_scalar=100 \
    embed_init_tgt=TRUE \
    dn_label_coef=1.0 \
    dn_bbox_coef=1.0 \
    dn_box_noise_scale=1.0 \
    use_ema=True \
    ema_epoch=0 \
    epochs=24 \
    lr_drop=20 \
    batch_size=8
```

#### 5-scale Model (Google Colab)
This model was trained on Google Colab using an A100 GPU.

```bash
python main.py \
  --output_dir /content/output \
  -c config/DINO/DINO_valve_5scale.py \
  --coco_path data \
  --pretrain_model_path ckpt/checkpoint0031_5scale.pth \
  --finetune_ignore label_enc.weight class_embed \
  --options \
    dn_scalar=100 \
    embed_init_tgt=TRUE \
    dn_label_coef=1.0 \
    dn_bbox_coef=1.0 \
    dn_box_noise_scale=1.0 \
    use_ema=True \
    ema_epoch=0 \
    epochs=24 \
    lr_drop=20 \
    batch_size=16
```

### Inference

To perform inference from checkpoints, run `inference_replication.ipynb`, which handles data and checkpoint download and inference.