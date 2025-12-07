import os
import shutil
import random
from pathlib import Path

# ========================
# 基本設定
# ========================
ROOT = Path(".")  # 你可以改成實際路徑，例如 Path("/home/adl/Desktop/2025_AI_CUP")
IMAGES_ROOT = ROOT / "training_image"
LABELS_ROOT = ROOT / "training_label"

TRAIN_IMAGES = ROOT / "train" / "images"
TRAIN_LABELS = ROOT / "train" / "labels"

# 要排除的病人編號（這些是給 val 用的）
EXCLUDE_PATIENT_IDS = [8, 9, 11, 28, 29, 30, 32]
EXCLUDE_PATIENT_NAMES = {f"patient{pid:04d}" for pid in EXCLUDE_PATIENT_IDS}

# positive : negative = 1 : 3
NEG_PER_POS = 3

# 允許的影像副檔名
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

# 隨機種子（可固定方便重現）
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ========================
# 工具函式
# ========================

def find_image_for_stem(img_dir: Path, stem: str):
    """在 img_dir 中，找檔名（不含副檔名）等於 stem 的圖片，回傳 Path 或 None。"""
    for ext in IMG_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ========================
# 主流程
# ========================

def main():
    ensure_dir(TRAIN_IMAGES)
    ensure_dir(TRAIN_LABELS)

    total_pos = 0
    total_neg = 0

    # 掃描所有病人資料夾，排除 EXCLUDE_PATIENT_NAMES
    patient_dirs = sorted(
        [p for p in IMAGES_ROOT.iterdir() if p.is_dir() and p.name.startswith("patient")]
    )

    patients_to_use = [
        p.name for p in patient_dirs
        if p.name not in EXCLUDE_PATIENT_NAMES
    ]

    print("[INFO] 將處理以下病人（作為 train）：")
    print(", ".join(patients_to_use))

    for patient in patients_to_use:
        img_dir = IMAGES_ROOT / patient
        lbl_dir = LABELS_ROOT / patient

        if not img_dir.exists():
            print(f"[WARN] 圖片資料夾不存在：{img_dir}")
            continue
        if not lbl_dir.exists():
            print(f"[WARN] 標籤資料夾不存在：{lbl_dir}")
            continue

        # 1. 找出所有 positive：有對應 txt 的影像
        positive_pairs = []  # (img_path, label_path)
        positive_stems = set()

        for txt_file in lbl_dir.glob("*.txt"):
            stem = txt_file.stem  # e.g. patient0001_0001
            img_path = find_image_for_stem(img_dir, stem)
            if img_path is None:
                print(f"[WARN] 找不到對應圖片：{stem} 在 {img_dir}")
                continue

            positive_pairs.append((img_path, txt_file))
            positive_stems.add(stem)

        num_pos = len(positive_pairs)
        if num_pos == 0:
            print(f"[INFO] {patient} 沒有 positive，略過")
            continue

        # 2. 找出 negative：這個病人裡，所有沒有 label 的圖片
        all_images = [
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        negative_images = [
            p for p in all_images
            if p.stem not in positive_stems
        ]

        if len(negative_images) == 0:
            print(f"[WARN] {patient} 沒有可用的 negative 圖片")
            continue

        # 3. 根據 1:5 比例抽 negative
        target_neg = num_pos * NEG_PER_POS
        if len(negative_images) < target_neg:
            print(
                f"[WARN] {patient} negative 不足，"
                f"需要 {target_neg} 張，只能用 {len(negative_images)} 張"
            )
            chosen_negatives = negative_images
        else:
            chosen_negatives = random.sample(negative_images, target_neg)

        # 4. 複製 positive 到 train/images & train/labels
        for img_path, lbl_path in positive_pairs:
            dst_img = TRAIN_IMAGES / img_path.name
            dst_lbl = TRAIN_LABELS / lbl_path.name

            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)

        # 5. 複製 negative 到 train/images，並建立空的 label 到 train/labels
        for img_path in chosen_negatives:
            dst_img = TRAIN_IMAGES / img_path.name
            dst_lbl = TRAIN_LABELS / f"{img_path.stem}.txt"

            shutil.copy2(img_path, dst_img)

            # 建立空 label 檔案代表沒有物件
            if not dst_lbl.exists():
                dst_lbl.write_text("")

        print(
            f"[DONE] {patient}: positive {num_pos} 張，"
            f"negative 選了 {len(chosen_negatives)} 張"
        )

        total_pos += num_pos
        total_neg += len(chosen_negatives)

    print("=" * 50)
    print(f"總計複製到 train 的 positive: {total_pos} 張")
    print(f"總計複製到 train 的 negative: {total_neg} 張")
    print("輸出資料夾：")
    print(f"  train/images -> {TRAIN_IMAGES.resolve()}")
    print(f"  train/labels -> {TRAIN_LABELS.resolve()}")


if __name__ == "__main__":
    main()
