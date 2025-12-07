#!/usr/bin/env python3
"""
WBF Ensemble for detection txt files

Input format (每個 txt):
Img_name class score x1 y1 x2 y2

Usage example:
python wbf_ensemble.py \
    --inputs pred_yolo11.txt pred_yolo12.txt pred_rtdetr.txt \
    --output ensemble.txt \
    --img-width 512 \
    --img-height 512 \
    --iou-thr 0.5 \
    --skip-box-thr 0.0 \
    --conf-thr 0.0
"""

import argparse
from collections import defaultdict
from typing import List, Dict, Any

from ensemble_boxes import weighted_boxes_fusion


def parse_args():
    parser = argparse.ArgumentParser(description="WBF ensemble for detection txt files")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of input prediction txt files (each = one model / TTA)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output txt file path",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        required=True,
        help="Image width in pixels (e.g., 512)",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        required=True,
        help="Image height in pixels (e.g., 512)",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold for WBF (default: 0.5)",
    )
    parser.add_argument(
        "--skip-box-thr",
        type=float,
        default=0.0,
        help="Drop input boxes with score < skip_box_thr BEFORE WBF (default: 0.0)",
    )
    parser.add_argument(
        "--conf-thr",
        type=float,
        default=0.25,
        help="Drop output boxes with score < conf_thr AFTER WBF (default: 0.0)",
    )
    return parser.parse_args()


def read_predictions(
    file_paths: List[str],
    img_width: int,
    img_height: int,
) -> Dict[str, Dict[str, Any]]:
    """
    讀多個 txt，整理成 per_image 結構：
    per_image[img_name] = {
        'boxes_list': [model1_boxes, model2_boxes, ...],
        'scores_list': [...],
        'labels_list': [...]
    }
    其中 boxes 是 normalized [x1, y1, x2, y2] ∈ [0,1]
    """
    num_models = len(file_paths)

    per_image: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "boxes_list": [[] for _ in range(num_models)],
            "scores_list": [[] for _ in range(num_models)],
            "labels_list": [[] for _ in range(num_models)],
        }
    )

    for model_idx, path in enumerate(file_paths):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 7:
                    # 格式不符就跳過
                    continue

                img_name, cls_str, score_str, x1_str, y1_str, x2_str, y2_str = parts
                try:
                    label = int(cls_str)
                    score = float(score_str)
                    x1 = float(x1_str)
                    y1 = float(y1_str)
                    x2 = float(x2_str)
                    y2 = float(y2_str)
                except ValueError:
                    # 有壞行就跳過
                    continue

                # normalize to [0, 1]
                nx1 = x1 / img_width
                ny1 = y1 / img_height
                nx2 = x2 / img_width
                ny2 = y2 / img_height

                # 簡單保險一下
                nx1 = max(0.0, min(1.0, nx1))
                ny1 = max(0.0, min(1.0, ny1))
                nx2 = max(0.0, min(1.0, nx2))
                ny2 = max(0.0, min(1.0, ny2))

                per_image[img_name]["boxes_list"][model_idx].append([nx1, ny1, nx2, ny2])
                per_image[img_name]["scores_list"][model_idx].append(score)
                per_image[img_name]["labels_list"][model_idx].append(label)

    return per_image


def run_wbf_for_all_images(
    per_image: Dict[str, Dict[str, Any]],
    img_width: int,
    img_height: int,
    iou_thr: float,
    skip_box_thr: float,
    conf_thr: float,
) -> List[str]:
    """
    對每張圖做 WBF，回傳輸出 txt 的每一行字串
    確保同一張圖內，confidence 由大到小排序
    """
    output_lines: List[str] = []

    for img_name in sorted(per_image.keys()):
        data = per_image[img_name]
        boxes_list = data["boxes_list"]
        scores_list = data["scores_list"]
        labels_list = data["labels_list"]

        # 檢查這張圖有沒有任何 box
        total_boxes = sum(len(b) for b in boxes_list)
        if total_boxes == 0:
            # 沒框就略過（或視需求寫一行空的）
            continue

        # weighted boxes fusion
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=None,          # 每個模型同權重，如要調整可改這裡
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        # 先把這張圖的 box 收集起來（過 conf_thr），再依 score 排序
        per_image_records = []
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            if score < conf_thr:
                continue
            per_image_records.append((score, label, box))

        # 依 confidence 由大到小排序
        per_image_records.sort(key=lambda x: x[0], reverse=True)

        # 再寫回輸出（這樣同張圖內就會是高分在上）
        for score, label, box in per_image_records:
            nx1, ny1, nx2, ny2 = box

            # 反 normalize 回 pixel
            x1 = int(round(nx1 * img_width))
            y1 = int(round(ny1 * img_height))
            x2 = int(round(nx2 * img_width))
            y2 = int(round(ny2 * img_height))

            # 簡單做一下 clipping
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            x2 = max(0, min(img_width - 1, x2))
            y2 = max(0, min(img_height - 1, y2))

            line = f"{img_name} {int(label)} {score:.4f} {x1} {y1} {x2} {y2}"
            output_lines.append(line)

    return output_lines


def main():
    args = parse_args()

    per_image = read_predictions(
        args.inputs,
        img_width=args.img_width,
        img_height=args.img_height,
    )

    output_lines = run_wbf_for_all_images(
        per_image,
        img_width=args.img_width,
        img_height=args.img_height,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_box_thr,
        conf_thr=args.conf_thr,
    )

    with open(args.output, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"Done! WBF result saved to: {args.output}")
    print(f"Total lines: {len(output_lines)}")


if __name__ == "__main__":
    main()
