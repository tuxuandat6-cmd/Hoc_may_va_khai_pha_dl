"""Tạo bộ ảnh biển số THẬT để huấn luyện lại OCR (tăng độ chính xác trên ảnh camera).

Dùng YOLOv8 đã huấn luyện để cắt biển số từ một thư mục ảnh xe, lưu ảnh crop và file labels.csv
với nhãn ĐIỀN SẴN bằng dự đoán của OCR hiện tại. Bạn chỉ cần mở labels.csv (Excel) sửa các nhãn sai
và xoá dòng ảnh không đọc được, rồi huấn luyện lại:
    python training/train_ocr.py --real-dir real_crops --epochs 40

Chạy:
    python training/make_crops.py --images duong_dan_thu_muc_anh --out real_crops
"""
import argparse
import csv
import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="real_crops")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    import config
    from core.pipeline import ALPRPipeline
    has_ocr = os.path.exists(config.OCR_WEIGHTS)
    if has_ocr:
        pipe = ALPRPipeline()
    else:
        from ultralytics import YOLO
        from core.pipeline import ALPRPipeline as _P
        pipe = _P.__new__(_P)
        pipe.detector = YOLO(config.YOLO_WEIGHTS)

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(p for ext in ("jpg", "jpeg", "png", "bmp") for p in
                   glob.glob(os.path.join(args.images, "**", f"*.{ext}"), recursive=True))[: args.limit]
    rows = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        for j, ((x1, y1, x2, y2), _) in enumerate(pipe.detect(img)):
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 12:
                continue
            name = f"{i:05d}_{j}.jpg"
            cv2.imwrite(os.path.join(args.out, name), crop)
            guess = pipe.read_crop(crop)[1] if has_ocr else ""
            rows.append((name, guess))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(paths)} ảnh, {len(rows)} biển số")
    with open(os.path.join(args.out, "labels.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)
    print(f"Đã lưu {len(rows)} ảnh crop vào {args.out}/ — hãy kiểm tra và sửa labels.csv trước khi huấn luyện.")


if __name__ == "__main__":
    main()
