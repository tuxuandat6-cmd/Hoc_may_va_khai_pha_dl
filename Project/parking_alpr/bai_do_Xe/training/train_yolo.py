"""Huấn luyện YOLOv8s phát hiện biển số (theo cấu hình trong báo cáo: 640x640, batch 16, patience 20).

Dữ liệu mặc định: bộ biển số Việt Nam công khai trên Roboflow Universe
  computer-vison-4zhos / license-plate-recognition-kvr7y  (v2: 3.132 ảnh, 1 lớp, giấy phép CC BY 4.0)
Cần API key MIỄN PHÍ của Roboflow: đăng ký tại roboflow.com -> Settings -> API Keys.

Chạy:
  python training/train_yolo.py --roboflow-key XXXXX --epochs 60
  python training/train_yolo.py --data path/to/data.yaml --epochs 60   # dùng bộ dữ liệu YOLO có sẵn (vd Kaggle)
Kết quả: weights/yolov8s_plate.pt
"""
import argparse
import os
import shutil

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RF_WORKSPACE, RF_PROJECT, RF_VERSION = "computer-vison-4zhos", "license-plate-recognition-kvr7y", 2


def download_roboflow(key, dest):
    from roboflow import Roboflow
    os.makedirs(dest, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(dest)
    try:
        ds = Roboflow(api_key=key).workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION).download("yolov8")
        return os.path.join(os.path.abspath(ds.location), "data.yaml")
    finally:
        os.chdir(cwd)


def fix_yaml(path):
    """data.yaml của Roboflow dùng đường dẫn '../train/images' hay bị lệch -> ghi lại đường dẫn tuyệt đối."""
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        p = os.path.join(base, folder, "images")
        if os.path.isdir(p):
            d[split] = p
    d.pop("path", None)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True)
    print("data.yaml:", d)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--data", default=None, help="data.yaml có sẵn (bỏ qua bước tải Roboflow)")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "weights", "yolov8s_plate.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO

    data = args.data
    if not data:
        if not args.roboflow_key:
            raise SystemExit("Cần --roboflow-key (miễn phí tại roboflow.com) hoặc --data path/to/data.yaml")
        data = download_roboflow(args.roboflow_key, os.path.join(HERE, "datasets"))
    data = fix_yaml(data)

    model = YOLO(args.model)
    model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
                project=os.path.join(HERE, "runs"), name="plate_det", exist_ok=True,
                optimizer="AdamW", lr0=0.002, cos_lr=True, plots=True)
    best = os.path.join(HERE, "runs", "plate_det", "weights", "best.pt")

    print("\n=== Đánh giá trên tập TEST ===")
    m = YOLO(best).val(data=data, split="test" if "test" in yaml.safe_load(open(data)) else "val",
                       imgsz=args.imgsz, plots=True, project=os.path.join(HERE, "runs"), name="plate_test",
                       exist_ok=True)
    print(f"Precision {m.box.mp:.3f} | Recall {m.box.mr:.3f} | mAP@50 {m.box.map50:.3f} | mAP@50-95 {m.box.map:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shutil.copy(best, args.out)
    print("Đã lưu:", args.out)


if __name__ == "__main__":
    main()
