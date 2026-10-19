"""Pipeline ALPR: YOLOv8s phát hiện biển số -> cắt vùng -> CNN+BiLSTM+CTC đọc ký tự."""
from dataclasses import dataclass

import cv2
import numpy as np

import config
from core import plate_utils


@dataclass
class PlateResult:
    box: tuple            # (x1, y1, x2, y2)
    det_conf: float
    raw_text: str         # chuỗi OCR gốc
    plate: str            # chuỗi đã chuẩn hoá (vd 29A12345)
    ocr_conf: float
    valid: bool           # đúng định dạng biển số VN
    crop: np.ndarray


def split_two_line(crop):
    """Biển 2 dòng (xe máy / biển vuông): tách dòng trên & dưới rồi ghép ngang thành 1 dòng.
    Chỉ dùng khi mô hình OCR được huấn luyện trên ảnh 1 dòng; mặc định KHÔNG dùng vì
    bộ dữ liệu trong báo cáo có cả biển 2 dòng nguyên khối."""
    h = crop.shape[0]
    top, bot = crop[: h // 2], crop[h // 2:]
    bot = cv2.resize(bot, (top.shape[1], top.shape[0]))
    return np.hstack([top, bot])


def suppress_nested(boxes, thr=0.6):
    """Bỏ khung nằm phần lớn bên trong một khung lớn hơn (YOLO đôi khi khoanh thêm 1 phần của chính biển số đó,
    vd chỉ cụm '56'). NMS thường của YOLO không loại được vì IoU giữa khung nhỏ và khung lớn thấp."""
    area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    keep = []
    for b, c in sorted(boxes, key=lambda x: area(x[0]), reverse=True):     # khung lớn xét trước
        inside = False
        for k, _ in keep:
            iw = max(0, min(b[2], k[2]) - max(b[0], k[0]))
            ih = max(0, min(b[3], k[3]) - max(b[1], k[1]))
            if area(b) and iw * ih / area(b) > thr:
                inside = True
                break
        if not inside:
            keep.append((b, c))
    return keep


class ALPRPipeline:
    def __init__(self, yolo_weights=config.YOLO_WEIGHTS, ocr_weights=config.OCR_WEIGHTS):
        from ultralytics import YOLO
        from core.ocr_model import PlateOCR
        self.detector = YOLO(yolo_weights)
        self.ocr = PlateOCR(ocr_weights)

    def detect(self, frame_bgr):
        res = self.detector.predict(frame_bgr, conf=config.DET_CONF, verbose=False)[0]
        h, w = frame_bgr.shape[:2]
        boxes = []
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            px, py = (x2 - x1) * config.DET_PAD_RATIO, (y2 - y1) * config.DET_PAD_RATIO
            boxes.append(((int(max(0, x1 - px)), int(max(0, y1 - py)),
                           int(min(w, x2 + px)), int(min(h, y2 + py))), float(c)))
        return suppress_nested(boxes)

    def read_crop(self, crop):
        raw, conf = self.ocr.read(crop)
        plate = plate_utils.normalize(raw)
        return raw, plate, conf, plate_utils.is_valid(plate)

    def process(self, frame_bgr):
        results = []
        for (x1, y1, x2, y2), dconf in self.detect(frame_bgr):
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            raw, plate, oconf, valid = self.read_crop(crop)
            results.append(PlateResult((x1, y1, x2, y2), dconf, raw, plate, oconf, valid, crop))
        return results


def draw(frame_bgr, results):
    out = frame_bgr.copy()
    for r in results:
        x1, y1, x2, y2 = r.box
        color = (0, 200, 0) if r.valid else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = plate_utils.format_plate(r.plate) or "?"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out