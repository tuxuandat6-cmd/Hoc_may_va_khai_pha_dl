"""Xử lý video: theo dõi biển số giữa các frame (IoU) + bỏ phiếu chuỗi OCR để ra kết quả ổn định,
mỗi xe chỉ được ghi nhận MỘT lần dù xuất hiện trong hàng trăm frame."""
from collections import Counter

import cv2

import config
from core.pipeline import draw


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


class Track:
    def __init__(self, r):
        self.box, self.votes, self.gap, self.done = r.box, Counter(), 0, False
        self.best_crop, self.best_conf = None, -1
        self.add(r)

    def add(self, r):
        self.box, self.gap = r.box, 0
        if r.valid:
            self.votes[r.plate] += 1
            if r.ocr_conf > self.best_conf:
                self.best_conf, self.best_crop = r.ocr_conf, r.crop

    def winner(self):
        if not self.votes:
            return None, 0
        return self.votes.most_common(1)[0]


def process_video(path, pipeline, on_confirm, direction="auto", progress=None):
    """Duyệt video, gọi on_confirm(plate, conf, crop, frame) khi một xe được xác nhận.
    Trả về đường dẫn video kết quả (có vẽ bounding box) và danh sách log."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = path.rsplit(".", 1)[0] + "_annotated.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / config.VIDEO_FRAME_STEP, (w, h))

    tracks, logs, idx, last = [], [], 0, None

    def finalize(t, frame):
        plate, n = t.winner()
        if plate and n >= config.VIDEO_MIN_VOTES and not t.done:
            t.done = True
            logs.append(on_confirm(plate, t.best_conf, t.best_crop, frame))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        last = frame
        if idx % config.VIDEO_FRAME_STEP:
            continue
        if progress:
            progress(idx / total, desc=f"Đang xử lý frame {idx}/{total}")

        results = pipeline.process(frame)
        matched = set()
        for r in results:
            best = max(tracks, key=lambda t: iou(t.box, r.box), default=None)
            if best is not None and id(best) not in matched and iou(best.box, r.box) >= config.TRACK_IOU:
                best.add(r)
                matched.add(id(best))
            else:
                t = Track(r)
                tracks.append(t)
                matched.add(id(t))

        for t in tracks:
            if id(t) not in matched:
                t.gap += 1
            # xác nhận sớm khi đã đủ phiếu -> phản hồi nhanh như barrier thật
            finalize(t, frame)
        tracks = [t for t in tracks if t.gap <= config.TRACK_MAX_GAP]
        writer.write(draw(frame, results))

    for t in tracks:                 # xe còn trong khung hình ở cuối video
        finalize(t, last)
    cap.release()
    writer.release()
    return out_path, logs
