"""Mô hình OCR CNN + BiLSTM + CTC (theo Chương II.3.4 / III.2 của báo cáo).

Kiến trúc:
  Ảnh xám 64x256x1
  -> Block1: Conv(64)  + BN + ReLU + MaxPool(2,2)
  -> Block2: Conv(128) + BN + ReLU + MaxPool(2,2)
  -> Block3: 2x[Conv(256) + BN + ReLU] + MaxPool(2,1)
  -> Block4: 2x[Conv(512) + BN + ReLU] + MaxPool(2,1)
  -> SEBlock(512) -> AdaptiveAvgPool(1, W) -> (T, B, 512)
  -> BiLSTM 2 lớp, hidden 256, dropout 0.3 -> Dropout 0.3 -> Linear(512 -> 37)

LƯU Ý: tên lớp / thứ tự layer phải khớp với code huấn luyện thì state_dict mới load được.
Nếu nhóm đã đặt tên khác, hãy dán class CRNN gốc vào file này thay cho class bên dưới
(giữ nguyên tên `CRNN` và hàm `forward` trả về (T, B, C) log-probs hoặc logits).
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

import config


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


def _conv(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    def __init__(self, num_classes=37, hidden=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            *_conv(1, 64), nn.MaxPool2d(2, 2),
            *_conv(64, 128), nn.MaxPool2d(2, 2),
            *_conv(128, 256), *_conv(256, 256), nn.MaxPool2d((2, 1)),
            *_conv(256, 512), *_conv(512, 512), nn.MaxPool2d((2, 1)),
        )
        self.se = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(512, hidden, num_layers=num_layers, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):                      # x: (B, 1, 64, 256)
        f = self.pool(self.se(self.cnn(x)))     # (B, 512, 1, W')
        f = f.squeeze(2).permute(2, 0, 1)       # (T, B, 512)
        out, _ = self.rnn(f)
        return self.fc(self.dropout(out))       # (T, B, C)


def to_one_line(img):
    """Biển 2 dòng (tỷ lệ rộng/cao < 2.5): cắt nửa trên & nửa dưới (chồng lấn 10% để không cắt mất nét chữ)
    rồi ghép ngang thành 1 dòng, để CRNN đọc tuần tự trái -> phải đúng thứ tự dòng trên rồi dòng dưới."""
    h, w = img.shape[:2]
    if h == 0 or w / h >= 2.5:
        return img
    top, bot = img[: int(h * 0.55)], img[int(h * 0.45):]
    bot = cv2.resize(bot, (top.shape[1], top.shape[0]))
    return np.hstack([top, bot])


def preprocess_plate(img_bgr, two_line_split=True, normalize="0_1"):
    """Tiền xử lý DÙNG CHUNG cho huấn luyện và suy luận: (ghép 2 dòng) -> xám -> 256x64 -> chuẩn hoá."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if two_line_split:
        gray = to_one_line(gray)
    img = cv2.resize(gray, (config.OCR_IMG_W, config.OCR_IMG_H)).astype(np.float32) / 255.0
    if normalize == "minus1_1":
        img = (img - 0.5) / 0.5
    return img


class PlateOCR:
    """Bọc mô hình: tiền xử lý ảnh biển số -> CRNN -> greedy CTC decode."""

    def __init__(self, weights=config.OCR_WEIGHTS, charset=config.CHARSET, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights, map_location=self.device, weights_only=False)

        # checkpoint do training/train_ocr.py tạo có cờ two_line_split; checkpoint khác mặc định không tách dòng
        self.two_line_split = bool(isinstance(ckpt, dict) and ckpt.get("two_line_split", False))
        self.normalize = ckpt.get("normalize", config.OCR_NORMALIZE) if isinstance(ckpt, dict) else config.OCR_NORMALIZE
        if isinstance(ckpt, nn.Module):         # lưu cả model bằng torch.save(model)
            self.model = ckpt
        else:
            state = ckpt
            if isinstance(ckpt, dict):
                charset = ckpt.get("charset", charset)
                for key in ("model_state_dict", "state_dict", "model"):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        state = ckpt[key]
                        break
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            fc_w = [v for k, v in state.items() if k.endswith("fc.weight")]
            num_classes = fc_w[-1].shape[0] if fc_w else len(charset) + 1
            self.model = CRNN(num_classes=num_classes)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Checkpoint OCR không khớp kiến trúc CRNN trong core/ocr_model.py.\n"
                    f"Thiếu: {missing[:5]}...\nThừa: {unexpected[:5]}...\n"
                    "=> Hãy dán class mô hình gốc từ notebook huấn luyện vào core/ocr_model.py.")
            if num_classes != len(charset) + 1:
                print(f"[CẢNH BÁO] Mô hình có {num_classes} lớp nhưng CHARSET có {len(charset)} ký tự "
                      f"(+1 blank = {len(charset) + 1}). Kết quả OCR sẽ sai lệch — sửa CHARSET trong config.py.")

        self.charset = charset
        self.model.to(self.device).eval()

    def preprocess(self, crop_bgr):
        img = preprocess_plate(crop_bgr, self.two_line_split, self.normalize)
        return torch.from_numpy(img)[None, None].to(self.device)

    def decode(self, logits):
        """Greedy CTC: lấy argmax, gộp ký tự lặp, bỏ blank (index 0)."""
        probs = logits.softmax(-1)[:, 0]        # (T, C)
        conf, idx = probs.max(-1)
        chars, confs, prev = [], [], 0
        for i, c in zip(idx.tolist(), conf.tolist()):
            if i != 0 and i != prev and i - 1 < len(self.charset):
                chars.append(self.charset[i - 1])
                confs.append(c)
            prev = i
        return "".join(chars), (float(np.mean(confs)) if confs else 0.0)

    @torch.no_grad()
    def read(self, crop_bgr):
        return self.decode(self.model(self.preprocess(crop_bgr)))
