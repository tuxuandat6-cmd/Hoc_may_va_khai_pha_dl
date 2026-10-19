"""Sinh ảnh biển số xe Việt Nam tổng hợp (synthetic) để huấn luyện OCR CNN + BiLSTM + CTC.

Các loại biển được sinh:
  - Ô tô biển dài 1 dòng   : 30A-123.45, 51LD-012.34
  - Ô tô biển vuông 2 dòng  : 30A / 123.45
  - Xe máy 2 dòng           : 29-B1 / 123.45 , 59-AC / 886.88 , 29-X3 / 1234
Màu nền: trắng chữ đen (cá nhân), vàng chữ đen (kinh doanh vận tải), xanh chữ trắng (cơ quan nhà nước).
Nhãn = chuỗi ký tự không dấu phân cách, đúng chuẩn của core/plate_utils.normalize: "30A12345", "29AC88688".

Ảnh sau khi vẽ được tăng cường để giống vùng crop từ YOLO thật: viền nền xung quanh, xoay/nghiêng phối cảnh,
mờ, nhiễu, nén JPEG, thay đổi độ sáng/tương phản, bóng đổ, ảnh nhỏ bị phóng to.
"""
import glob
import io
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROVINCES = [f"{i:02d}" for i in list(range(11, 100)) if i not in (13, 42, 44, 45, 46, 87, 91, 96)]
SERIES = list("ABCDEFGHKLMNPSTUVXYZ")
CAR_2LETTER = ["LD", "KT", "NG", "NN", "HC", "MD"]       # biển ô tô có 2 chữ seri
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ"         # bộ ký tự của mô hình (index 0 = blank)

COLORS = [  # (nền, chữ, xác suất)
    ((240, 240, 236), (20, 20, 20), 0.70),
    ((233, 190, 40), (20, 20, 20), 0.18),
    ((30, 70, 170), (245, 245, 245), 0.12),
]


def _fonts():
    import matplotlib
    base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    fonts = [os.path.join(base, f) for f in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "DejaVuSansMono-Bold.ttf")]
    here = os.path.dirname(os.path.abspath(__file__))
    fonts += glob.glob(os.path.join(here, "fonts", "*.ttf")) + glob.glob(os.path.join(here, "fonts", "*.otf"))
    sysd = "/usr/share/fonts/truetype"
    fonts += glob.glob(f"{sysd}/**/DejaVuSansCondensed-Bold.ttf", recursive=True)
    return [f for f in fonts if os.path.exists(f)]


FONTS = _fonts()


# ------------------------------------------------------------------ sinh nội dung biển số
def random_plate(rng=random):
    """Trả về (kind, top, bottom, label). top/bottom là chuỗi hiển thị, label là chuỗi huấn luyện."""
    prov = rng.choice(PROVINCES)
    kind = rng.choices(["car_long", "car_square", "moto"], weights=[0.35, 0.2, 0.45])[0]
    if kind == "moto":
        r = rng.random()
        if r < 0.6:
            series = rng.choice(SERIES) + str(rng.randint(1, 9))          # B1, F8
        elif r < 0.85:
            series = rng.choice(SERIES) + rng.choice(SERIES)               # AC, AA
        else:
            series = rng.choice(SERIES) + str(rng.randint(1, 9))
        nums = f"{rng.randint(0, 99999):05d}" if rng.random() < 0.8 else f"{rng.randint(0, 9999):04d}"
        top = f"{prov}-{series}"
    else:
        series = rng.choice(CAR_2LETTER) if rng.random() < 0.1 else rng.choice(SERIES)
        nums = f"{rng.randint(0, 99999):05d}" if rng.random() < 0.85 else f"{rng.randint(0, 9999):04d}"
        top = f"{prov}{series}"
    bottom = f"{nums[:3]}.{nums[3:]}" if len(nums) == 5 else nums
    label = f"{prov}{series}{nums}"
    return kind, top, bottom, label


# ------------------------------------------------------------------ vẽ biển số
def _draw_text(text, font_path, height, squeeze, color):
    """Vẽ một dòng chữ cao `height` px, ép ngang theo hệ số `squeeze` (font biển số VN khá hẹp)."""
    font = ImageFont.truetype(font_path, int(height * 1.25))
    l, t, r, b = font.getbbox(text)
    img = Image.new("L", (r - l + 4, b - t + 4), 0)
    ImageDraw.Draw(img).text((2 - l, 2 - t), text, font=font, fill=255)
    img = img.resize((max(1, int(img.width * squeeze * height / img.height)), height), Image.LANCZOS)
    rgba = Image.new("RGBA", img.size, color + (255,))
    rgba.putalpha(img)
    return rgba


def render_plate(kind, top, bottom, rng=random):
    bg, fg, _ = rng.choices(COLORS, weights=[c[2] for c in COLORS])[0]
    font = rng.choice(FONTS)
    squeeze = rng.uniform(0.62, 0.85)
    if kind == "car_long":
        W, H = 520, 110
        lines = [top + rng.choice(["-", "-", " "]) + bottom]
    else:
        W, H = (280, 200) if kind == "car_square" else (250, 190)
        lines = [top, bottom]

    plate = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(plate)
    bw = rng.randint(3, 7)
    d.rounded_rectangle([bw, bw, W - bw - 1, H - bw - 1], radius=rng.randint(6, 16), outline=fg, width=bw)

    n = len(lines)
    pad_y = int(H * rng.uniform(0.08, 0.14))
    line_h = int((H - 2 * pad_y) / n * rng.uniform(0.78, 0.9))
    gap = (H - 2 * pad_y - n * line_h) // max(1, n)
    y = pad_y + gap // 2
    for s in lines:
        img = _draw_text(s, font, line_h, squeeze, fg)
        if img.width > W - 24:                                  # quá rộng -> thu lại
            img = img.resize((W - 24, line_h), Image.LANCZOS)
        plate.paste(img, ((W - img.width) // 2, y), img)
        y += line_h + gap
    return np.array(plate)


# ------------------------------------------------------------------ tăng cường dữ liệu
def augment(img, rng=random):
    h, w = img.shape[:2]
    # 1) viền nền xung quanh (giống bounding box YOLO lệch/rộng)
    px, py = int(w * rng.uniform(0.0, 0.12)), int(h * rng.uniform(0.0, 0.18))
    bgc = [rng.randint(20, 220)] * 3 if rng.random() < 0.6 else [rng.randint(0, 255) for _ in range(3)]
    img = cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=bgc)
    h, w = img.shape[:2]
    # 2) biến đổi phối cảnh (biển nghiêng, chụp chéo)
    m = rng.uniform(0.0, 0.12)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-m, m) * w, rng.uniform(-m, m) * h] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 3) xoay nhẹ
    if rng.random() < 0.5:
        R = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), 1.0)
        img = cv2.warpAffine(img, R, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 4) cắt lệch (bounding box cắt sát / cắt hụt)
    if rng.random() < 0.4:
        cx, cy = int(w * rng.uniform(0, 0.05)), int(h * rng.uniform(0, 0.06))
        img = img[cy:h - int(h * rng.uniform(0, 0.06)), cx:w - int(w * rng.uniform(0, 0.05))]
    # 5) ánh sáng: độ sáng/tương phản, gamma, bóng đổ
    img = img.astype(np.float32)
    img = img * rng.uniform(0.55, 1.3) + rng.uniform(-50, 40)
    if rng.random() < 0.35:
        hh, ww = img.shape[:2]
        x0 = rng.randint(0, ww)
        mask = np.ones((hh, ww), np.float32)
        mask[:, x0:] *= rng.uniform(0.4, 0.8)
        img *= cv2.GaussianBlur(mask, (31, 31), 0)[..., None]
    img = np.clip(img, 0, 255)
    if rng.random() < 0.3:
        img = 255 * (img / 255) ** rng.uniform(0.6, 1.6)
    img = img.astype(np.uint8)
    # 6) độ phân giải thấp (biển ở xa camera) rồi phóng to lại
    if rng.random() < 0.6:
        hh, ww = img.shape[:2]
        f = rng.uniform(0.15, 0.6)
        small = cv2.resize(img, (max(8, int(ww * f)), max(6, int(hh * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (ww, hh), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
    # 7) mờ chuyển động / gaussian
    r = rng.random()
    if r < 0.25:
        k = rng.choice([3, 5, 7])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1 / k
        img = cv2.filter2D(img, -1, kern)
    elif r < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.8))
    # 8) nhiễu + nén JPEG
    if rng.random() < 0.5:
        img = np.clip(img + np.random.normal(0, rng.uniform(3, 15), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 80)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def sample(rng=random, aug=True):
    """Một mẫu huấn luyện: (ảnh BGR uint8, nhãn)."""
    kind, top, bottom, label = random_plate(rng)
    img = cv2.cvtColor(render_plate(kind, top, bottom, rng), cv2.COLOR_RGB2BGR)
    return (augment(img, rng) if aug else img), label


if __name__ == "__main__":
    # Xem thử: python plate_synth.py  -> ghi preview_synth.jpg
    rng = random.Random(0)
    tiles = []
    for _ in range(24):
        img, label = sample(rng)
        t = cv2.resize(img, (256, 64))
        t = cv2.copyMakeBorder(t, 0, 18, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, label, (4, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        tiles.append(t)
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 24, 4)])
    cv2.imwrite("preview_synth.jpg", grid)
    print("Đã ghi preview_synth.jpg", FONTS)
