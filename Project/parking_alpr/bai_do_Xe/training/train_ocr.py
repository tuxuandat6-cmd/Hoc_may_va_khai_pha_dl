"""Huấn luyện OCR CNN + BiLSTM + CTC (đúng class CRNN trong core/ocr_model.py của app).

Dữ liệu:
  - Ảnh biển số tổng hợp sinh ngẫu nhiên mỗi epoch (plate_synth.py) -> dữ liệu không giới hạn.
  - (Tuỳ chọn) ảnh biển số THẬT đã cắt sẵn: một thư mục chứa ảnh + file labels.csv (cột: filename,label).
    Ảnh thật được trộn với tỷ lệ --real-ratio và chia 85/15 train/val. Càng nhiều ảnh thật càng tốt.

Chạy:
  python training/train_ocr.py --epochs 40                       # chỉ dữ liệu tổng hợp
  python training/train_ocr.py --real-dir my_crops --epochs 40   # trộn thêm ảnh thật
Kết quả: weights/crnn_bilstm_ctc.pth (lưu kèm charset) — app tự đọc.
"""
import argparse
import csv
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from core.ocr_model import CRNN, preprocess_plate   # noqa: E402  — cùng kiến trúc & tiền xử lý với app
from core.plate_utils import clean       # noqa: E402
import plate_synth                       # noqa: E402

CHARSET = plate_synth.CHARSET
C2I = {c: i + 1 for i, c in enumerate(CHARSET)}   # 0 = CTC blank
IMG_H, IMG_W = 64, 256


def to_tensor(img_bgr):
    """Dùng đúng hàm tiền xử lý của app (ghép biển 2 dòng thành 1 dòng -> xám -> 256x64 -> /255)."""
    return torch.from_numpy(preprocess_plate(img_bgr, two_line_split=True))[None]


def encode(label):
    return [C2I[c] for c in label if c in C2I]


class SynthDataset(Dataset):
    def __init__(self, n, seed=None):
        self.n, self.seed = n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i) if self.seed is not None else random
        img, label = plate_synth.sample(rng)
        return to_tensor(img), label


class RealDataset(Dataset):
    def __init__(self, items, aug=False):
        self.items, self.aug = items, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = cv2.imread(path)
        if self.aug and random.random() < 0.7:
            img = plate_synth.augment(img)
        return to_tensor(img), label


class Mixed(Dataset):
    """Mỗi epoch: n mẫu, tỷ lệ `ratio` lấy từ ảnh thật (lặp lại nếu ít), phần còn lại là ảnh tổng hợp."""
    def __init__(self, synth, real, n, ratio):
        self.synth, self.real, self.n, self.ratio = synth, real, n, ratio if real else 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.real and random.random() < self.ratio:
            return self.real[random.randrange(len(self.real))]
        return self.synth[i]


def load_real(real_dir):
    items = []
    with open(os.path.join(real_dir, "labels.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lab = clean(row["label"])
            p = os.path.join(real_dir, row["filename"])
            if lab and os.path.exists(p):
                items.append((p, lab))
    random.Random(0).shuffle(items)
    k = int(len(items) * 0.85)
    return items[:k], items[k:]


def collate(batch):
    imgs, labels = zip(*batch)
    targets = [torch.tensor(encode(l), dtype=torch.long) for l in labels]
    return (torch.stack(imgs), torch.cat(targets), torch.tensor([len(t) for t in targets]), list(labels))


def greedy_decode(logits):
    idx = logits.argmax(-1).permute(1, 0).cpu().numpy()     # (B, T)
    out = []
    for seq in idx:
        s, prev = [], 0
        for k in seq:
            if k != 0 and k != prev:
                s.append(CHARSET[k - 1])
            prev = k
        out.append("".join(s))
    return out


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errs = chars = exact = n = 0
    for imgs, _, _, labels in loader:
        preds = greedy_decode(model(imgs.to(device)))
        for p, g in zip(preds, labels):
            errs += edit_distance(p, g)
            chars += len(g)
            exact += p == g
            n += 1
    cer = errs / max(1, chars)
    return {"CER": cer, "CharAcc": max(0.0, 1 - cer), "PlateAcc": exact / max(1, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--samples-per-epoch", type=int, default=20000)
    ap.add_argument("--val-samples", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--real-dir", default=None, help="thư mục ảnh thật + labels.csv (tuỳ chọn)")
    ap.add_argument("--real-ratio", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "weights", "crnn_bilstm_ctc.pth"))
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {device} | font: {[os.path.basename(f) for f in plate_synth.FONTS]}")

    real_tr, real_va = load_real(args.real_dir) if args.real_dir else ([], [])
    if args.real_dir:
        print(f"Ảnh thật: {len(real_tr)} train / {len(real_va)} val")
    train_ds = Mixed(SynthDataset(args.samples_per_epoch), RealDataset(real_tr, aug=True) if real_tr else None,
                     args.samples_per_epoch, args.real_ratio)
    kw = dict(batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
              pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, **kw)
    val_synth = DataLoader(SynthDataset(args.val_samples, seed=12345), shuffle=False, **kw)
    val_real = DataLoader(RealDataset(real_va), shuffle=False, **kw) if real_va else None

    model = CRNN(num_classes=len(CHARSET) + 1).to(device)
    print(f"Số tham số: {sum(p.numel() for p in model.parameters()):,}")
    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-5)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best, bad, history = -1, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for imgs, targets, tlen, _ in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)                                   # (T, B, C)
            logp = logits.log_softmax(2)
            ilen = torch.full((imgs.size(0),), logp.size(0), dtype=torch.long)
            loss = crit(logp.float(), targets, ilen, tlen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1

        m = evaluate(model, val_real or val_synth, device)
        ms = evaluate(model, val_synth, device) if val_real else m
        score = m["PlateAcc"] + m["CharAcc"] * 0.1
        sched.step(score)
        history.append({"epoch": ep, "loss": tot / nb, **m})
        tag = "thật" if val_real else "tổng hợp"
        print(f"Epoch {ep:3d} | loss {tot / nb:.4f} | val({tag}) CharAcc {m['CharAcc']:.4f} "
              f"CER {m['CER']:.4f} PlateAcc {m['PlateAcc']:.4f}"
              + (f" | val(tổng hợp) PlateAcc {ms['PlateAcc']:.4f}" if val_real else "")
              + f" | lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s", flush=True)

        if score > best:
            best, bad = score, 0
            torch.save({"model_state_dict": model.state_dict(), "charset": CHARSET,
                        "img_h": IMG_H, "img_w": IMG_W, "normalize": "0_1", "two_line_split": True,
                        "epoch": ep, "metrics": m, "history": history}, args.out)
            print(f"  ↳ lưu mô hình tốt nhất -> {args.out}")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping sau {args.patience} epoch không cải thiện.")
                break
    print("Xong. Mô hình tốt nhất:", torch.load(args.out, map_location="cpu", weights_only=False)["metrics"])


if __name__ == "__main__":
    main()
