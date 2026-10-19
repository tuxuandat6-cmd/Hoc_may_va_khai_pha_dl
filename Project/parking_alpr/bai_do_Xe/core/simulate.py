"""Sinh dữ liệu ra/vào bãi xe mô phỏng nhiều tháng để phục vụ khai phá dữ liệu.

Vì một bãi xe thử nghiệm không có đủ lịch sử thật, ta mô phỏng 5 nhóm khách có hành vi khác nhau.
Nhãn nhóm thật (ground truth) được lưu riêng ra CSV — KHÔNG đưa vào CSDL — để sau này đánh giá
thuật toán phân cụm (Adjusted Rand Index). Dữ liệu cũng được chèn sẵn một số bất thường
(xe bỏ quên nhiều ngày, OCR đọc sai biển lúc ra) để kiểm chứng thuật toán phát hiện bất thường.
"""
import heapq
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config

PROVINCES = ["29", "30", "51", "59", "43", "15", "14", "36", "37", "92", "60", "79", "88", "99"]
SERIES = list("ABCDEFGHKLMNPSTUVXYZ")

# Mỗi nhóm: số xe, cách sinh lượt gửi trong 1 ngày
GROUPS = {
    "Nhân viên giờ hành chính": dict(n=45),
    "Làm ca chiều": dict(n=15),
    "Cư dân gửi qua đêm": dict(n=20),
    "Khách mua sắm cuối tuần": dict(n=40),
    "Khách vãng lai": dict(n=350),
}
GROUND_TRUTH_CSV = os.path.join(config.DATA_DIR, "sim_ground_truth.csv")


def _plate(rng, used):
    while True:
        p = f"{rng.choice(PROVINCES)}{rng.choice(SERIES)}{rng.randint(10000, 99999)}"
        if p not in used:
            used.add(p)
            return p


def _mutate(rng, plate):
    """Giả lập OCR đọc sai một ký tự (vd 8 -> B, 0 -> D) khi xe ra."""
    swaps = {"8": "6", "0": "8", "1": "7", "5": "6", "3": "8", "6": "8", "9": "8", "2": "7", "4": "1", "7": "1"}
    i = rng.randrange(len(plate) - 5, len(plate))
    return plate[:i] + swaps.get(plate[i], "0") + plate[i + 1:]


def _candidate_visits(rng, nrng, group, day):
    """Trả về list (giờ vào [float, giờ trong ngày], thời gian gửi [giờ]) của 1 xe trong 1 ngày."""
    wd = day.weekday() < 5
    if group == "Nhân viên giờ hành chính":
        if wd and rng.random() < 0.88:
            return [(nrng.normal(7.7, 0.45), max(6.5, nrng.normal(9.6, 0.6)))]
    elif group == "Làm ca chiều":
        if rng.random() < (0.75 if wd else 0.55):
            return [(nrng.normal(13.6, 0.4), max(5, nrng.normal(8.2, 0.5)))]
    elif group == "Cư dân gửi qua đêm":
        if rng.random() < 0.82:
            return [(nrng.normal(19.3, 1.0), max(8, nrng.normal(12.3, 1.3)))]
    elif group == "Khách mua sắm cuối tuần":
        if not wd and rng.random() < 0.5:
            return [(rng.uniform(9, 18), nrng.uniform(1.5, 4.5))]
        if wd and rng.random() < 0.03:
            return [(rng.uniform(17, 20), nrng.uniform(1, 2.5))]
    else:  # vãng lai: hiếm khi quay lại, gửi ngắn, giờ ngẫu nhiên trong ngày
        if rng.random() < (0.03 if wd else 0.05):
            return [(rng.uniform(8, 21), float(np.clip(nrng.lognormal(0.3, 0.6), 0.2, 6)))]
    return []


def generate(days=90, capacity=55, seed=42, end=None):
    """Sinh dữ liệu. Trả về (sessions_df, events_df, truth_df).
    sessions: Plate, TimeIn, TimeOut (NaT nếu xe còn trong bãi)
    events  : Plate, Direction, EventTime, Source, Confidence, Note"""
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    end = (end or datetime.now()).replace(microsecond=0)
    start = (end - timedelta(days=days)).replace(hour=0, minute=0, second=0)

    used, cars = set(), []
    for g, spec in GROUPS.items():
        cars += [(_plate(rng, used), g) for _ in range(spec["n"])]
    truth = pd.DataFrame(cars, columns=["Plate", "Group"])

    # 1) lượt đến dự kiến
    arrivals = []
    day = start
    while day < end:
        for plate, g in cars:
            for h, dur in _candidate_visits(rng, nrng, g, day):
                t_in = day + timedelta(hours=float(np.clip(h, 0, 23.95)))
                arrivals.append((t_in, plate, timedelta(hours=float(dur))))
        day += timedelta(days=1)
    # Bất thường 1: xe bỏ quên trong bãi 2-6 ngày
    for _ in range(max(3, days // 15)):
        t_in = start + timedelta(hours=rng.uniform(0, days * 24 - 150))
        arrivals.append((t_in, _plate(rng, used), timedelta(hours=rng.uniform(48, 144))))
    arrivals.sort()

    # 2) chạy mô phỏng theo thời gian, có giới hạn sức chứa
    sessions, events = [], []
    inside, inside_set = [], set()           # heap (time_out, plate)

    def conf():
        return round(float(np.clip(nrng.normal(0.93, 0.04), 0.6, 0.999)), 3)

    for t_in, plate, dur in arrivals:
        if t_in > end:
            break
        while inside and inside[0][0] <= t_in:
            t_out, p, sess_idx = heapq.heappop(inside)
            inside_set.discard(p)
            sessions[sess_idx][2] = t_out
            if rng.random() < 0.012:          # Bất thường 2: OCR đọc sai biển lúc ra
                events.append((_mutate(rng, p), "RA", t_out, "mô phỏng", round(rng.uniform(0.55, 0.75), 3),
                               "CẢNH BÁO - không có lượt vào"))
                events.append((p, "RA", t_out + timedelta(seconds=40), "nhập tay", None, None))
            else:
                events.append((p, "RA", t_out, "mô phỏng", conf(), None))
        if plate in inside_set:
            continue
        if len(inside) >= capacity:
            events.append((plate, "VÀO", t_in, "mô phỏng", conf(), "TỪ CHỐI - bãi đầy"))
            continue
        t_out = t_in + dur
        sessions.append([plate, t_in, None])
        heapq.heappush(inside, (t_out, plate, len(sessions) - 1))
        inside_set.add(plate)
        events.append((plate, "VÀO", t_in, "mô phỏng", conf(), None))

    # xe chưa ra trước thời điểm "end" -> vẫn trong bãi (TimeOut = NULL)
    while inside:
        t_out, p, sess_idx = heapq.heappop(inside)
        if t_out <= end:
            sessions[sess_idx][2] = t_out
            events.append((p, "RA", t_out, "mô phỏng", conf(), None))

    s = pd.DataFrame(sessions, columns=["Plate", "TimeIn", "TimeOut"])
    s["TimeIn"] = s["TimeIn"].dt.floor("s")
    s["TimeOut"] = pd.to_datetime(s["TimeOut"]).dt.floor("s")
    e = pd.DataFrame(events, columns=["Plate", "Direction", "EventTime", "Source", "Confidence", "Note"])
    e["EventTime"] = pd.to_datetime(e["EventTime"]).dt.floor("s")
    e = e.sort_values("EventTime", kind="stable").reset_index(drop=True)
    return s, e, truth


def save_ground_truth(truth):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    truth.to_csv(GROUND_TRUTH_CSV, index=False, encoding="utf-8-sig")


def load_ground_truth():
    return pd.read_csv(GROUND_TRUTH_CSV) if os.path.exists(GROUND_TRUTH_CSV) else None
