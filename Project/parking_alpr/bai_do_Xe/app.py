"""Hệ thống quản lý bãi đỗ xe bằng nhận dạng biển số (YOLOv8s + CNN-BiLSTM-CTC) — giao diện Gradio.

Chạy:  python app.py   ->  mở http://127.0.0.1:7860
"""
import os
from datetime import datetime

import cv2
import gradio as gr
import pandas as pd

import config
from core import plate_utils
from core.parking_db import ParkingLot

lot = ParkingLot()
_pipeline = None

DIRECTIONS = {"Tự động (trong bãi → RA, chưa có → VÀO)": "auto", "Cổng VÀO": "in", "Cổng RA": "out"}


def get_pipeline():
    """Nạp mô hình một lần khi cần (để giao diện vẫn mở được dù chưa có trọng số)."""
    global _pipeline
    if _pipeline is None:
        for p, name in ((config.YOLO_WEIGHTS, "YOLOv8"), (config.OCR_WEIGHTS, "OCR")):
            if not os.path.exists(p):
                raise gr.Error(f"Chưa có trọng số {name}: {p}. Xem README để đặt file vào thư mục weights/.")
        from core.pipeline import ALPRPipeline
        _pipeline = ALPRPipeline()
    return _pipeline


def save_snapshot(img_bgr, plate):
    if img_bgr is None:
        return None
    path = os.path.join(config.SNAPSHOT_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{plate}.jpg")
    cv2.imwrite(path, img_bgr)
    return path


# ---------------------------------------------------------------- trạng thái bãi
def status_html():
    cap, used = lot.capacity, lot.occupied()
    free = cap - used
    pct = used / cap * 100 if cap else 100
    color = "#16a34a" if pct < 70 else ("#d97706" if pct < 90 else "#dc2626")
    full = "<div class='full'>BÃI ĐÃ ĐẦY</div>" if free <= 0 else ""
    return f"""
    <div class='stats'>
      <div class='card'><div class='lbl'>Chỗ trống</div><div class='num' style='color:{color}'>{free}</div></div>
      <div class='card'><div class='lbl'>Đang đỗ</div><div class='num'>{used}</div></div>
      <div class='card'><div class='lbl'>Sức chứa</div><div class='num'>{cap}</div></div>
    </div>
    <div class='bar'><div style='width:{min(pct, 100):.0f}%;background:{color}'></div></div>{full}"""


def inside_df():
    rows = lot.vehicles_inside()
    return pd.DataFrame([(plate_utils.format_plate(p), t.replace("T", " ")) for p, t in rows],
                        columns=["Biển số", "Giờ vào"])


def history_df(plate_filter=""):
    rows = lot.history(plate_utils.clean(plate_filter))
    return pd.DataFrame(
        [(t.replace("T", " "), plate_utils.format_plate(p), d, s, f"{c:.2f}" if c is not None else "", n or "")
         for t, p, d, s, c, n in rows],
        columns=["Thời gian", "Biển số", "Hướng", "Nguồn", "Độ tin cậy OCR", "Ghi chú"])


def refresh_all(plate_filter=""):
    """Chỉ cập nhật thanh trạng thái. Bảng "Xe trong bãi" và "Lịch sử" được tải lại khi người dùng mở tab đó
    (cập nhật component nằm trong tab đang ẩn có thể làm treo việc chuyển tab ở một số phiên bản Gradio)."""
    return (status_html(),)


# ---------------------------------------------------------------- xử lý ảnh
def handle_image(img_rgb, direction_label, plate_filter):
    if img_rgb is None:
        raise gr.Error("Hãy tải ảnh lên hoặc chụp từ webcam.")
    pipe = get_pipeline()
    from core.pipeline import draw
    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = pipe.process(frame)

    msgs, crops = [], []
    if not results:
        msgs.append("❌ Không phát hiện được biển số nào trong ảnh.")
    for r in results:
        shown = plate_utils.format_plate(r.plate)
        crops.append((cv2.cvtColor(r.crop, cv2.COLOR_BGR2RGB), f"{shown} (OCR {r.ocr_conf:.2f})"))
        if not r.valid:
            msgs.append(f"⚠️ Đọc được '{r.raw_text}' nhưng không đúng định dạng biển số VN — "
                        "chưa ghi nhận. Hãy sửa tay ở tab 'Nhập tay'.")
            continue
        snap = save_snapshot(r.crop, r.plate)
        ok, msg = lot.register(r.plate, DIRECTIONS[direction_label], "ảnh", r.ocr_conf, snap)
        msgs.append(msg)

    annotated = cv2.cvtColor(draw(frame, results), cv2.COLOR_BGR2RGB)
    return (annotated, crops, "\n".join(msgs), *refresh_all(plate_filter))


# ---------------------------------------------------------------- xử lý video
def handle_video(video_path, direction_label, plate_filter, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Hãy tải video lên.")
    pipe = get_pipeline()
    from core.video_processor import process_video
    direction = DIRECTIONS[direction_label]

    def on_confirm(plate, conf, crop, frame):
        snap = save_snapshot(crop, plate)
        _, msg = lot.register(plate, direction, "video", conf, snap)
        return msg

    out_path, logs = process_video(video_path, pipe, on_confirm, direction, progress)
    text = "\n".join(logs) if logs else "Không xác nhận được biển số nào (thử giảm VIDEO_MIN_VOTES trong config.py)."
    return (out_path, text, *refresh_all(plate_filter))


# ---------------------------------------------------------------- nhập tay
def handle_manual(plate_text, direction, plate_filter):
    plate = plate_utils.normalize(plate_text)
    if not plate_utils.is_valid(plate):
        raise gr.Error(f"'{plate_text}' không đúng định dạng biển số (vd: 29A12345, 59U102795, 60LD03645).")
    _, msg = lot.register(plate, direction, "nhập tay", None, None, cooldown=False)
    return (msg, *refresh_all(plate_filter))


def set_capacity(n, plate_filter):
    try:
        lot.set_capacity(n)
    except ValueError as e:
        raise gr.Error(str(e))
    return ("Đã cập nhật sức chứa.", *refresh_all(plate_filter))


def reset_db(plate_filter):
    lot.reset()
    from core.simulate import GROUND_TRUTH_CSV
    if os.path.exists(GROUND_TRUTH_CSV):
        os.remove(GROUND_TRUTH_CSV)
    return ("Đã xoá toàn bộ dữ liệu.", *refresh_all(plate_filter))


def export_csv():
    path = os.path.join(config.DATA_DIR, f"lich_su_{datetime.now():%Y%m%d_%H%M%S}.csv")
    history_df().to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ---------------------------------------------------------------- khai phá dữ liệu
def _load():
    from core import analytics as A
    raw = lot.load_sessions()
    if raw.empty:
        raise gr.Error("Chưa có dữ liệu lịch sử. Hãy bấm 'Sinh dữ liệu mô phỏng' hoặc ghi nhận xe trước.")
    return A, A.prepare(raw), lot.load_events()


def simulate_data(days, capacity, plate_filter, progress=gr.Progress()):
    from core import simulate
    progress(0.1, desc="Đang mô phỏng lượt xe ra/vào...")
    sessions, events, truth = simulate.generate(int(days), int(capacity))
    progress(0.5, desc=f"Đang ghi {len(sessions) + len(events):,} dòng vào SQL Server...")
    lot.bulk_load(sessions, events, capacity=int(capacity))
    simulate.save_ground_truth(truth)
    n = lambda x: f"{x:,}".replace(",", ".")
    msg = (f"Đã sinh {n(len(sessions))} lượt gửi / {n(len(events))} sự kiện trong {int(days)} ngày "
           f"cho {truth.shape[0]} xe (5 nhóm hành vi), sức chứa {int(capacity)} chỗ.")
    return (msg, *refresh_all(plate_filter))


def run_overview():
    A, s, e = _load()
    return A.descriptive(s, e, lot.capacity)


def run_cluster(auto_k, k):
    A, s, _ = _load()
    from core.simulate import load_ground_truth
    return A.clustering(s, load_ground_truth(), k=None if auto_k else int(k))


def run_forecast(horizon, future_hours):
    A, s, _ = _load()
    return A.forecast(s, lot.capacity, int(horizon), future_hours=int(future_hours))


def run_anomaly(contamination):
    A, s, e = _load()
    return A.anomalies(s, e, contamination / 100)


def run_rules(min_support, min_conf):
    A, s, _ = _load()
    return A.association_rules(s, min_support, min_conf)


# ---------------------------------------------------------------- giao diện
CSS = """
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.kpi{border:1px solid var(--border-color-primary);border-radius:10px;padding:10px 14px;background:var(--block-background-fill)}
.kpi .lbl{font-size:12px}
.kpi .val{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.25}
.kpi .sub{font-size:12px;opacity:.65}
.warn{margin-bottom:10px;padding:10px 14px;border-radius:10px;background:#fff4e0;color:#7a4b00;border:1px solid #f0c674}
.stats{display:flex;gap:12px}
.card{flex:1;border:1px solid var(--border-color-primary);border-radius:12px;padding:14px 18px;background:var(--block-background-fill)}
.lbl{font-size:13px;opacity:.7;text-transform:uppercase;letter-spacing:.04em}
.num{font-size:40px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}
.bar{height:10px;border-radius:5px;background:var(--border-color-primary);margin-top:12px;overflow:hidden}
.bar>div{height:100%;transition:width .4s}
.full{margin-top:10px;padding:8px;border-radius:8px;background:#dc2626;color:#fff;text-align:center;font-weight:700}
"""

GR6 = int(gr.__version__.split(".")[0]) >= 6
_style = dict(css=CSS, theme=gr.themes.Soft())

with gr.Blocks(title="Bãi đỗ xe thông minh", **({} if GR6 else _style)) as demo:
    gr.Markdown("# 🅿️ Bãi đỗ xe thông minh\nNhận dạng biển số bằng **YOLOv8s + CNN‑BiLSTM‑CTC**, "
                "tự động trừ/cộng chỗ trống và ghi lịch sử ra vào.")
    status = gr.HTML(status_html())

    with gr.Tabs():
        with gr.Tab("📷 Ảnh"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="Ảnh xe (tải lên hoặc chụp webcam)", sources=["upload", "webcam", "clipboard"])
                    img_dir = gr.Radio(list(DIRECTIONS), value=list(DIRECTIONS)[0], label="Hướng")
                    img_btn = gr.Button("Nhận dạng & ghi nhận", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="Kết quả")
                    img_crops = gr.Gallery(label="Vùng biển số", height=140, columns=4)
                    img_msg = gr.Textbox(label="Thông báo", lines=3)

        with gr.Tab("🎞️ Video"):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="Video camera cổng", sources=["upload"])
                    vid_dir = gr.Radio(list(DIRECTIONS), value=list(DIRECTIONS)[0], label="Hướng")
                    vid_btn = gr.Button("Xử lý video", variant="primary")
                with gr.Column():
                    vid_out = gr.Video(label="Video đã đánh dấu")
                    vid_msg = gr.Textbox(label="Xe được ghi nhận", lines=8)

        with gr.Tab("✍️ Nhập tay"):
            gr.Markdown("Dùng khi OCR đọc sai hoặc camera không nhận được biển số.")
            with gr.Row():
                man_plate = gr.Textbox(label="Biển số", placeholder="29A-123.45")
                man_in = gr.Button("Xe VÀO", variant="primary")
                man_out = gr.Button("Xe RA")
            man_msg = gr.Textbox(label="Thông báo")

        with gr.Tab("🚗 Xe trong bãi") as tab_inside:
            inside_tbl = gr.Dataframe(inside_df(), interactive=False)

        with gr.Tab("📜 Lịch sử ra vào") as tab_hist:
            with gr.Row():
                hist_filter = gr.Textbox(label="Lọc theo biển số", placeholder="vd: 29A")
                export_btn = gr.Button("Xuất CSV")
            export_file = gr.File(label="File CSV")
            hist_tbl = gr.Dataframe(history_df(), interactive=False)

        with gr.Tab("📊 Phân tích & Khai phá dữ liệu"):
            with gr.Accordion("Dữ liệu mô phỏng (dùng khi chưa có đủ lịch sử thật)", open=False):
                gr.Markdown("Sinh lịch sử ra/vào của **5 nhóm khách** (đi làm giờ hành chính, ca chiều, gửi qua đêm, "
                            "mua sắm cuối tuần, vãng lai), có giới hạn sức chứa và chèn sẵn bất thường "
                            "(xe bỏ quên nhiều ngày, OCR đọc sai). **Thao tác này xoá dữ liệu hiện có.**")
                with gr.Row():
                    sim_days = gr.Slider(30, 180, value=90, step=15, label="Số ngày")
                    sim_cap = gr.Number(value=55, precision=0, label="Sức chứa")
                    sim_btn = gr.Button("Sinh dữ liệu mô phỏng", variant="stop")
                sim_msg = gr.Textbox(label="Kết quả", lines=1)

            with gr.Tabs():
                with gr.Tab("1. Tổng quan"):
                    ov_btn = gr.Button("Phân tích", variant="primary")
                    ov_kpi = gr.HTML()
                    with gr.Row():
                        ov_f1, ov_f2 = gr.Plot(show_label=False), gr.Plot(show_label=False)
                    with gr.Row():
                        ov_f4, ov_f3 = gr.Plot(show_label=False), gr.Plot(show_label=False)

                with gr.Tab("2. Phân cụm khách hàng"):
                    gr.Markdown("**K-Means** trên hồ sơ hành vi từng biển số: tần suất, giờ vào (mã hoá sin/cos vì là biến "
                                "tròn), thời gian gửi, độ đều đặn, tỷ lệ cuối tuần, tỷ lệ qua đêm.")
                    with gr.Row():
                        cl_auto = gr.Checkbox(True, label="Tự chọn K theo Silhouette")
                        cl_k = gr.Slider(2, 8, value=4, step=1, label="K (khi không tự chọn)")
                        cl_btn = gr.Button("Phân cụm", variant="primary")
                    cl_msg = gr.Textbox(label="Kết quả", lines=3)
                    cl_f1 = gr.Plot(show_label=False)
                    cl_f2 = gr.Plot(show_label=False)
                    cl_tbl = gr.Dataframe(label="Hồ sơ trung bình của mỗi nhóm", interactive=False)

                with gr.Tab("3. Dự báo chỗ trống"):
                    gr.Markdown("Dự báo **số xe trong bãi** vài giờ tới bằng hồi quy trên chuỗi thời gian "
                                "(đặc trưng: giờ, thứ, độ trễ 1/2/3/24/168 giờ). So sánh Baseline, Linear Regression, "
                                "Random Forest trên 14 ngày cuối.")
                    with gr.Row():
                        fc_h = gr.Slider(1, 6, value=1, step=1, label="Đánh giá: dự báo trước bao nhiêu giờ")
                        fc_future = gr.Slider(3, 24, value=12, step=1, label="Dự báo cho bao nhiêu giờ tới")
                        fc_btn = gr.Button("Dự báo", variant="primary")
                    fc_msg = gr.Textbox(label="Kết quả", lines=3)
                    fc_metrics = gr.Dataframe(label="Sai số trên tập kiểm thử", interactive=False)
                    fc_f2 = gr.Plot(show_label=False)
                    fc_f1 = gr.Plot(show_label=False)

                with gr.Tab("4. Phát hiện bất thường"):
                    gr.Markdown("**Isolation Forest** trên từng lượt gửi (thời gian gửi, giờ vào, cuối tuần, tần suất "
                                "của xe, độ lệch so với thói quen) kết hợp **luật nghiệp vụ** "
                                "(gửi > 24h, xe ra không có lượt vào → nghi OCR đọc sai).")
                    with gr.Row():
                        an_cont = gr.Slider(0.2, 5, value=1, step=0.1, label="Tỷ lệ bất thường ước tính (%)")
                        an_btn = gr.Button("Phát hiện", variant="primary")
                    an_msg = gr.Textbox(label="Kết quả", lines=2)
                    an_fig = gr.Plot(show_label=False)
                    an_tbl = gr.Dataframe(label="Danh sách bất thường", interactive=False, max_height=420)

                with gr.Tab("5. Luật kết hợp"):
                    gr.Markdown("**Apriori**: mỗi lượt gửi là một giao dịch gồm các thuộc tính rời rạc "
                                "(ngày, khung giờ vào, thời lượng gửi, khách quen/ít đến, qua đêm).")
                    with gr.Row():
                        ar_sup = gr.Slider(0.01, 0.3, value=0.05, step=0.01, label="Min support")
                        ar_conf = gr.Slider(0.3, 1.0, value=0.6, step=0.05, label="Min confidence")
                        ar_btn = gr.Button("Tìm luật", variant="primary")
                    ar_msg = gr.Textbox(label="Kết quả", lines=2)
                    ar_tbl = gr.Dataframe(label="Luật kết hợp (sắp theo Lift)", interactive=False)

        with gr.Tab("⚙️ Cài đặt"):
            cap_in = gr.Number(value=lot.capacity, precision=0, label="Sức chứa bãi (số chỗ)")
            cap_btn = gr.Button("Lưu sức chứa")
            reset_btn = gr.Button("Xoá toàn bộ dữ liệu", variant="stop")
            set_msg = gr.Textbox(label="Thông báo")

    tables = [status]
    tab_inside.select(inside_df, None, inside_tbl)
    tab_hist.select(history_df, hist_filter, hist_tbl)
    img_btn.click(handle_image, [img_in, img_dir, hist_filter], [img_out, img_crops, img_msg, *tables])
    vid_btn.click(handle_video, [vid_in, vid_dir, hist_filter], [vid_out, vid_msg, *tables])
    man_in.click(lambda p, f: handle_manual(p, "in", f), [man_plate, hist_filter], [man_msg, *tables])
    man_out.click(lambda p, f: handle_manual(p, "out", f), [man_plate, hist_filter], [man_msg, *tables])
    hist_filter.change(history_df, hist_filter, hist_tbl)
    export_btn.click(export_csv, None, export_file)
    cap_btn.click(set_capacity, [cap_in, hist_filter], [set_msg, *tables])
    reset_btn.click(reset_db, hist_filter, [set_msg, *tables])
    demo.load(refresh_all, hist_filter, tables)

    sim_btn.click(simulate_data, [sim_days, sim_cap, hist_filter], [sim_msg, *tables])
    ov_btn.click(run_overview, None, [ov_kpi, ov_f1, ov_f2, ov_f3, ov_f4])
    cl_btn.click(run_cluster, [cl_auto, cl_k], [cl_f1, cl_f2, cl_tbl, cl_msg])
    fc_btn.click(run_forecast, [fc_h, fc_future], [fc_f1, fc_metrics, fc_f2, fc_msg])
    an_btn.click(run_anomaly, an_cont, [an_fig, an_tbl, an_msg])
    ar_btn.click(run_rules, [ar_sup, ar_conf], [ar_tbl, ar_msg])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, **(_style if GR6 else {}))
