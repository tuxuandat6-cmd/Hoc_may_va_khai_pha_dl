# 🅿️ Bãi đỗ xe thông minh — Nhận dạng biển số (YOLOv8s + CNN‑BiLSTM‑CTC)

Kế thừa pipeline ALPR trong báo cáo BTL: **YOLOv8s** phát hiện biển số → cắt vùng biển →
**CNN + BiLSTM + CTC** đọc ký tự. Thêm phần quản lý bãi xe: đếm chỗ trống, xe vào trừ 1 chỗ,
xe ra cộng 1 chỗ, ghi lịch sử ra/vào vào **SQL Server**, giao diện Gradio.

## Cấu trúc

```
parking_alpr/
├── app.py                  # Giao diện Gradio (chạy file này)
├── config.py               # Đường dẫn trọng số, CHARSET, sức chứa, tham số video
├── core/
│   ├── ocr_model.py        # Kiến trúc CRNN (CNN+SE+BiLSTM+CTC) + greedy CTC decode
│   ├── pipeline.py         # YOLOv8 -> crop -> OCR, vẽ kết quả
│   ├── plate_utils.py      # Chuẩn hoá, sửa nhầm 0/O 8/B 5/S..., kiểm tra định dạng biển VN
│   ├── video_processor.py  # Theo dõi xe qua các frame + bỏ phiếu -> mỗi xe chỉ ghi 1 lần
│   ├── analytics.py        # KHAI PHÁ DỮ LIỆU: thống kê, K-Means, dự báo, Isolation Forest, Apriori
│   ├── simulate.py         # Sinh dữ liệu ra/vào mô phỏng 5 nhóm khách để phân tích
│   └── parking_db.py       # SQL Server: sức chứa, xe trong bãi, lịch sử
├── sql/create_database.sql # Script tạo CSDL + truy vấn mẫu (mở bằng SSMS, tuỳ chọn)
├── training/               # HUẤN LUYỆN MÔ HÌNH (chạy trên Google Colab)
│   ├── train_colab.ipynb   # Notebook: bấm chạy lần lượt -> ra 2 file trọng số
│   ├── train_yolo.py       # YOLOv8s phát hiện biển số (dữ liệu Roboflow)
│   ├── train_ocr.py        # CNN + BiLSTM + CTC (dữ liệu tổng hợp + ảnh thật tuỳ chọn)
│   └── plate_synth.py      # Bộ sinh ảnh biển số Việt Nam tổng hợp
├── weights/                # ĐẶT 2 FILE TRỌNG SỐ VÀO ĐÂY
└── data/                   # ảnh chụp biển số, file CSV xuất ra (tự tạo)
```

## 🏋️ Bước 0 — Huấn luyện mô hình (nếu chưa có trọng số)

Máy không có GPU thì dùng **Google Colab** (miễn phí):

1. Vào [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → chọn `training/train_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Chạy lần lượt các ô: tải `parking_alpr.zip` lên → cài thư viện → huấn luyện YOLOv8 (cần
   [Roboflow API key](https://roboflow.com) miễn phí) → huấn luyện OCR → chạy thử → tải `weights.zip` về.
4. Giải nén `weights.zip` vào `parking_alpr/weights/` rồi chạy `python app.py`.

Tổng thời gian khoảng 1,5 giờ trên GPU T4.

| Mô hình | Dữ liệu |
|---|---|
| YOLOv8s | [License Plate Recognition (Roboflow)](https://universe.roboflow.com/computer-vison-4zhos/license-plate-recognition-kvr7y) — 3.132 ảnh biển số VN, CC BY 4.0 |
| CNN+BiLSTM+CTC | Ảnh biển số VN **tổng hợp** do `plate_synth.py` sinh (1/2 dòng, 3 màu nền, augmentation mạnh) — cùng ý tưởng với nhóm ảnh "generated" trong báo cáo. Có thể trộn ảnh thật bằng `--real-dir` |

⚠️ OCR chỉ học từ ảnh tổng hợp sẽ đọc ảnh camera thật kém hơn số liệu validation. Muốn tốt hơn: gom vài trăm ảnh
biển số thật đã cắt, gõ nhãn vào `labels.csv` (`filename,label`) và huấn luyện lại với `--real-dir`.

## Cơ sở dữ liệu SQL Server

- Mặc định kết nối `localhost`, **Windows Authentication**, database `ParkingDB`.
  App tự tạo database và 3 bảng khi chạy lần đầu (hoặc chạy `sql/create_database.sql` trong SSMS).
- Cần cài **ODBC Driver 17 hoặc 18 for SQL Server** (thường đã có sẵn khi cài SSMS).
- Bảng: `Settings` (sức chứa), `ParkingSessions` (mỗi lượt gửi xe, `TimeOut IS NULL` = xe đang trong bãi),
  `ParkingEvents` (nhật ký mọi lần quét, gồm cả lượt bị từ chối).
- Đổi cấu hình trong `config.py`:
  - Dùng SQL Express: `SQL_SERVER = "localhost\\SQLEXPRESS"`
  - Dùng tài khoản `sa`: `SQL_TRUSTED = False`, điền `SQL_USER`, `SQL_PASSWORD`.

## Cài đặt & chạy

```bash
pip install -r requirements.txt
# copy trọng số đã huấn luyện:
#   weights/yolov8s_plate.pt     (best.pt của YOLOv8s)
#   weights/crnn_bilstm_ctc.pth  (best_model.pth của CNN+BiLSTM+CTC)
python app.py          # mở http://127.0.0.1:7860
```

Giao diện vẫn mở được khi chưa có trọng số (tab Nhập tay, Lịch sử, Cài đặt dùng bình thường).

## ⚠️ 3 việc phải kiểm tra với code huấn luyện của nhóm

1. **Kiến trúc OCR** — `core/ocr_model.py` dựng lại theo mô tả trong báo cáo, nhưng tên layer
   có thể khác notebook gốc (báo cáo ghi 8.554.853 tham số, bản dựng lại là 7.709.381). Nếu khi
   chạy báo lỗi *"Checkpoint OCR không khớp kiến trúc"*, hãy dán class mô hình gốc vào file đó
   (giữ tên `CRNN`, `forward` trả về tensor `(T, B, C)`).
2. **CHARSET** trong `config.py` phải **giống hệt** lúc huấn luyện (index 0 = blank, ký tự thứ i
   → index i+1). Báo cáo ghi 36 ký tự; chuỗi mặc định đang có 35 → copy đúng chuỗi từ notebook.
   Tốt nhất lúc lưu checkpoint thêm `"charset": CHARSET`, app sẽ tự đọc.
3. **Chuẩn hoá pixel** (`OCR_NORMALIZE`): `"0_1"` nếu lúc train chia 255, `"minus1_1"` nếu
   chuẩn hoá về [-1, 1].

## Cách hệ thống quyết định VÀO / RA

| Chế độ | Hành vi |
|---|---|
| Tự động | Biển số đang trong bãi → ghi **RA** (+1 chỗ), chưa có → ghi **VÀO** (−1 chỗ) |
| Cổng VÀO | Luôn là xe vào; bãi đầy thì từ chối và ghi "TỪ CHỐI – bãi đầy" |
| Cổng RA | Luôn là xe ra; nếu không có lượt vào thì cảnh báo (thường do OCR đọc sai) |

Chống đếm trùng: cùng một biển số trong `COOLDOWN_SEC` giây (mặc định 30s) bị bỏ qua.
Với video, mỗi xe được theo dõi qua các frame (IoU) và chỉ ghi nhận khi cùng một chuỗi được
đọc ≥ `VIDEO_MIN_VOTES` lần — giúp kết quả ổn định hơn so với đọc từng frame (Plate Accuracy
của OCR là 79%, nên bỏ phiếu nhiều frame cải thiện đáng kể).

Hậu xử lý theo định dạng biển VN (hướng phát triển "thứ năm" trong báo cáo): mã tỉnh và 4 số
cuối luôn là số, ký tự seri đầu luôn là chữ → tự sửa các lỗi nhầm O↔0, B↔8, S↔5, Z↔2...
Chuỗi không đúng định dạng sẽ **không** được ghi nhận, nhân viên sửa ở tab *Nhập tay*.

## 📊 Phần khai phá dữ liệu (tab "Phân tích & Khai phá dữ liệu")

Nguồn dữ liệu là chính lịch sử ra/vào trong SQL Server (`ParkingSessions`, `ParkingEvents`).
Khi chưa có đủ lịch sử thật, bấm **"Sinh dữ liệu mô phỏng"**: tạo 30–180 ngày dữ liệu của 470 xe thuộc
5 nhóm hành vi (đi làm giờ hành chính, ca chiều, gửi qua đêm, mua sắm cuối tuần, vãng lai), có giới hạn
sức chứa (sinh ra tình huống bãi đầy) và chèn sẵn bất thường (xe bỏ quên nhiều ngày, OCR đọc sai lúc ra).
Nhãn nhóm thật chỉ lưu ra `data/sim_ground_truth.csv` để **đánh giá** phân cụm, không đưa vào CSDL.
⚠️ Trong báo cáo cần ghi rõ đây là dữ liệu mô phỏng.

| # | Bài toán | Phương pháp | Đánh giá | Kết quả trên dữ liệu mô phỏng 90 ngày |
|---|---|---|---|---|
| 1 | Thống kê mô tả | Heatmap thứ×giờ, chuỗi lấp đầy, phân bố thời gian gửi | KPI | Cao điểm 7h–8h, bãi đầy 23 giờ, 41 lượt bị từ chối |
| 2 | Phân khúc khách hàng | Phân đoạn tần suất (< 0,6 lượt/tuần = vãng lai) + **K-Means**; K chọn bằng **Silhouette** (Elbow tham khảo); PCA để trực quan | Silhouette, **Adjusted Rand Index** so với nhãn thật | K = 4, Silhouette 0,87, ARI 0,98 |
| 3 | Dự báo số xe trong bãi | Hồi quy chuỗi thời gian, đặc trưng giờ (sin/cos), thứ, độ trễ 1/2/3/24/168h; **Baseline vs Linear Regression vs Random Forest**; chia train/test theo thời gian | MAE, RMSE, R² | RF: MAE 1,48 xe, R² 0,976 (baseline MAE 2,55) |
| 4 | Phát hiện bất thường | **Isolation Forest** + luật nghiệp vụ | Đối chiếu bất thường đã chèn | Bắt đủ 6/6 xe bỏ quên, 75/75 lượt OCR đọc sai |
| 5 | Luật kết hợp | **Apriori** (mlxtend), tỉa luật dư thừa & luật hiển nhiên | Support, Confidence, Lift | VD: vào sáng sớm → gửi 8–12h (conf 0,96) |

Một số lựa chọn kỹ thuật đáng nêu trong báo cáo:
- **Giờ là biến tròn** (23h gần 0h) nên được mã hoá sin/cos; giờ vào trung bình tính bằng trung bình vòng tròn.
- **Phân đoạn trước khi phân cụm**: xe chỉ đến 2–3 lần có hồ sơ rất nhiễu; phân cụm trực tiếp chỉ đạt ARI ≈ 0,70,
  phân đoạn theo tần suất trước rồi mới K-Means đạt ARI ≈ 0,98.
- **Chia dữ liệu theo thời gian** (14 ngày cuối làm test), không xáo trộn ngẫu nhiên, để tránh rò rỉ tương lai.
- Isolation Forest chỉ chạy trên lượt gửi **đã kết thúc** (xe đang trong bãi chưa có thời gian gửi đầy đủ).
- Toàn bộ hàm trong `core/analytics.py` nhận DataFrame nên có thể chạy độc lập trong notebook để viết báo cáo:

```python
from core.simulate import generate
from core import analytics as A
s, e, truth = generate(days=90, capacity=55)
s = A.prepare(s)
fig_k, fig_pca, profile, msg = A.clustering(s, truth)
```
