"""Cấu hình chung cho hệ thống bãi đỗ xe."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Trọng số mô hình ----------------
# Đặt file best.pt (YOLOv8s phát hiện biển số) và file .pth (CNN + BiLSTM + CTC) vào thư mục weights/
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", os.path.join(WEIGHTS_DIR, "yolov8s_plate.pt"))
OCR_WEIGHTS = os.environ.get("OCR_WEIGHTS", os.path.join(WEIGHTS_DIR, "crnn_bilstm_ctc.pth"))

# ---------------- Phát hiện biển số ----------------
DET_CONF = 0.5          # ngưỡng confidence của YOLO
DET_PAD_RATIO = 0.04    # nới rộng bounding box một chút trước khi cắt (tránh cắt mất ký tự)

# ---------------- OCR (CNN + BiLSTM + CTC) ----------------
OCR_IMG_H, OCR_IMG_W = 64, 256
# Bộ ký tự PHẢI GIỐNG HỆT lúc huấn luyện (index 0 = CTC blank, ký tự thứ i -> index i+1).
# Checkpoint do training/train_ocr.py tạo ra có lưu sẵn "charset" nên app tự dùng, không cần sửa ở đây.
CHARSET = os.environ.get("OCR_CHARSET", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ")
# Cách chuẩn hoá pixel lúc huấn luyện: "0_1" (x/255) hoặc "minus1_1" ((x/255-0.5)/0.5)
OCR_NORMALIZE = "0_1"

# ---------------- SQL Server ----------------
SQL_SERVER = os.environ.get("SQL_SERVER", "localhost")      # vd: "localhost", "localhost\\SQLEXPRESS"
SQL_DATABASE = os.environ.get("SQL_DATABASE", "ParkingDB")  # tự tạo nếu chưa có
# Windows Authentication (mặc định). Muốn dùng tài khoản sa: đặt SQL_TRUSTED = False và điền user/password.
SQL_TRUSTED = True
SQL_USER = os.environ.get("SQL_USER", "sa")
SQL_PASSWORD = os.environ.get("SQL_PASSWORD", "")
# Để trống = tự chọn driver ODBC mới nhất đã cài (ODBC Driver 18/17 for SQL Server)
SQL_DRIVER = os.environ.get("SQL_DRIVER", "")

# ---------------- Bãi xe ----------------
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
DEFAULT_CAPACITY = 50
# Trong khoảng thời gian này cùng một biển số sẽ không bị ghi nhận lặp lại (chống đếm trùng khi xử lý video)
COOLDOWN_SEC = 30

# ---------------- Xử lý video ----------------
VIDEO_FRAME_STEP = 5    # cứ 5 frame thì xử lý 1 frame
VIDEO_MIN_VOTES = 3     # một biển số phải được đọc giống nhau ít nhất N lần mới được xác nhận
TRACK_IOU = 0.3         # ngưỡng IoU để gom các bounding box cùng một xe
TRACK_MAX_GAP = 3       # số lần xử lý liên tiếp không thấy xe thì coi như xe đã đi khỏi khung hình

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
