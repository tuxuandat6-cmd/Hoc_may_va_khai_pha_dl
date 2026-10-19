"""Hậu xử lý chuỗi biển số Việt Nam: làm sạch, sửa lỗi nhầm ký tự, kiểm tra định dạng, định dạng hiển thị."""
import re

# Vị trí bắt buộc là số mà OCR trả về chữ -> đổi sang số tương ứng
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "T": "1",
            "Z": "2", "S": "5", "B": "8", "G": "6"}
# Vị trí bắt buộc là chữ (ký tự seri thứ 3) mà OCR trả về số -> đổi sang chữ
TO_LETTER = {"0": "D", "8": "B", "5": "S", "2": "Z", "1": "T", "4": "A", "6": "G", "7": "T"}

# Biển dân sự: 2 số mã tỉnh + 1-2 chữ seri + 4-6 số (vd 29A12345, 59U102795, 60LD03645, 29MD271035)
CIVIL_RE = re.compile(r"^(\d{2})([A-ZĐ]{1,2})(\d{4,6})$")
# Biển quân đội / đặc biệt: 2 chữ + 4 số (vd TM5004, QA6085)
SPECIAL_RE = re.compile(r"^([A-Z]{2})(\d{4})$")


def clean(text: str) -> str:
    text = (text or "").upper()
    return re.sub(r"[^0-9A-ZĐ]", "", text)


def fix_confusions(t: str) -> str:
    """Sửa các lỗi nhầm phổ biến theo vị trí (0/O, 8/B, 5/S, ...)."""
    if len(t) < 6 or SPECIAL_RE.match(t):
        return t
    c = list(t)
    for i in (0, 1):                       # mã tỉnh luôn là số
        c[i] = TO_DIGIT.get(c[i], c[i])
    c[2] = TO_LETTER.get(c[2], c[2])       # ký tự seri đầu tiên luôn là chữ
    for i in range(len(c) - 4, len(c)):    # 4 ký tự cuối luôn là số
        c[i] = TO_DIGIT.get(c[i], c[i])
    return "".join(c)


def normalize(text: str) -> str:
    return fix_confusions(clean(text))


def is_valid(plate: str) -> bool:
    return bool(CIVIL_RE.match(plate) or SPECIAL_RE.match(plate))


def format_plate(plate: str) -> str:
    """29A12345 -> 29A-123.45 ; 59U102795 -> 59U1-027.95 ; TM5004 -> TM-50-04"""
    m = CIVIL_RE.match(plate)
    if m:
        prov, series, nums = m.groups()
        if len(nums) == 6:                 # có chữ số seri (xe máy): 59U1 + 02795
            series, nums = series + nums[0], nums[1:]
        tail = f"{nums[:3]}.{nums[3:]}" if len(nums) == 5 else nums
        return f"{prov}{series}-{tail}"
    m = SPECIAL_RE.match(plate)
    if m:
        return f"{m.group(1)}-{m.group(2)[:2]}-{m.group(2)[2:]}"
    return plate
