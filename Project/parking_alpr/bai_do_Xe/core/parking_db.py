"""Quản lý bãi đỗ xe bằng SQL Server: sức chứa, xe đang trong bãi, lịch sử ra/vào."""
import threading
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
import pyodbc

import config

ENTRY, EXIT = "VÀO", "RA"

SCHEMA = [
    """IF OBJECT_ID('dbo.Settings') IS NULL
       CREATE TABLE dbo.Settings (
           [Key]   NVARCHAR(50)  NOT NULL PRIMARY KEY,
           [Value] NVARCHAR(200) NOT NULL)""",
    """IF OBJECT_ID('dbo.ParkingSessions') IS NULL
       CREATE TABLE dbo.ParkingSessions (              -- mỗi lượt gửi xe
           Id      INT IDENTITY(1,1) PRIMARY KEY,
           Plate   NVARCHAR(20)  NOT NULL,
           TimeIn  DATETIME2(0)  NOT NULL,
           TimeOut DATETIME2(0)  NULL,
           ImgIn   NVARCHAR(400) NULL,
           ImgOut  NVARCHAR(400) NULL)""",
    """IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Sessions_Open')
       CREATE INDEX IX_Sessions_Open ON dbo.ParkingSessions(Plate, TimeOut)""",
    """IF OBJECT_ID('dbo.ParkingEvents') IS NULL
       CREATE TABLE dbo.ParkingEvents (                -- nhật ký từng lần quét
           Id         INT IDENTITY(1,1) PRIMARY KEY,
           Plate      NVARCHAR(20)  NOT NULL,
           Direction  NVARCHAR(10)  NOT NULL,
           EventTime  DATETIME2(0)  NOT NULL,
           Source     NVARCHAR(50)  NULL,
           Confidence FLOAT         NULL,
           Snapshot   NVARCHAR(400) NULL,
           Note       NVARCHAR(200) NULL)""",
    """IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Events_Plate')
       CREATE INDEX IX_Events_Plate ON dbo.ParkingEvents(Plate, Id DESC)""",
]


def _driver():
    if config.SQL_DRIVER:
        return config.SQL_DRIVER
    found = [d for d in pyodbc.drivers() if "SQL Server" in d]
    if not found:
        raise RuntimeError("Chưa cài ODBC Driver cho SQL Server. Tải 'ODBC Driver 18 for SQL Server' từ Microsoft.")
    odbc = sorted(d for d in found if d.startswith("ODBC Driver"))
    return odbc[-1] if odbc else found[-1]


def conn_str(database):
    s = f"DRIVER={{{_driver()}}};SERVER={config.SQL_SERVER};DATABASE={database};TrustServerCertificate=yes;"
    if config.SQL_TRUSTED:
        s += "Trusted_Connection=yes;"
    else:
        s += f"UID={config.SQL_USER};PWD={config.SQL_PASSWORD};"
    return s


def _fmt(t):
    return t.strftime("%Y-%m-%d %H:%M:%S") if t else None


class ParkingLot:
    def __init__(self):
        self.lock = threading.Lock()
        self._ensure_database()
        with self._conn() as c:
            cur = c.cursor()
            for stmt in SCHEMA:
                cur.execute(stmt)
            cur.execute("IF NOT EXISTS (SELECT 1 FROM dbo.Settings WHERE [Key]='capacity') "
                        "INSERT INTO dbo.Settings VALUES ('capacity', ?)", str(config.DEFAULT_CAPACITY))

    @staticmethod
    def _ensure_database():
        """Tạo database ParkingDB nếu chưa có (CREATE DATABASE phải chạy với autocommit)."""
        try:
            c = pyodbc.connect(conn_str("master"), autocommit=True, timeout=5)
        except pyodbc.Error as e:
            raise RuntimeError(f"Không kết nối được SQL Server '{config.SQL_SERVER}'. Kiểm tra service SQL Server "
                               f"đang chạy và tên server (vd 'localhost\\SQLEXPRESS').\nChi tiết: {e}") from e
        try:
            c.cursor().execute(f"IF DB_ID(N'{config.SQL_DATABASE}') IS NULL CREATE DATABASE [{config.SQL_DATABASE}]")
        finally:
            c.close()

    @contextmanager
    def _conn(self):
        c = pyodbc.connect(conn_str(config.SQL_DATABASE), timeout=10)
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    # ---------- trạng thái ----------
    @staticmethod
    def _capacity(cur):
        return int(cur.execute("SELECT [Value] FROM dbo.Settings WHERE [Key]='capacity'").fetchval())

    @staticmethod
    def _occupied(cur):
        return cur.execute("SELECT COUNT(*) FROM dbo.ParkingSessions WHERE TimeOut IS NULL").fetchval()

    @property
    def capacity(self):
        with self._conn() as c:
            return self._capacity(c.cursor())

    def occupied(self):
        with self._conn() as c:
            return self._occupied(c.cursor())

    def free(self):
        with self._conn() as c:
            cur = c.cursor()
            return self._capacity(cur) - self._occupied(cur)

    def set_capacity(self, n):
        n = int(n)
        with self._conn() as c:
            cur = c.cursor()
            used = self._occupied(cur)
            if n < used:
                raise ValueError(f"Sức chứa ({n}) không thể nhỏ hơn số xe đang trong bãi ({used}).")
            cur.execute("UPDATE dbo.Settings SET [Value]=? WHERE [Key]='capacity'", str(n))

    def is_inside(self, plate):
        with self._conn() as c:
            return c.cursor().execute("SELECT 1 FROM dbo.ParkingSessions WHERE Plate=? AND TimeOut IS NULL",
                                      plate).fetchone() is not None

    # ---------- ghi nhận xe ----------
    @staticmethod
    def _log(cur, plate, direction, now, source, confidence, snapshot, note=None):
        cur.execute("INSERT INTO dbo.ParkingEvents(Plate,Direction,EventTime,Source,Confidence,Snapshot,Note) "
                    "VALUES (?,?,?,?,?,?,?)", plate, direction, now, source, confidence, snapshot, note)

    def register(self, plate, direction="auto", source="", confidence=None, snapshot=None, cooldown=True):
        """Ghi nhận một lượt quét.
        direction: "in" | "out" | "auto" (tự đoán: xe đang trong bãi -> RA, ngược lại -> VÀO).
        Trả về (ok: bool, message: str)."""
        now = datetime.now().replace(microsecond=0)
        confidence = float(confidence) if confidence is not None else None
        with self.lock, self._conn() as c:
            cur = c.cursor()
            last = cur.execute("SELECT TOP 1 EventTime FROM dbo.ParkingEvents WHERE Plate=? ORDER BY Id DESC",
                               plate).fetchval()
            if cooldown and last is not None and (now - last).total_seconds() < config.COOLDOWN_SEC:
                return False, f"{plate}: vừa được ghi nhận {int((now - last).total_seconds())}s trước, bỏ qua (chống trùng)."

            inside = cur.execute("SELECT Id, TimeIn FROM dbo.ParkingSessions WITH (UPDLOCK) "
                                 "WHERE Plate=? AND TimeOut IS NULL", plate).fetchone()
            if direction == "auto":
                direction = "out" if inside else "in"

            if direction == "in":
                if inside:
                    return False, f"{plate}: xe đã ở trong bãi, không thể vào lần nữa."
                used, cap = self._occupied(cur), self._capacity(cur)
                if used >= cap:
                    self._log(cur, plate, ENTRY, now, source, confidence, snapshot, "TỪ CHỐI - bãi đầy")
                    return False, f"{plate}: BÃI ĐÃ ĐẦY ({used}/{cap}), không cho xe vào."
                cur.execute("INSERT INTO dbo.ParkingSessions(Plate,TimeIn,ImgIn) VALUES (?,?,?)", plate, now, snapshot)
                self._log(cur, plate, ENTRY, now, source, confidence, snapshot)
                return True, f"✅ {plate} VÀO bãi lúc {_fmt(now)}. Còn trống {cap - used - 1}/{cap} chỗ."

            # direction == "out"
            if not inside:
                self._log(cur, plate, EXIT, now, source, confidence, snapshot, "CẢNH BÁO - không có lượt vào")
                return False, f"⚠️ {plate}: không tìm thấy lượt vào của xe này (kiểm tra lại biển số)."
            cur.execute("UPDATE dbo.ParkingSessions SET TimeOut=?, ImgOut=? WHERE Id=?", now, snapshot, inside.Id)
            self._log(cur, plate, EXIT, now, source, confidence, snapshot)
            h, rem = divmod(int((now - inside.TimeIn).total_seconds()), 3600)
            return True, f"✅ {plate} RA bãi lúc {_fmt(now)} (đã gửi {h}h{rem // 60:02d}p)."

    # ---------- truy vấn ----------
    def vehicles_inside(self):
        with self._conn() as c:
            rows = c.cursor().execute("SELECT Plate, TimeIn FROM dbo.ParkingSessions "
                                      "WHERE TimeOut IS NULL ORDER BY TimeIn DESC").fetchall()
        return [(r.Plate, _fmt(r.TimeIn)) for r in rows]

    def history(self, plate_filter="", limit=500):
        q = f"SELECT TOP {int(limit)} EventTime, Plate, Direction, Source, Confidence, Note FROM dbo.ParkingEvents"
        args = []
        if plate_filter:
            q += " WHERE Plate LIKE ?"
            args.append(f"%{plate_filter.upper()}%")
        q += " ORDER BY Id DESC"
        with self._conn() as c:
            rows = c.cursor().execute(q, *args).fetchall()
        return [(_fmt(r.EventTime), r.Plate, r.Direction, r.Source, r.Confidence, r.Note) for r in rows]

    def reset(self):
        with self.lock, self._conn() as c:
            c.cursor().execute("DELETE FROM dbo.ParkingSessions; DELETE FROM dbo.ParkingEvents;")

    # ---------- dữ liệu cho khai phá ----------
    def load_sessions(self):
        with self._conn() as c:
            rows = c.cursor().execute("SELECT Plate, TimeIn, TimeOut FROM dbo.ParkingSessions").fetchall()
        df = pd.DataFrame([tuple(r) for r in rows], columns=["Plate", "TimeIn", "TimeOut"])
        df["TimeIn"], df["TimeOut"] = pd.to_datetime(df["TimeIn"]), pd.to_datetime(df["TimeOut"])
        return df

    def load_events(self):
        with self._conn() as c:
            rows = c.cursor().execute("SELECT Plate, Direction, EventTime, Source, Confidence, Note "
                                      "FROM dbo.ParkingEvents").fetchall()
        df = pd.DataFrame([tuple(r) for r in rows],
                          columns=["Plate", "Direction", "EventTime", "Source", "Confidence", "Note"])
        df["EventTime"] = pd.to_datetime(df["EventTime"])
        return df

    def bulk_load(self, sessions, events, capacity=None):
        """Xoá dữ liệu cũ và nạp hàng loạt dữ liệu (dùng cho dữ liệu mô phỏng)."""
        def py(v):
            if v is None or (not isinstance(v, str) and pd.isna(v)):
                return None
            return v.to_pydatetime() if isinstance(v, pd.Timestamp) else v

        s_rows = [tuple(py(v) for v in r) for r in sessions[["Plate", "TimeIn", "TimeOut"]].itertuples(index=False)]
        e_rows = [tuple(py(v) for v in r) for r in events[["Plate", "Direction", "EventTime", "Source",
                                                            "Confidence", "Note"]].itertuples(index=False)]
        with self.lock, self._conn() as c:
            cur = c.cursor()
            cur.fast_executemany = True
            cur.execute("DELETE FROM dbo.ParkingSessions; DELETE FROM dbo.ParkingEvents;")
            if s_rows:
                cur.executemany("INSERT INTO dbo.ParkingSessions(Plate,TimeIn,TimeOut) VALUES (?,?,?)", s_rows)
            if e_rows:
                cur.executemany("INSERT INTO dbo.ParkingEvents(Plate,Direction,EventTime,Source,Confidence,Note) "
                                "VALUES (?,?,?,?,?,?)", e_rows)
            if capacity:
                cur.execute("UPDATE dbo.Settings SET [Value]=? WHERE [Key]='capacity'", str(int(capacity)))
