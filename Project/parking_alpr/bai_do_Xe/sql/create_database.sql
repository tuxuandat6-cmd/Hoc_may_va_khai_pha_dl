-- Tạo CSDL cho hệ thống bãi đỗ xe (chạy trong SSMS, server: localhost).
-- Không bắt buộc: app.py tự tạo database và bảng nếu chưa có.
IF DB_ID(N'ParkingDB') IS NULL CREATE DATABASE ParkingDB;
GO
USE ParkingDB;
GO

IF OBJECT_ID('dbo.Settings') IS NULL
CREATE TABLE dbo.Settings (
    [Key]   NVARCHAR(50)  NOT NULL PRIMARY KEY,
    [Value] NVARCHAR(200) NOT NULL
);
IF NOT EXISTS (SELECT 1 FROM dbo.Settings WHERE [Key] = 'capacity')
    INSERT INTO dbo.Settings VALUES ('capacity', '50');

-- Mỗi lượt gửi xe: TimeOut = NULL nghĩa là xe đang trong bãi
IF OBJECT_ID('dbo.ParkingSessions') IS NULL
CREATE TABLE dbo.ParkingSessions (
    Id      INT IDENTITY(1,1) PRIMARY KEY,
    Plate   NVARCHAR(20)  NOT NULL,
    TimeIn  DATETIME2(0)  NOT NULL,
    TimeOut DATETIME2(0)  NULL,
    ImgIn   NVARCHAR(400) NULL,
    ImgOut  NVARCHAR(400) NULL
);
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Sessions_Open')
    CREATE INDEX IX_Sessions_Open ON dbo.ParkingSessions(Plate, TimeOut);

-- Nhật ký từng lần quét (gồm cả lượt bị từ chối / cảnh báo)
IF OBJECT_ID('dbo.ParkingEvents') IS NULL
CREATE TABLE dbo.ParkingEvents (
    Id         INT IDENTITY(1,1) PRIMARY KEY,
    Plate      NVARCHAR(20)  NOT NULL,
    Direction  NVARCHAR(10)  NOT NULL,   -- N'VÀO' / N'RA'
    EventTime  DATETIME2(0)  NOT NULL,
    Source     NVARCHAR(50)  NULL,       -- ảnh / video / nhập tay
    Confidence FLOAT         NULL,
    Snapshot   NVARCHAR(400) NULL,
    Note       NVARCHAR(200) NULL
);
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Events_Plate')
    CREATE INDEX IX_Events_Plate ON dbo.ParkingEvents(Plate, Id DESC);
GO

-- ===== Một số truy vấn hữu ích =====
-- Số chỗ trống hiện tại
SELECT CAST(s.[Value] AS INT) AS SucChua,
       (SELECT COUNT(*) FROM dbo.ParkingSessions WHERE TimeOut IS NULL) AS DangDo,
       CAST(s.[Value] AS INT) - (SELECT COUNT(*) FROM dbo.ParkingSessions WHERE TimeOut IS NULL) AS ConTrong
FROM dbo.Settings s WHERE s.[Key] = 'capacity';

-- Xe đang trong bãi
SELECT Plate, TimeIn FROM dbo.ParkingSessions WHERE TimeOut IS NULL ORDER BY TimeIn DESC;

-- Lịch sử ra vào hôm nay
SELECT EventTime, Plate, Direction, Source, Note
FROM dbo.ParkingEvents
WHERE CAST(EventTime AS DATE) = CAST(SYSDATETIME() AS DATE)
ORDER BY Id DESC;
