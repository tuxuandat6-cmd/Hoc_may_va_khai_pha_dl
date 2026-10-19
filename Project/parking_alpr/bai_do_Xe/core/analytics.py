"""Khai phá dữ liệu lịch sử ra/vào bãi xe.

1. Thống kê mô tả & trực quan hoá   : lưu lượng theo giờ/thứ, tỷ lệ lấp đầy, thời gian gửi.
2. Phân cụm khách hàng (K-Means)     : chọn K bằng Elbow + Silhouette, đặt tên cụm theo hồ sơ.
3. Dự báo số xe trong bãi (hồi quy)  : Baseline vs Linear Regression vs Random Forest, MAE/RMSE/R².
4. Phát hiện bất thường              : Isolation Forest + luật nghiệp vụ.
5. Luật kết hợp (Apriori)            : "Nếu vào sáng sớm ngày thường thì gửi 8-12h".

Mọi hàm nhận DataFrame (sessions: Plate, TimeIn, TimeOut; events: Plate, Direction, EventTime, ...)
nên có thể chạy độc lập với CSDL.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (adjusted_rand_score, mean_absolute_error, mean_squared_error, r2_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- bảng màu (đã kiểm tra mù màu)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = [[0, "#eef4fc"], [0.5, "#6aa2e6"], [1, "#0f4a94"]]
CRITICAL, MUTED = "#d03b3b", "#9a9893"
DOW = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


def _layout(fig, title, x=None, y=None, height=340, legend=True):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        template="plotly_white", height=height, margin=dict(l=50, r=20, t=50, b=45),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#52514e"),
        hoverlabel=dict(bgcolor="white", font_size=12), showlegend=legend,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", title=None),
    )
    fig.update_xaxes(title=x, gridcolor="rgba(128,128,128,0.15)", zeroline=False, linecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(title=y, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


SIM_HINT = "Mở mục 'Dữ liệu mô phỏng' phía trên và bấm 'Sinh dữ liệu mô phỏng' để có dữ liệu nhiều tháng."


def _days(s):
    return 0 if s.empty else max(1, (s["TimeIn"].max() - s["TimeIn"].min()).days + 1)


def empty_fig(msg="Chưa có dữ liệu"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, "", height=260, legend=False)


# ---------------------------------------------------------------- tiền xử lý
def prepare(sessions, now=None):
    """Làm sạch & tạo đặc trưng cấp lượt gửi."""
    s = sessions.copy()
    now = pd.Timestamp(now or pd.Timestamp.now()).floor("s")
    s["Open"] = s["TimeOut"].isna()
    s["TimeOutEff"] = s["TimeOut"].fillna(now)
    s = s[s["TimeOutEff"] >= s["TimeIn"]]
    s["DurationH"] = (s["TimeOutEff"] - s["TimeIn"]).dt.total_seconds() / 3600
    s["HourIn"] = s["TimeIn"].dt.hour + s["TimeIn"].dt.minute / 60
    s["Dow"] = s["TimeIn"].dt.dayofweek
    s["Weekend"] = s["Dow"] >= 5
    s["Overnight"] = s["TimeOutEff"].dt.normalize() > s["TimeIn"].dt.normalize()
    s["PlateVisits"] = s.groupby("Plate")["Plate"].transform("size")
    return s.reset_index(drop=True)


def hourly_occupancy(s):
    """Số xe trong bãi tại đầu mỗi giờ (thuật toán quét: +1 lúc vào, -1 lúc ra)."""
    if s.empty:
        return pd.Series(dtype=float)
    delta = pd.concat([pd.Series(1, index=s["TimeIn"]), pd.Series(-1, index=s["TimeOutEff"])]).sort_index()
    cum = delta.groupby(level=0).sum().cumsum()
    grid = pd.date_range(s["TimeIn"].min().ceil("h"), s["TimeOutEff"].max().floor("h"), freq="h")
    pos = cum.index.searchsorted(grid, side="right") - 1
    vals = np.where(pos >= 0, cum.values[np.clip(pos, 0, None)], 0)
    return pd.Series(vals, index=grid, name="Occupancy").astype(float)


# ================================================================ 1. THỐNG KÊ MÔ TẢ
def descriptive(s, events, capacity):
    if s.empty:
        return "<p>Chưa có dữ liệu.</p>", *[empty_fig()] * 4
    occ = hourly_occupancy(s)
    closed = s[~s["Open"]]
    by_hour = s.groupby(s["TimeIn"].dt.hour).size().reindex(range(24), fill_value=0)
    n_days = max(1, (s["TimeIn"].max() - s["TimeIn"].min()).days)
    rejected = int((events["Note"].fillna("").str.contains("bãi đầy")).sum()) if not events.empty else 0
    full_hours = int((occ >= capacity).sum())
    kpis = [
        ("Lượt gửi", f"{len(s):,}".replace(",", "."), f"{n_days} ngày"),
        ("Biển số khác nhau", f"{s['Plate'].nunique():,}".replace(",", "."), ""),
        ("Thời gian gửi TB", f"{closed['DurationH'].mean():.1f} h", f"trung vị {closed['DurationH'].median():.1f} h"),
        ("Giờ cao điểm xe vào", f"{by_hour.idxmax()}h–{by_hour.idxmax() + 1}h", f"{by_hour.max() / n_days:.1f} xe/ngày"),
        ("Lấp đầy cao nhất", f"{occ.max() / capacity * 100:.0f}%", f"{int(occ.max())}/{capacity} chỗ"),
        ("Lượt bị từ chối (đầy)", str(rejected), f"{full_hours} giờ bãi đầy"),
    ]
    warn = (f"<div class='warn'>⚠️ Mới có {len(s)} lượt gửi trong {n_days} ngày — quá ít để rút ra quy luật. "
            f"{SIM_HINT}</div>") if n_days < 14 else ""
    html = warn + "<div class='kpis'>" + "".join(
        f"<div class='kpi'><div class='lbl'>{a}</div><div class='val'>{b}</div><div class='sub'>{c}</div></div>"
        for a, b, c in kpis) + "</div>"

    # Heatmap lượt xe vào trung bình / ngày theo (thứ, giờ)
    days_per_dow = s["TimeIn"].dt.normalize().drop_duplicates().dt.dayofweek.value_counts().reindex(range(7), fill_value=1)
    piv = s.pivot_table(index="Dow", columns=s["TimeIn"].dt.hour, values="Plate", aggfunc="size", fill_value=0)
    piv = piv.reindex(index=range(7), columns=range(24), fill_value=0).div(days_per_dow, axis=0)
    f1 = go.Figure(go.Heatmap(z=piv.values, x=[f"{h}h" for h in range(24)], y=DOW, colorscale=SEQ_BLUE,
                              xgap=2, ygap=2, colorbar=dict(title="xe/giờ", thickness=10),
                              hovertemplate="%{y} %{x}: %{z:.1f} xe vào<extra></extra>"))
    _layout(f1, "Lượt xe vào trung bình theo thứ và giờ", legend=False)
    f1.update_yaxes(autorange="reversed")

    # Tỷ lệ lấp đầy 14 ngày gần nhất
    last = occ[occ.index >= occ.index.max() - pd.Timedelta(days=14)]
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=last.index, y=last.values, mode="lines", line=dict(color=C[0], width=2),
                            name="Số xe trong bãi", hovertemplate="%{x|%d/%m %Hh}: %{y:.0f} xe<extra></extra>"))
    f2.add_hline(y=capacity, line=dict(color=CRITICAL, width=1.5, dash="dash"),
                 annotation_text=f"Sức chứa {capacity}", annotation_position="top left")
    _layout(f2, "Số xe trong bãi — 14 ngày gần nhất", y="Số xe", legend=False)
    f2.update_yaxes(range=[0, capacity * 1.08], tickformat="d")
    f2.update_layout(hovermode="x unified")

    # Phân bố thời gian gửi
    f3 = go.Figure(go.Histogram(x=closed["DurationH"].clip(upper=24), xbins=dict(start=0, end=24, size=0.5),
                                marker=dict(color=C[0], line=dict(color="white", width=1)),
                                hovertemplate="%{x} h: %{y} lượt<extra></extra>"))
    _layout(f3, "Phân bố thời gian gửi (gộp ≥ 24h)", x="Giờ", y="Số lượt", legend=False)

    # Lưu lượng vào/ra theo giờ: ngày thường vs cuối tuần
    f4 = go.Figure()
    for i, (lbl, mask) in enumerate((("Ngày thường", ~s["Weekend"]), ("Cuối tuần", s["Weekend"]))):
        nd = max(1, s.loc[mask, "TimeIn"].dt.normalize().nunique())
        y = s[mask].groupby(s.loc[mask, "TimeIn"].dt.hour).size().reindex(range(24), fill_value=0) / nd
        f4.add_trace(go.Scatter(x=list(range(24)), y=y, mode="lines+markers", name=lbl,
                                line=dict(color=C[i], width=2), marker=dict(size=8),
                                hovertemplate=f"{lbl} %{{x}}h: %{{y:.1f}} xe<extra></extra>"))
    _layout(f4, "Xe vào trung bình mỗi giờ", x="Giờ trong ngày", y="Xe / giờ")
    f4.update_layout(hovermode="x unified")
    return html, f1, f2, f3, f4


# ================================================================ 2. PHÂN CỤM K-MEANS
FEATURES = ["Lượt/tuần", "Giờ vào TB", "Gửi TB (h)", "Độ lệch giờ vào", "Tỷ lệ cuối tuần", "Tỷ lệ qua đêm"]


def plate_features(s, min_visits=2):
    weeks = max(1.0, (s["TimeIn"].max() - s["TimeIn"].min()).days / 7)
    g = s.groupby("Plate")
    ang = 2 * np.pi * s["HourIn"] / 24                      # giờ là biến tròn: 23h gần 0h
    sin_m, cos_m = np.sin(ang).groupby(s["Plate"]).mean(), np.cos(ang).groupby(s["Plate"]).mean()
    f = pd.DataFrame({
        "Lượt/tuần": g.size() / weeks,
        "Giờ vào TB": (np.degrees(np.arctan2(sin_m, cos_m)) / 15) % 24,
        "Gửi TB (h)": g["DurationH"].median(),
        "Độ lệch giờ vào": np.sqrt(-2 * np.log(np.clip(np.hypot(sin_m, cos_m), 1e-6, 1))) * 24 / (2 * np.pi),
        "Tỷ lệ cuối tuần": g["Weekend"].mean(),
        "Tỷ lệ qua đêm": g["Overnight"].mean(),
    })
    f["_visits"] = g.size()
    return f[f["_visits"] >= min_visits].drop(columns="_visits")


def _model_matrix(f):
    """Ma trận đưa vào K-Means: mã hoá giờ vào bằng sin/cos, log hoá tần suất & thời gian gửi."""
    a = 2 * np.pi * f["Giờ vào TB"] / 24
    X = pd.DataFrame({"freq": np.log1p(f["Lượt/tuần"]), "sin": np.sin(a), "cos": np.cos(a),
                      "dur": np.log1p(f["Gửi TB (h)"]), "std": f["Độ lệch giờ vào"].clip(upper=6) / 6,
                      "wkend": f["Tỷ lệ cuối tuần"], "night": f["Tỷ lệ qua đêm"]})
    return StandardScaler().fit_transform(X)


REGULAR_MIN_PER_WEEK = 0.6     # ngưỡng phân đoạn: dưới mức này là khách vãng lai
VISITOR = "Khách vãng lai"


def _name_cluster(p):
    if p["Tỷ lệ qua đêm"] > 0.5:
        return "Gửi qua đêm"
    if p["Tỷ lệ cuối tuần"] > 0.6:
        return "Khách cuối tuần"
    if p["Giờ vào TB"] < 10.5:
        return "Đi làm giờ hành chính"
    return "Làm ca chiều/tối"


def clustering(s, truth=None, k=None, kmin=2, kmax=8, seed=42):
    """Bước 1 — phân đoạn theo tần suất (kiểu RFM): < 0.6 lượt/tuần là khách vãng lai,
    vì xe chỉ đến 2-3 lần thì giờ vào/thời gian gửi trung bình không đáng tin, chỉ gây nhiễu.
    Bước 2 — K-Means trên nhóm khách quen, chọn K bằng Silhouette (Elbow để tham khảo)."""
    f_all = plate_features(s)
    reg = f_all["Lượt/tuần"] >= REGULAR_MIN_PER_WEEK
    f = f_all[reg].copy()
    if len(f) < kmax + 2:
        return (empty_fig("Cần thêm dữ liệu để phân cụm"), empty_fig(), pd.DataFrame(),
                f"Chưa đủ dữ liệu: cần ít nhất {kmax + 2} xe đến thường xuyên (≥ {REGULAR_MIN_PER_WEEK} lượt/tuần), "
                f"hiện có {len(f)} xe trong {_days(s)} ngày.\n{SIM_HINT}")
    X = _model_matrix(f)
    ks, inertia, sil = list(range(kmin, kmax + 1)), [], []
    for kk in ks:
        km = KMeans(kk, n_init=10, random_state=seed).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
    best_k = int(k) if k else ks[int(np.argmax(sil))]
    km = KMeans(best_k, n_init=10, random_state=seed).fit(X)
    f["Cụm"] = km.labels_
    best_sil = silhouette_score(X, km.labels_)

    prof = f.groupby("Cụm")[FEATURES].mean()
    names, seen = {}, {}
    for c, row in prof.iterrows():
        n = _name_cluster(row)
        seen[n] = seen.get(n, 0) + 1
        names[c] = n if seen[n] == 1 else f"{n} ({seen[n]})"
    f["Nhóm"] = f["Cụm"].map(names)
    f_all["Nhóm"] = VISITOR
    f_all.loc[f.index, "Nhóm"] = f["Nhóm"]
    order = f["Nhóm"].value_counts().index.tolist()

    # Elbow + Silhouette: 2 thang đo khác nhau -> 2 biểu đồ con dùng chung trục K (không dual-axis)
    from plotly.subplots import make_subplots
    f1 = make_subplots(rows=1, cols=2, subplot_titles=("Elbow — Inertia", "Silhouette (càng cao càng tốt)"))
    f1.add_trace(go.Scatter(x=ks, y=inertia, mode="lines+markers", line=dict(color=C[0], width=2),
                            marker=dict(size=8), hovertemplate="K=%{x}: %{y:.0f}<extra></extra>"), 1, 1)
    f1.add_trace(go.Scatter(x=ks, y=sil, mode="lines+markers", line=dict(color=C[0], width=2),
                            marker=dict(size=8), hovertemplate="K=%{x}: %{y:.3f}<extra></extra>"), 1, 2)
    f1.add_trace(go.Scatter(x=[best_k], y=[best_sil], mode="markers",
                            marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=CRITICAL, width=2)),
                            hovertemplate=f"K được chọn = {best_k}<extra></extra>"), 1, 2)
    _layout(f1, f"Chọn số cụm K cho khách quen — K = {best_k}", legend=False)
    f1.update_xaxes(title="K", dtick=1)

    # PCA 2D (khách vãng lai vẽ màu xám để làm nền)
    pca = PCA(2, random_state=seed).fit(X)
    xy = pca.transform(X)
    f2 = go.Figure()
    for i, n in enumerate(order):
        m = (f["Nhóm"] == n).values
        f2.add_trace(go.Scatter(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"{n} ({m.sum()})",
            marker=dict(size=9, color=C[i % len(C)], opacity=0.85, line=dict(color="white", width=1)),
            text=f.index[m], customdata=f.loc[m, ["Lượt/tuần", "Giờ vào TB", "Gửi TB (h)"]].values,
            hovertemplate="<b>%{text}</b><br>%{customdata[0]:.1f} lượt/tuần · vào %{customdata[1]:.1f}h"
                          " · gửi %{customdata[2]:.1f}h<extra>" + n + "</extra>"))
    var = pca.explained_variance_ratio_ * 100
    _layout(f2, "Các nhóm khách quen (chiếu PCA 2 chiều)", x=f"PC1 ({var[0]:.0f}% phương sai)",
            y=f"PC2 ({var[1]:.0f}% phương sai)", height=420)

    table = f_all.groupby("Nhóm")[FEATURES].mean().round(2)
    table.insert(0, "Số xe", f_all["Nhóm"].value_counts())
    table = table.reindex(order + [VISITOR]).reset_index()

    msg = (f"Bước 1 — phân đoạn theo tần suất: {int((~reg).sum())} xe có < {REGULAR_MIN_PER_WEEK} lượt/tuần "
           f"→ '{VISITOR}'.\nBước 2 — K-Means trên {len(f)} khách quen, {X.shape[1]} đặc trưng chuẩn hoá. "
           f"K = {best_k} ({'tự chọn theo Silhouette' if not k else 'do người dùng chọn'}), "
           f"Silhouette = {best_sil:.3f}.")
    if truth is not None:
        t = truth.set_index("Plate").reindex(f_all.index)["Group"]
        ok = t.notna()
        if ok.sum() > 10:
            ari = adjusted_rand_score(t[ok], f_all.loc[ok, "Nhóm"])
            msg += (f"\nĐối chiếu nhãn thật của dữ liệu mô phỏng: Adjusted Rand Index = {ari:.3f} "
                    "(1 = khớp hoàn toàn, 0 = ngẫu nhiên).")
    return f1, f2, table, msg


# ================================================================ 3. DỰ BÁO SỐ XE TRONG BÃI
def _forecast_frame(occ, h):
    df = pd.DataFrame({"y_now": occ})
    for lag in (1, 2, 3, 24, 168):
        df[f"lag{lag}"] = occ.shift(lag)
    df["roll3"] = occ.rolling(3).mean()
    tgt_time = df.index + pd.Timedelta(hours=h)
    a = 2 * np.pi * tgt_time.hour / 24
    df["h_sin"], df["h_cos"] = np.sin(a), np.cos(a)
    df["dow"] = tgt_time.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["same_hour_last_week"] = occ.shift(168 - h)      # giá trị cùng giờ tuần trước của thời điểm đích
    df["same_hour_yesterday"] = occ.shift(24 - h)
    df["target"] = occ.shift(-h)
    return df


def forecast(s, capacity, horizon=1, test_days=14, future_hours=12, seed=42):
    occ = hourly_occupancy(s)
    if len(occ) < 24 * (test_days + 10):
        need = test_days + 10
        return (empty_fig(f"Cần ≥ {need} ngày dữ liệu để dự báo"),
                pd.DataFrame(columns=["Mô hình", "MAE (xe)", "RMSE (xe)", "R²"]), empty_fig(),
                f"Chưa đủ dữ liệu: dự báo cần ít nhất {need} ngày lịch sử, hiện có {_days(s)} ngày.\n{SIM_HINT}")
    df = _forecast_frame(occ, horizon).dropna()
    feats = [c for c in df.columns if c != "target"]
    split = df.index.max() - pd.Timedelta(days=test_days)
    tr, te = df[df.index <= split], df[df.index > split]

    models = {
        "Baseline (cùng giờ tuần trước)": None,
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
    }
    preds, rows = {}, []
    for name, m in models.items():
        p = te["same_hour_last_week"].values if m is None else m.fit(tr[feats], tr["target"]).predict(te[feats])
        p = np.clip(p, 0, capacity)
        preds[name] = p
        rows.append((name, mean_absolute_error(te["target"], p),
                     np.sqrt(mean_squared_error(te["target"], p)), r2_score(te["target"], p)))
    metrics = pd.DataFrame(rows, columns=["Mô hình", "MAE (xe)", "RMSE (xe)", "R²"]).round(3)
    best = metrics.loc[metrics["MAE (xe)"].idxmin(), "Mô hình"]

    tgt_idx = te.index + pd.Timedelta(hours=horizon)
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=tgt_idx, y=te["target"], name="Thực tế", mode="lines",
                            line=dict(color=MUTED, width=2)))
    for i, name in enumerate(["Random Forest", "Linear Regression"]):
        f1.add_trace(go.Scatter(x=tgt_idx, y=preds[name], name=name, mode="lines",
                                line=dict(color=C[i], width=2, dash=None if i == 0 else "dot")))
    _layout(f1, f"Tập kiểm thử {test_days} ngày cuối — dự báo trước {horizon} giờ", y="Số xe trong bãi")
    f1.update_layout(hovermode="x unified")
    f1.update_traces(hovertemplate="%{y:.0f} xe")

    # Dự báo các giờ tới: mô hình trực tiếp (direct) — mỗi bước k dùng một Random Forest riêng
    now_t, fut = occ.index.max(), []
    for k in range(1, future_hours + 1):
        d = _forecast_frame(occ, k)
        train = d.dropna()
        rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        rf.fit(train[feats], train["target"])
        x_last = d.loc[[now_t], feats]
        fut.append((now_t + pd.Timedelta(hours=k), float(np.clip(rf.predict(x_last)[0], 0, capacity))))
    fut = pd.DataFrame(fut, columns=["time", "pred"])
    hist = occ[occ.index >= now_t - pd.Timedelta(hours=36)]
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Đã xảy ra", mode="lines",
                            line=dict(color=C[0], width=2)))
    f2.add_trace(go.Scatter(x=[now_t, *fut["time"]], y=[hist.iloc[-1], *fut["pred"]], name="Dự báo",
                            mode="lines+markers", line=dict(color=C[1], width=2, dash="dash"), marker=dict(size=8)))
    f2.add_hline(y=capacity, line=dict(color=CRITICAL, width=1.5, dash="dash"),
                 annotation_text=f"Sức chứa {capacity}", annotation_position="top left")
    _layout(f2, f"Dự báo số xe {future_hours} giờ tới", y="Số xe")
    f2.update_layout(hovermode="x unified")
    f2.update_traces(hovertemplate="%{y:.0f} xe")

    peak = fut.loc[fut["pred"].idxmax()]
    msg = (f"Mô hình tốt nhất trên tập kiểm thử: {best} "
           f"(MAE = {metrics['MAE (xe)'].min():.2f} xe). Huấn luyện {len(tr)} mẫu, kiểm thử {len(te)} mẫu "
           f"(chia theo thời gian, không xáo trộn để tránh rò rỉ dữ liệu tương lai).\n"
           f"Dự báo cao điểm sắp tới: ~{peak['pred']:.0f} xe lúc {peak['time']:%H:%M %d/%m} "
           f"→ còn khoảng {max(0, capacity - round(peak['pred']))} chỗ trống.")
    return f1, metrics, f2, msg


# ================================================================ 4. PHÁT HIỆN BẤT THƯỜNG
def anomalies(s, events, contamination=0.01, seed=42):
    if len(s) < 50:
        return (empty_fig("Chưa đủ dữ liệu"),
                pd.DataFrame(columns=["Phát hiện bởi", "Biển số", "Thời điểm", "Gửi (h)", "Lý do", "Điểm bất thường"]),
                f"Chưa đủ dữ liệu: cần ít nhất 50 lượt gửi, hiện có {len(s)} lượt.\n{SIM_HINT}")
    d = s[~s["Open"]].copy()          # xe đang trong bãi chưa có thời gian gửi hoàn chỉnh
    g = d.groupby("Plate")["DurationH"]
    d["DurDev"] = ((d["DurationH"] - g.transform("median")) / (g.transform("std").fillna(0) + 1)).fillna(0)
    a = 2 * np.pi * d["HourIn"] / 24
    X = np.column_stack([np.log1p(d["DurationH"]), np.sin(a), np.cos(a), d["Weekend"],
                         np.log1p(d["PlateVisits"]), d["DurDev"].clip(-10, 10)])
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    d["Score"] = -iso.score_samples(X)
    d["IF"] = iso.predict(X) == -1

    p99_dur = d["DurationH"].quantile(0.99)
    hour_freq = d["TimeIn"].dt.hour.map(d["TimeIn"].dt.hour.value_counts(normalize=True))
    rows = []
    for i, r in d[d["IF"]].iterrows():
        why = []
        if r["DurationH"] > p99_dur:
            why.append(f"gửi lâu hơn 99% số lượt ({r['DurationH']:.0f}h)")
        if abs(r["DurDev"]) > 3:
            why.append("khác hẳn thói quen của xe này")
        if hour_freq[i] < 0.01:
            why.append(f"vào lúc {int(r['HourIn'])}h — khung giờ rất ít xe")
        if r["PlateVisits"] >= 10 and r["Weekend"] and not why:
            why.append("khách quen nhưng đến vào cuối tuần")
        rows.append(("Isolation Forest", r["Plate"], r["TimeIn"], f"{r['DurationH']:.1f}",
                     ", ".join(why) or "tổ hợp giờ vào / thời gian gửi hiếm gặp", round(r["Score"], 3)))
    for _, r in s[s["DurationH"] > 24].iterrows():
        rows.append(("Luật: gửi > 24h", r["Plate"], r["TimeIn"], f"{r['DurationH']:.1f}",
                     "Xe còn trong bãi > 24h — có thể bị bỏ quên" if r["Open"] else "Gửi nhiều ngày liên tục", None))
    if not events.empty:
        for _, r in events[events["Note"].fillna("").str.contains("không có lượt vào")].iterrows():
            rows.append(("Luật: ra không có lượt vào", r["Plate"], r["EventTime"], "",
                         f"Nghi OCR đọc sai (độ tin cậy {r['Confidence']:.2f})" if pd.notna(r["Confidence"])
                         else "Nghi OCR đọc sai", None))
    table = pd.DataFrame(rows, columns=["Phát hiện bởi", "Biển số", "Thời điểm", "Gửi (h)", "Lý do", "Điểm bất thường"])
    table = table.sort_values(["Điểm bất thường", "Thời điểm"], ascending=False, na_position="last")
    table["Thời điểm"] = pd.to_datetime(table["Thời điểm"]).dt.strftime("%Y-%m-%d %H:%M")

    norm = d[~d["IF"]]
    sample = norm.sample(min(len(norm), 3000), random_state=seed)
    f = go.Figure()
    f.add_trace(go.Scattergl(x=sample["HourIn"], y=sample["DurationH"], mode="markers", name="Bình thường",
                             marker=dict(size=6, color=C[0], opacity=0.35),
                             hovertemplate="vào %{x:.1f}h · gửi %{y:.1f}h<extra></extra>"))
    an = d[d["IF"]]
    f.add_trace(go.Scatter(x=an["HourIn"], y=an["DurationH"], mode="markers", name=f"Bất thường ({len(an)})",
                           marker=dict(size=10, color=CRITICAL, symbol="x-thin", line=dict(width=2.5, color=CRITICAL)),
                           text=an["Plate"], hovertemplate="<b>%{text}</b><br>vào %{x:.1f}h · gửi %{y:.1f}h<extra></extra>"))
    _layout(f, "Giờ vào × thời gian gửi (trục dọc log)", x="Giờ vào", y="Thời gian gửi (giờ)", height=420)
    ticks = [0.25, 0.5, 1, 2, 4, 8, 12, 24, 48, 96]
    f.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{t:g}" for t in ticks])
    f.update_xaxes(range=[0, 24], dtick=3)
    msg = (f"Isolation Forest (300 cây, contamination = {contamination:.1%}) đánh dấu {int(d['IF'].sum())} / {len(d)} "
           f"lượt gửi bất thường. Luật nghiệp vụ phát hiện thêm "
           f"{int((table['Phát hiện bởi'] != 'Isolation Forest').sum())} trường hợp.")
    return f, table, msg


# ================================================================ 5. LUẬT KẾT HỢP (APRIORI)
def _items(r):
    h = r["HourIn"]
    slot = ("Sáng sớm (5–9h)" if 5 <= h < 9 else "Sáng (9–12h)" if 9 <= h < 12 else
            "Chiều (12–17h)" if 12 <= h < 17 else "Tối (17–22h)" if 17 <= h < 22 else "Đêm (22–5h)")
    dur = r["DurationH"]
    dlabel = ("< 1h" if dur < 1 else "1–3h" if dur < 3 else "3–8h" if dur < 8 else "8–12h" if dur < 12 else "> 12h")
    freq = "Khách quen" if r["PlateVisits"] >= 10 else "Khách ít đến"
    return [f"Ngày={'Cuối tuần' if r['Weekend'] else 'Ngày thường'}", f"Vào={slot}", f"Gửi={dlabel}",
            f"Khách={freq}", f"Qua đêm={'Có' if r['Overnight'] else 'Không'}"]


def association_rules(s, min_support=0.05, min_confidence=0.6, top=20):
    from mlxtend.frequent_patterns import apriori, association_rules as ar
    from mlxtend.preprocessing import TransactionEncoder
    closed = s[~s["Open"]]
    if len(closed) < 50:
        return (pd.DataFrame(columns=["Nếu", "Thì", "Support", "Confidence", "Lift"]),
                f"Chưa đủ dữ liệu: cần ít nhất 50 lượt gửi đã kết thúc, hiện có {len(closed)} lượt.\n{SIM_HINT}")
    tx = closed.apply(_items, axis=1).tolist()
    te = TransactionEncoder()
    onehot = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame(), "Không có tập phổ biến — hãy giảm min support."
    rules = ar(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] > 1.05]
    # "Gửi" và "Qua đêm" được suy ra từ cùng giờ vào/ra nên luật nối 2 thuộc tính này là hiển nhiên
    # (vd "gửi > 12h => qua đêm"), không phải tri thức mới -> loại bỏ
    attr = lambda items: {i.split("=")[0] for i in items}
    derived = lambda r: (attr(r["consequents"]) <= {"Gửi", "Qua đêm"}
                         and attr(r["antecedents"]) & ({"Gửi", "Qua đêm"} - attr(r["consequents"])))
    rules = rules[~rules.apply(lambda r: bool(derived(r)), axis=1)]
    # bỏ luật hiển nhiên do cách rời rạc hoá (vd "Gửi > 12h" => "Qua đêm")
    rules = rules[rules["consequents"].apply(len) == 1].sort_values(["lift", "confidence"], ascending=False)
    # tỉa luật dư thừa: bỏ luật có vế trái là tập cha của một luật đơn giản hơn cùng vế phải mà
    # không tăng confidence đáng kể (vd "A & B => C" khi "A => C" đã có confidence tương đương)
    keep = []
    for idx, r in rules.sort_values("antecedents", key=lambda x: x.apply(len)).iterrows():
        if not any(rules.at[j, "consequents"] == r["consequents"] and rules.at[j, "antecedents"] < r["antecedents"]
                   and r["confidence"] - rules.at[j, "confidence"] < 0.05 for j in keep):
            keep.append(idx)
    rules = rules.loc[keep].sort_values(["lift", "confidence"], ascending=False).head(top)
    fmt = lambda x: " & ".join(sorted(x))
    out = pd.DataFrame({
        "Nếu": rules["antecedents"].apply(fmt), "Thì": rules["consequents"].apply(fmt),
        "Support": rules["support"].round(3), "Confidence": rules["confidence"].round(3),
        "Lift": rules["lift"].round(2)})
    msg = (f"Apriori trên {len(tx)} giao dịch (mỗi lượt gửi = 1 giao dịch gồm 5 thuộc tính đã rời rạc hoá), "
           f"tìm được {len(freq)} tập phổ biến, hiển thị {len(out)} luật có lift cao nhất.")
    return out.reset_index(drop=True), msg