"""Streamlit dashboard: Beijing multi-site air quality (PRSA), Mar 2013 - Feb 2017.

Run from the project root:  streamlit run dashboard/dashboard.py
"""
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "main_data.csv"

# Design tokens (same as the notebook): gray = context, red = worst, green = best, blue = baseline
GRAY, BLUE, RED, GREEN = "#B8BEC7", "#2F5D8C", "#C0392B", "#2E8B6B"
INK, MUTED = "#2B2F36", "#5B6270"
LIMIT_DAILY, LIMIT_ANNUAL = 75, 35  # China GB 3095-2012 grade II, ug/m3
SOURCE = "Sumber: Beijing Multi-Site Air-Quality Data (PRSA), 1 Mar 2013 – 28 Feb 2017"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
SEASONS = {12: "Dingin", 1: "Dingin", 2: "Dingin", 3: "Semi", 4: "Semi", 5: "Semi",
           6: "Panas", 7: "Panas", 8: "Panas", 9: "Gugur", 10: "Gugur", 11: "Gugur"}
SEASON_ORDER = ["Dingin", "Semi", "Panas", "Gugur"]
WIND_BINS = [0, 0.5, 1, 1.5, 2, 3, 4, np.inf]
WIND_LABELS = ["≤0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3–4", ">4"]
AQI_BINS = [0, 35, 75, 115, 150, 250, np.inf]
AQI_LABELS = ["Sangat baik", "Baik", "Tercemar ringan", "Tercemar sedang", "Tercemar berat", "Tercemar sangat berat"]
AQI_COLORS = ["#2E8B6B", "#A7C77B", "#F2C14E", "#E8833A", "#C0392B", "#6C2B5A"]
AQI_TEXT = ["white", INK, INK, INK, "white", "white"]
NAMES = {"CO": "CO", "NO2": "NO₂", "SO2": "SO₂", "O3": "O₃", "WSPM": "Kecepatan angin", "TEMP": "Suhu",
         "DEWP": "Titik embun", "PRES": "Tekanan udara", "RAIN": "Curah hujan"}
WEATHER = ["WSPM", "TEMP", "DEWP", "PRES", "RAIN"]
CORR_COLS = ["CO", "NO2", "SO2", "O3"] + WEATHER

# Approximate station locations (not in the dataset; used for visualization only)
COORDS = {
    "Aotizhongxin": (39.982, 116.397), "Changping": (40.217, 116.230), "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417), "Guanyuan": (39.929, 116.339), "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628), "Nongzhanguan": (39.937, 116.461), "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407), "Wanliu": (39.987, 116.287), "Wanshouxigong": (39.878, 116.352),
}
CENTER = (39.9087, 116.3975)  # Tiananmen Square
ZONES = ["Pusat kota (≤10 km)", "Pinggiran dalam (10–30 km)", "Pinggiran luar (>30 km)"]

plt.rcParams.update({
    "figure.dpi": 100, "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8EAEE", "axes.axisbelow": True,
})


# --------------------------------------------------------------------------- data
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def period_label(dates):
    """March-February period label, e.g. 2013/14."""
    start = dates.dt.year - (dates.dt.month < 3).astype(int)
    return start.astype(str) + "/" + ((start + 1) % 100).astype(str).str.zfill(2)


@st.cache_data(show_spinner="Memuat data...")
def load_data():
    """Hourly cleaned data plus the daily station means (>= 20 valid hours)."""
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df["date"] = df["datetime"].dt.normalize()
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["season"] = df["month"].map(SEASONS)
    df["period"] = period_label(df["datetime"])
    df["wind_bin"] = pd.cut(df["WSPM"], bins=WIND_BINS, labels=WIND_LABELS, include_lowest=True)

    g = df.groupby(["station", "date"])["PM2.5"].agg(["mean", "count"]).reset_index()
    daily = g[g["count"] >= 20].rename(columns={"mean": "PM2.5"}).drop(columns="count").reset_index(drop=True)
    daily["season"] = daily["date"].dt.month.map(SEASONS)
    daily["period"] = period_label(daily["date"])
    daily["category"] = pd.cut(daily["PM2.5"], bins=AQI_BINS, labels=AQI_LABELS, include_lowest=True)
    return df, daily


# --------------------------------------------------------------------------- charts
def add_titles(fig, title, subtitle):
    """Left-aligned takeaway title, subtitle and source note."""
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.915, subtitle, ha="left", fontsize=9.5, color=MUTED)
    fig.text(0.01, 0.012, SOURCE, ha="left", fontsize=8, color=MUTED)


def finish(fig):
    fig.tight_layout(rect=[0, 0.04, 1, 0.9])
    return fig


def chart_stations(stats):
    """Mean PM2.5 and share of days above the daily limit, per station."""
    order = stats.index[::-1]
    top, low = stats.index[0], stats.index[-1]
    many = len(stats) > 1
    colors = [RED if s == top else GREEN if s == low else GRAY for s in order] if many else [BLUE]
    panels = [("mean", "Rata-rata PM2.5 (µg/m³)", "%.1f"),
              ("pct_days_over", f"Hari dengan rata-rata harian > {LIMIT_DAILY} µg/m³ (%)", "%.1f%%")]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.6, 2.0 + 0.34 * len(order))), sharey=True)
    for ax, (col, label, fmt) in zip(axes, panels):
        bars = ax.barh(list(order), stats.loc[order, col], color=colors, height=0.7)
        ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
        ax.set_xlabel(label)
        ax.set_xlim(0, max(stats[col].max() * 1.15, 1))
        ax.set_ylim(-0.6, len(order) + 0.2)
        ax.grid(axis="y", visible=False)
    axes[0].axvline(LIMIT_ANNUAL, color=INK, ls="--", lw=1)
    axes[0].text(LIMIT_ANNUAL + 1, len(order) - 0.45, f"Baku mutu tahunan ({LIMIT_ANNUAL})", fontsize=8.5, va="bottom")
    title = f"{top} paling tercemar dan {low} paling bersih" if many else f"Stasiun {top}"
    add_titles(fig, title, f"Rata-rata PM2.5 dan persentase hari melampaui baku mutu harian ({LIMIT_DAILY} µg/m³). "
                           f"Garis putus-putus: baku mutu tahunan ({LIMIT_ANNUAL} µg/m³).")
    return finish(fig)


def chart_time(monthly, season_hour):
    """Monthly profile (bars) and hourly profile per season (lines)."""
    vals = monthly.reindex(range(1, 13))
    peak_m, low_m = vals.idxmax(), vals.idxmin()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    colors = [RED if m == peak_m else GREEN if m == low_m else GRAY for m in vals.index]
    bars = ax.bar(MONTHS, vals.fillna(0).values, color=colors, width=0.72)
    for rect, v in zip(bars, vals.values):
        if pd.notna(v):  # months outside the selected range stay empty
            ax.text(rect.get_x() + rect.get_width() / 2, v + vals.max() * 0.015, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Rata-rata PM2.5 (µg/m³)")
    ax.set_xlabel("Bulan")
    ax.set_ylim(0, vals.max() * 1.15)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    styles = {"Dingin": (RED, 2.8), "Panas": (GREEN, 2.2), "Semi": ("#7C8696", 1.6), "Gugur": ("#AEB5C1", 1.6)}
    present = [s for s in SEASON_ORDER if s in season_hour.columns and season_hour[s].notna().any()]
    for s in present:
        color, lw = styles[s]
        ax.plot(season_hour.index, season_hour[s], color=color, lw=lw)
        ax.text(23.5, season_hour[s].dropna().iloc[-1], s, color=color, va="center", fontsize=9, fontweight="bold")
    hi = max(present, key=lambda s: season_hour[s].max())
    h_hi, h_lo = season_hour[hi].idxmax(), season_hour[hi].idxmin()
    ax.scatter([h_hi, h_lo], season_hour.loc[[h_hi, h_lo], hi], color=styles[hi][0], zorder=3, s=28)
    ax.annotate(f"{season_hour.loc[h_hi, hi]:.0f} (pukul {h_hi:02d}:00)", (h_hi, season_hour.loc[h_hi, hi]),
                xytext=(-6, 9), textcoords="offset points", ha="right", fontsize=8.5, color=styles[hi][0])
    ax.annotate(f"{season_hour.loc[h_lo, hi]:.0f} (pukul {h_lo:02d}:00)", (h_lo, season_hour.loc[h_lo, hi]),
                xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8.5, color=styles[hi][0])
    ax.set_xlim(0, 26.5)
    ax.set_ylim(0, season_hour.max().max() * 1.18)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Jam (waktu setempat)")
    ax.set_ylabel("Rata-rata PM2.5 (µg/m³)")

    add_titles(fig, f"PM2.5 memuncak pada {MONTHS[peak_m - 1]}; profil jam per musim (musim {hi.lower()} tertinggi)",
               "Rata-rata stasiun terpilih: profil bulanan (kiri; merah = tertinggi, hijau = terendah) dan profil jam per musim (kanan).")
    return finish(fig)


def chart_trend(by_period, period_station):
    """Yearly means (bars) and first-vs-last period per station (dumbbell)."""
    first_p, last_p, best_p = by_period.index[0], by_period.index[-1], by_period.idxmin()
    chg_total = (by_period[last_p] / by_period[first_p] - 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [0.85, 1.15]})

    ax = axes[0]
    colors = [RED if p == last_p else GREEN if p == best_p else BLUE if p == first_p else GRAY for p in by_period.index]
    bars = ax.bar(list(by_period.index), by_period.values, color=colors, width=0.62)
    labels = [f"{v:.1f}" if p == first_p else f"{v:.1f}\n({v / by_period[first_p] - 1:+.1%})" for p, v in by_period.items()]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.axhline(LIMIT_ANNUAL, color=INK, ls="--", lw=1, label=f"Baku mutu tahunan ({LIMIT_ANNUAL})")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_ylim(0, by_period.max() * 1.25)
    ax.set_xlabel("Periode (Maret–Februari)")
    ax.set_ylabel("Rata-rata PM2.5 (µg/m³)")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ps = period_station[[first_p, last_p]].dropna().sort_values(last_p)
    y = np.arange(len(ps))
    ax.hlines(y, ps[first_p], ps[last_p], color=GRAY, lw=2.4, zorder=1)
    ax.scatter(ps[first_p], y, color=BLUE, s=55, zorder=2, label=first_p)
    ax.scatter(ps[last_p], y, color=RED, s=55, zorder=2, label=last_p)
    for i, (a, b) in enumerate(zip(ps[first_p], ps[last_p])):
        ax.text(max(a, b) + 2, i, f"{b / a - 1:+.0%}", va="center", fontsize=8.5, color=MUTED)
    ax.axvline(LIMIT_ANNUAL, color=INK, ls="--", lw=1)
    ax.text(LIMIT_ANNUAL + 1, len(ps) - 0.35, f"Baku mutu tahunan ({LIMIT_ANNUAL})", fontsize=8.5, va="bottom")
    ax.set_yticks(y, list(ps.index))
    ax.set_xlim(0, ps.max().max() * 1.15)
    ax.set_ylim(-0.7, len(ps) + 0.3)
    ax.set_xlabel("Rata-rata PM2.5 (µg/m³)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", frameon=False)

    add_titles(fig, f"PM2.5 berubah {chg_total:+.1f}% dari {first_p} ke {last_p}; level terbaru {by_period[last_p] / LIMIT_ANNUAL:.1f}× baku mutu",
               "Rata-rata per periode (kiri; hijau = terbaik, merah = terbaru) dan per stasiun (kanan). "
               f"Garis putus-putus: baku mutu tahunan ({LIMIT_ANNUAL} µg/m³).")
    return finish(fig)


def chart_weather(corr, by_wind):
    """Correlation with PM2.5 (bars) and mean PM2.5 by wind-speed class (bars)."""
    c = corr.sort_values()
    top_pos, top_neg = c.idxmax(), c.idxmin()
    top_weather = corr[[w for w in WEATHER if w in corr.index]].abs().idxmax()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [0.9, 1.1]})

    ax = axes[0]
    colors = [RED if k == top_pos and c[k] > 0 else GREEN if k == top_neg and c[k] < 0 else GRAY for k in c.index]
    bars = ax.barh([NAMES[k] for k in c.index], c.values, color=colors, height=0.66)
    ax.bar_label(bars, fmt="%+.2f", padding=3, fontsize=9)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Korelasi Pearson dengan PM2.5 (r)")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    w = by_wind
    bars = ax.bar(range(len(w)), w["mean"], color=[RED if m > LIMIT_DAILY else GREEN for m in w["mean"]], width=0.7)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax.set_xticks(range(len(w)), [f"{lab}\nn={cnt:,}" for lab, cnt in zip(w.index, w["count"])], fontsize=8.5)
    ax.axhline(LIMIT_DAILY, color=INK, ls="--", lw=1)
    ax.text(len(w) - 0.5, LIMIT_DAILY + 2, f"{LIMIT_DAILY} µg/m³ (baku mutu harian)", ha="right", fontsize=8.5)
    ax.set_ylim(0, max(w["mean"].max() * 1.15, LIMIT_DAILY * 1.1))
    ax.set_xlabel("Kecepatan angin (m/s) dan jumlah jam pengukuran (n)")
    ax.set_ylabel("Rata-rata PM2.5 (µg/m³)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=RED, label=f"Rata-rata > {LIMIT_DAILY}"), Patch(color=GREEN, label=f"Rata-rata ≤ {LIMIT_DAILY}")],
              loc="upper right", frameon=False, fontsize=8.5)

    add_titles(fig, f"Korelasi terkuat: {NAMES[top_pos]}; faktor cuaca terkuat: {NAMES[top_weather].lower()}",
               "Korelasi per variabel (kiri; merah = positif terkuat, hijau = negatif terkuat) dan rata-rata PM2.5 per kelas kecepatan angin (kanan).")
    return finish(fig)


def aqi_share(daily_f, by):
    """Share (%) of station-days per air-quality category, overall plus one row per group."""
    overall = daily_f["category"].value_counts(normalize=True).reindex(AQI_LABELS).fillna(0) * 100
    key, order = {"Musim": ("season", SEASON_ORDER), "Stasiun": ("station", None), "Periode": ("period", None)}[by]
    tab = pd.crosstab(daily_f[key], daily_f["category"], normalize="index") * 100
    tab = tab.reindex(columns=AQI_LABELS, fill_value=0)
    if order:
        tab = tab.reindex([o for o in order if o in tab.index])
    elif key == "station":
        tab = tab.loc[tab[AQI_LABELS[4:]].sum(axis=1).sort_values(ascending=False).index]
    else:
        tab = tab.sort_index()
    label = {"Musim": "Semua musim", "Stasiun": "Semua stasiun", "Periode": "Semua periode"}[by]
    return pd.concat([overall.to_frame(label).T, tab])


def chart_aqi(share, n_days, by):
    """100% stacked bars of air-quality categories."""
    rows = [str(r) for r in share.index[::-1]]  # first row ends up on top
    values = share.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, max(3.8, 2.2 + 0.4 * len(rows))))
    left = np.zeros(len(rows))
    for cat, color, txt in zip(AQI_LABELS, AQI_COLORS, AQI_TEXT):
        vals = values[cat].values
        ax.barh(rows, vals, left=left, color=color, height=0.62, label=cat)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 4:  # label only segments wide enough to read
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color=txt)
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("Persentase hari-stasiun (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=8.5,
              handlelength=1.2, columnspacing=1.1)
    heavy = share[AQI_LABELS[4:]].sum(axis=1).iloc[0]
    add_titles(fig, f"{heavy:.0f}% hari tercemar berat atau lebih buruk; distribusi kategori per {by.lower()}",
               f"Kategori kualitas udara harian (China AQI untuk PM2.5); n = {n_days:,} hari-stasiun. "
               "Berat+ = tercemar berat dan sangat berat (> 150 µg/m³).")
    return finish(fig)


def station_geo(stats):
    """Coordinates, distance zone and PM2.5 stats for the selected stations."""
    geo = pd.DataFrame({s: COORDS[s] for s in stats.index}, index=["lat", "lon"]).T
    geo["dist_km"] = haversine_km(geo["lat"], geo["lon"], *CENTER)
    geo["zone"] = pd.cut(geo["dist_km"], bins=[0, 10, 30, np.inf], labels=ZONES)
    return geo.join(stats[["mean", "pct_days_over"]])


def build_map(geo):
    """Interactive folium map: marker size and color follow mean PM2.5."""
    cmap, norm = plt.get_cmap("OrRd"), mcolors.Normalize(vmin=60, vmax=90)
    m = folium.Map(location=[40.02, 116.40], zoom_start=9, tiles="OpenStreetMap", control_scale=True)
    for name, row in geo.iterrows():
        popup = (f"<b>{name}</b><br>Rata-rata PM2.5: {row['mean']:.1f} µg/m³<br>"
                 f"Hari > {LIMIT_DAILY} µg/m³: {row['pct_days_over']:.1f}%<br>"
                 f"Jarak dari pusat kota: {row['dist_km']:.0f} km<br>Zona: {row['zone']}")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=row["mean"] / 4.5, weight=1.2, color=INK, fill=True,
            fill_color=mcolors.to_hex(cmap(norm(row["mean"]))), fill_opacity=0.85,
            tooltip=f"{name}: {row['mean']:.1f} µg/m³", popup=folium.Popup(popup, max_width=260)).add_to(m)
    folium.CircleMarker(list(CENTER), radius=5, color=INK, fill=True, fill_color=INK,
                        tooltip="Pusat kota (Tiananmen)").add_to(m)
    return m


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def embed_html(html, height):
    """Render a raw HTML page: st.iframe on recent Streamlit, components.html on older versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


# --------------------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Kualitas Udara Beijing", page_icon="🌫️", layout="wide")
    df, daily = load_data()
    all_stations = sorted(df["station"].unique())
    d_min, d_max = df["datetime"].min().date(), df["datetime"].max().date()

    with st.sidebar:
        st.header("Filter")
        picked = st.date_input("Rentang tanggal", value=(d_min, d_max), min_value=d_min, max_value=d_max)
        chosen = st.multiselect("Stasiun", all_stations, default=all_stations)
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start, end = picked
        else:
            start, end = d_min, d_max
            st.info("Pilih tanggal akhir untuk menerapkan rentang; sementara seluruh periode ditampilkan.")
        st.divider()
        st.caption("**Baku mutu PM2.5 (China GB 3095-2012, Kelas II):** 75 µg/m³ rata-rata harian dan 35 µg/m³ rata-rata tahunan.")
        st.caption("**Catatan:** koordinat stasiun pada peta adalah perkiraan lokasi dan tidak ada di dataset asli.")

    if not chosen:
        st.warning("Pilih minimal satu stasiun pada panel filter.")
        st.stop()

    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    d = df[df["station"].isin(chosen) & (df["date"] >= t0) & (df["date"] <= t1)]
    dd = daily[daily["station"].isin(chosen) & (daily["date"] >= t0) & (daily["date"] <= t1)]
    if d.empty or dd.empty or d["PM2.5"].notna().sum() == 0:
        st.warning("Tidak ada data PM2.5 yang valid untuk filter ini. Perluas rentang tanggal.")
        st.stop()

    # Station-level summary used by several tabs
    stats = d.groupby("station")["PM2.5"].agg(mean="mean", median="median")
    stats["pct_days_over"] = dd["PM2.5"].gt(LIMIT_DAILY).groupby(dd["station"]).mean() * 100
    stats = stats.dropna().sort_values("mean", ascending=False)
    stats.index = stats.index.astype(str)

    st.title("🌫️ Dashboard Kualitas Udara Beijing (PM2.5)")
    st.caption(f"{len(chosen)} stasiun · {start:%d %b %Y} – {end:%d %b %Y} · "
               f"{len(d):,} jam pengukuran · {len(dd):,} hari-stasiun valid")

    mean_pm = d["PM2.5"].mean()
    heavy_pct = dd["PM2.5"].gt(150).mean() * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rata-rata PM2.5", f"{mean_pm:.1f} µg/m³", help=f"{mean_pm / LIMIT_ANNUAL:.1f}× baku mutu tahunan ({LIMIT_ANNUAL} µg/m³).")
    k2.metric(f"Hari melampaui {LIMIT_DAILY} µg/m³", f"{dd['PM2.5'].gt(LIMIT_DAILY).mean() * 100:.1f}%",
              help="Persentase hari-stasiun dengan rata-rata harian di atas baku mutu harian.")
    k3.metric("Stasiun paling tercemar", stats.index[0], help=f"Rata-rata {stats['mean'].iloc[0]:.1f} µg/m³.")
    k4.metric("Hari tercemar berat+ (> 150)", f"{heavy_pct:.1f}%", help="Kategori tercemar berat dan sangat berat (China AQI).")

    tabs = st.tabs(["1 · Stasiun", "2 · Waktu", "3 · Tren", "4 · Cuaca", "Analisis lanjutan"])

    # ---- Q1: stations
    with tabs[0]:
        st.subheader("Stasiun mana yang paling dan paling tidak tercemar?")
        show(chart_stations(stats))
        if len(stats) > 1:
            gap = (stats["mean"].iloc[0] / stats["mean"].iloc[-1] - 1) * 100
            st.info(f"**{stats.index[0]}** tertinggi ({stats['mean'].iloc[0]:.1f} µg/m³) dan **{stats.index[-1]}** terendah "
                    f"({stats['mean'].iloc[-1]:.1f} µg/m³); selisih {gap:.0f}%. Persentase hari melampaui {LIMIT_DAILY} µg/m³ "
                    f"berkisar {stats['pct_days_over'].min():.1f}%–{stats['pct_days_over'].max():.1f}%. Semua stasiun terpilih "
                    f"{stats['mean'].min() / LIMIT_ANNUAL:.1f}–{stats['mean'].max() / LIMIT_ANNUAL:.1f}× di atas baku mutu tahunan.")
        else:
            st.info(f"Stasiun **{stats.index[0]}**: rata-rata {stats['mean'].iloc[0]:.1f} µg/m³ dan "
                    f"{stats['pct_days_over'].iloc[0]:.1f}% hari melampaui {LIMIT_DAILY} µg/m³.")

    # ---- Q2: time
    with tabs[1]:
        st.subheader("Kapan polusi paling tinggi (bulan, musim, jam)?")
        monthly = d.groupby("month")["PM2.5"].mean()
        hourly = d.groupby("hour")["PM2.5"].mean()
        season_hour = d.pivot_table(index="hour", columns="season", values="PM2.5", aggfunc="mean")
        show(chart_time(monthly, season_hour))
        peak_h, low_h = int(hourly.idxmax()), int(hourly.idxmin())
        text = f"Per jam, puncak pada pukul **{peak_h:02d}:00** ({hourly[peak_h]:.1f} µg/m³) dan terendah pukul {low_h:02d}:00 ({hourly[low_h]:.1f})."
        if monthly.notna().sum() > 1:
            pm_, lm_ = int(monthly.idxmax()), int(monthly.idxmin())
            text = (f"Puncak bulanan pada **{MONTHS[pm_ - 1]}** ({monthly[pm_]:.1f} µg/m³), {(monthly[pm_] / monthly[lm_] - 1) * 100:.0f}% lebih tinggi "
                    f"dari titik terendah di **{MONTHS[lm_ - 1]}** ({monthly[lm_]:.1f}). ") + text
        st.info(text)

    # ---- Q3: trend (date filter ignored so every period has 12 full months)
    with tabs[2]:
        st.subheader("Apakah polusi membaik dari tahun ke tahun?")
        st.caption("Tab ini memakai seluruh periode data agar tiap periode Maret–Februari lengkap 12 bulan; hanya filter stasiun yang berlaku.")
        ds = df[df["station"].isin(chosen)]
        by_period = ds.groupby("period")["PM2.5"].mean()
        period_station = ds.pivot_table(index="station", columns="period", values="PM2.5", aggfunc="mean")
        by_period.index, period_station.index, period_station.columns = (
            by_period.index.astype(str), period_station.index.astype(str), period_station.columns.astype(str))
        show(chart_trend(by_period, period_station))
        first_p, last_p, best_p = by_period.index[0], by_period.index[-1], by_period.idxmin()
        st.info(f"Dari {first_p} ke {last_p} rata-rata PM2.5 berubah **{(by_period[last_p] / by_period[first_p] - 1) * 100:+.1f}%** "
                f"({by_period[first_p]:.1f} → {by_period[last_p]:.1f} µg/m³). Titik terbaik pada {best_p} ({by_period[best_p]:.1f}); "
                f"level terbaru **{by_period[last_p] / LIMIT_ANNUAL:.1f}×** baku mutu tahunan, perlu penurunan ±{(1 - LIMIT_ANNUAL / by_period[last_p]) * 100:.0f}% lagi. "
                "Empat periode belum cukup untuk menyimpulkan efektivitas kebijakan secara kausal.")

    # ---- Q4: weather
    with tabs[3]:
        st.subheader("Faktor cuaca dan polutan apa yang terkait dengan PM2.5?")
        corr = d[["PM2.5"] + CORR_COLS].corr()["PM2.5"].drop("PM2.5").dropna()
        by_wind = d.groupby("wind_bin", observed=True)["PM2.5"].agg(["mean", "count"]).dropna()
        if len(corr) < 2 or by_wind.empty:
            st.info("Data terlalu sedikit untuk menghitung korelasi. Perluas rentang tanggal.")
        else:
            show(chart_weather(corr, by_wind))
            top_pos = corr.idxmax()
            top_weather = corr[[w for w in WEATHER if w in corr.index]].abs().idxmax()
            below = by_wind.index[by_wind["mean"] < LIMIT_DAILY]
            text = (f"**{NAMES[top_pos]}** paling berkorelasi dengan PM2.5 (r = {corr[top_pos]:+.2f}); faktor cuaca terkuat adalah "
                    f"**{NAMES[top_weather].lower()}** (r = {corr[top_weather]:+.2f}). ")
            if len(below):
                text += (f"Rata-rata PM2.5 pertama kali di bawah {LIMIT_DAILY} µg/m³ pada kelas angin **{below[0]} m/s**; "
                         f"angin tenang (≤ {WIND_LABELS[0].replace('≤', '')} m/s) rata-rata {by_wind['mean'].iloc[0]:.1f} µg/m³.")
            else:
                text += f"Pada filter ini rata-rata PM2.5 tidak pernah turun di bawah {LIMIT_DAILY} µg/m³ pada kelas angin mana pun."
            st.info(text)

    # ---- Advanced: binning + geospatial
    with tabs[4]:
        st.subheader("Binning kategori kualitas udara dan peta stasiun")
        by = st.radio("Kelompokkan kategori berdasarkan", ["Musim", "Stasiun", "Periode"], horizontal=True)
        share = aqi_share(dd, by)
        show(chart_aqi(share, len(dd), by))
        clean, heavy = share[AQI_LABELS[:2]].sum(axis=1).iloc[0], share[AQI_LABELS[4:]].sum(axis=1).iloc[0]
        st.info(f"Hanya **{clean:.0f}%** hari-stasiun berkategori baik/sangat baik (≤ 75 µg/m³) dan **{heavy:.0f}%** tercemar berat atau lebih buruk (> 150 µg/m³). "
                "Batas kelas: ≤35, 35–75, 75–115, 115–150, 150–250, >250 µg/m³ (China AQI, HJ 633-2012).")

        st.markdown("#### Peta stasiun (geospatial)")
        geo = station_geo(stats)
        left, right = st.columns([3, 2])
        with left:
            embed_html(build_map(geo).get_root().render(), height=520)
        with right:
            zones = (geo.groupby("zone", observed=True)
                        .agg(stasiun=("mean", "size"), pm25=("mean", "mean"), hari_lewat=("pct_days_over", "mean")).round(1))
            zones.columns = ["Jumlah stasiun", "Rata-rata PM2.5 (µg/m³)", f"Hari > {LIMIT_DAILY} (%)"]
            zones.index = zones.index.astype(str)
            st.markdown("**Rata-rata per zona jarak dari pusat kota**")
            st.dataframe(zones)
            st.caption("Lingkaran lebih besar dan lebih gelap = rata-rata PM2.5 lebih tinggi. Klik penanda untuk detail.")
            st.download_button("Unduh data harian terfilter (CSV)", dd.drop(columns=["category"]).to_csv(index=False),
                               file_name="pm25_harian_terfilter.csv", mime="text/csv")

    st.divider()
    st.caption(SOURCE + " · Pembersihan data dan analisis lengkap ada di notebook.ipynb.")


if __name__ == "__main__":
    main()
