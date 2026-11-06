"""Global Climate Change & Air Quality Analysis Dashboard.

Run with: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="AirLens | India Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = Path("dataset/city_day.csv")
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]
CITY_COORDINATES = {
    "Ahmedabad": (23.0225, 72.5714), "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707), "Delhi": (28.6139, 77.2090),
    "Hyderabad": (17.3850, 78.4867), "Kolkata": (22.5726, 88.3639),
    "Mumbai": (19.0760, 72.8777), "Pune": (18.5204, 73.8567),
}
AQI_COLORS = {
    "Good": "#22c55e", "Satisfactory": "#84cc16", "Moderate": "#facc15",
    "Poor": "#fb923c", "Very Poor": "#ef4444", "Severe": "#a855f7", "Unavailable": "#94a3b8",
}


def aqi_category(aqi: float) -> str:
    if pd.isna(aqi): return "Unavailable"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


@st.cache_data(show_spinner=False)
def make_demo_data() -> pd.DataFrame:
    """Create a compact fallback that keeps the dashboard explorable."""
    rng = np.random.default_rng(42)
    days = pd.date_range("2018-01-01", "2020-12-31", freq="D")
    baseline = {"Ahmedabad": 145, "Bengaluru": 82, "Chennai": 98, "Delhi": 205,
                "Hyderabad": 118, "Kolkata": 165, "Mumbai": 105, "Pune": 92}
    frames = []
    for city, base in baseline.items():
        season = 35 * np.cos(2 * np.pi * (days.dayofyear - 20) / 365.25)
        trend = np.linspace(8, -5, len(days))
        aqi = np.clip(base + season + trend + rng.normal(0, 24, len(days)), 25, 480)
        frame = pd.DataFrame({
            "City": city, "Date": days, "AQI": np.round(aqi),
            "PM2.5": np.clip(aqi * rng.normal(.42, .05, len(days)), 5, None),
            "PM10": np.clip(aqi * rng.normal(.65, .07, len(days)), 8, None),
            "NO2": np.clip(aqi * rng.normal(.16, .03, len(days)), 2, None),
            "SO2": np.clip(aqi * rng.normal(.06, .02, len(days)), 1, None),
            "CO": np.clip(aqi * rng.normal(.010, .003, len(days)), .1, None),
            "O3": np.clip(35 + 20 * np.sin(2 * np.pi * days.dayofyear / 365.25) + rng.normal(0, 8, len(days)), 2, None),
        })
        frame["AQI_Bucket"] = frame["AQI"].apply(aqi_category)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Loading air-quality records…")
def load_data(path_text: str) -> tuple[pd.DataFrame, bool]:
    path = Path(path_text)
    if not path.exists():
        return make_demo_data(), True
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    if "Date" not in data or "City" not in data:
        raise ValueError("The dataset must contain City and Date columns.")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in set(POLLUTANTS + ["AQI"]).intersection(data.columns):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["City", "Date"]).sort_values("Date")
    if "AQI_Bucket" not in data:
        data["AQI_Bucket"] = data["AQI"].apply(aqi_category)
    data["AQI_Bucket"] = data["AQI_Bucket"].fillna("Unavailable")
    return data, False


def fmt_number(value: float, suffix: str = "") -> str:
    return "—" if pd.isna(value) else f"{value:,.0f}{suffix}"


def quality_note(value: float) -> str:
    return aqi_category(value) if not pd.isna(value) else "No reading"


def available(columns: Iterable[str], data: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in data.columns and data[column].notna().any()]


def inject_css() -> None:
    st.markdown("""<style>
    .stApp { background: #f4f7fb; color: #172033; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #071a33 0%, #0b3157 100%); }
    [data-testid="stSidebar"] * { color: #eef6ff; }
    .hero { background: linear-gradient(115deg, #063865, #0b6980 58%, #168b76); padding: 2.1rem 2.4rem; border-radius: 22px; color: white; margin-bottom: 1rem; box-shadow: 0 14px 32px rgba(5, 46, 80, .18); }
    .hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: -.05em; }
    .hero p { margin: .55rem 0 0; color: #d9f4f1; font-size: 1.05rem; }
    .eyebrow { color: #8ce7ce; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .74rem; }
    [data-testid="stMetric"] { background: white; padding: 1rem 1.1rem; border-radius: 14px; border: 1px solid #e3eaf3; box-shadow: 0 4px 14px rgba(33, 57, 91, .05); }
    div[data-testid="stMetricLabel"] { color: #637083; }
    .insight { background: #e8f7f2; border-left: 4px solid #12a67d; padding: .85rem 1rem; border-radius: 0 10px 10px 0; color: #164a42; margin-top: .5rem; }
    .source-note { color: #64748b; font-size: .82rem; }
    </style>""", unsafe_allow_html=True)


inject_css()
try:
    data, is_demo = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not read `{DATA_PATH}`: {error}")
    st.stop()

with st.sidebar:
    st.markdown("## 🌍 AirLens")
    st.caption("Environmental intelligence for healthier cities")
    st.divider()
    cities = sorted(data["City"].dropna().unique())
    selected_cities = st.multiselect("Cities", cities, default=cities[: min(4, len(cities))])
    min_date, max_date = data["Date"].min().date(), data["Date"].max().date()
    date_range = st.date_input("Analysis period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    aggregation = st.selectbox("Trend resolution", ["Monthly", "Weekly", "Daily"], index=0)
    st.divider()
    st.caption("AQI follows India’s National Air Quality Index categories.")

if not selected_cities:
    st.warning("Select at least one city to begin exploring.")
    st.stop()
start_date, end_date = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[-1])) if isinstance(date_range, tuple) else (pd.Timestamp(date_range), pd.Timestamp(date_range))
filtered = data[(data["City"].isin(selected_cities)) & (data["Date"].between(start_date, end_date))].copy()
if filtered.empty:
    st.warning("No observations match those filters.")
    st.stop()

st.markdown("""<div class="hero"><div class="eyebrow">India • Climate & air-quality intelligence</div><h1>See the air your city is breathing.</h1><p>Explore pollution patterns, compare locations, and turn daily environmental data into clear action.</p></div>""", unsafe_allow_html=True)
if is_demo:
    st.info("Demo mode — `dataset/city_day.csv` was not found. Add the official dataset to switch to real readings.")

latest_date = filtered["Date"].max()
latest = filtered[filtered["Date"] == latest_date]
avg_aqi = filtered["AQI"].mean()
best_city = filtered.groupby("City")["AQI"].mean().mean() if filtered["AQI"].notna().any() else np.nan
records = len(filtered)
metric_cols = st.columns(4)
metric_cols[0].metric("Average AQI", fmt_number(avg_aqi), quality_note(avg_aqi))
metric_cols[1].metric("Latest reading", latest_date.strftime("%d %b %Y"), f"{len(latest):,} city readings")
metric_cols[2].metric("Cities compared", len(selected_cities), "Selected locations")
metric_cols[3].metric("Observations analysed", f"{records:,}", f"Since {filtered['Date'].min():%b %Y}")

tab_overview, tab_trends, tab_pollutants, tab_compare = st.tabs(["Overview", "Trends", "Pollutants", "City comparison"])

with tab_overview:
    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Air quality across selected cities")
        city_snapshot = filtered.groupby("City", as_index=False).agg(AQI=("AQI", "mean"))
        city_snapshot["Category"] = city_snapshot["AQI"].apply(aqi_category)
        city_snapshot["lat"] = city_snapshot["City"].map(lambda x: CITY_COORDINATES.get(x, (20.5937, 78.9629))[0])
        city_snapshot["lon"] = city_snapshot["City"].map(lambda x: CITY_COORDINATES.get(x, (20.5937, 78.9629))[1])
        fig_map = px.scatter_geo(city_snapshot, lat="lat", lon="lon", size="AQI", color="Category", hover_name="City", hover_data={"AQI": ":.0f", "lat": False, "lon": False}, color_discrete_map=AQI_COLORS, scope="asia", projection="natural earth", size_max=35)
        fig_map.update_geos(center={"lat": 21, "lon": 79}, projection_scale=4.3, showland=True, landcolor="#edf2f7", showcountries=True, countrycolor="#cbd5e1")
        fig_map.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="AQI category", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_map, use_container_width=True)
    with right:
        st.subheader("Health-risk distribution")
        buckets = filtered["AQI_Bucket"].fillna("Unavailable").value_counts().rename_axis("Category").reset_index(name="Days")
        fig_donut = px.pie(buckets, names="Category", values="Days", hole=.62, color="Category", color_discrete_map=AQI_COLORS)
        fig_donut.update_layout(height=325, margin=dict(l=5, r=5, t=10, b=5), legend=dict(orientation="h", y=-.15), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)
        worst = city_snapshot.loc[city_snapshot["AQI"].idxmax()]
        st.markdown(f"<div class='insight'><b>Key signal:</b> {worst['City']} has the highest average AQI in this view ({worst['AQI']:.0f}, {worst['Category']}).</div>", unsafe_allow_html=True)

with tab_trends:
    st.subheader("AQI trend over time")
    frequency = {"Monthly": "MS", "Weekly": "W-MON", "Daily": "D"}[aggregation]
    trend = filtered.set_index("Date").groupby("City")["AQI"].resample(frequency).mean().reset_index()
    fig_trend = px.line(trend, x="Date", y="AQI", color="City", markers=aggregation != "Daily", color_discrete_sequence=px.colors.qualitative.Bold)
    fig_trend.add_hrect(y0=0, y1=50, fillcolor="#22c55e", opacity=.06, line_width=0, annotation_text="Good", annotation_position="top left")
    fig_trend.add_hrect(y0=200, y1=300, fillcolor="#fb923c", opacity=.08, line_width=0, annotation_text="Poor", annotation_position="top left")
    fig_trend.update_layout(height=470, hovermode="x unified", yaxis_title="AQI", xaxis_title="", legend_title_text="City", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff")
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("Shaded bands provide quick context: AQI 0–50 is Good; 200–300 is Poor.")

with tab_pollutants:
    pollutant_options = available(POLLUTANTS, filtered)
    if not pollutant_options:
        st.info("This data source does not include pollutant concentration columns.")
    else:
        pollutant = st.selectbox("Pollutant", pollutant_options, index=0)
        left, right = st.columns([1.4, 1])
        with left:
            pollutant_trend = filtered.set_index("Date").groupby("City")[pollutant].resample("MS").mean().reset_index()
            fig_pollutant = px.area(pollutant_trend, x="Date", y=pollutant, color="City", groupnorm=None, color_discrete_sequence=px.colors.qualitative.Safe)
            fig_pollutant.update_layout(height=420, hovermode="x unified", yaxis_title=f"{pollutant} concentration", xaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff")
            st.plotly_chart(fig_pollutant, use_container_width=True)
        with right:
            st.markdown("#### What is driving AQI?")
            correlation_columns = available(["AQI"] + pollutant_options, filtered)
            correlations = filtered[correlation_columns].corr(numeric_only=True)["AQI"].drop("AQI").sort_values().reset_index()
            correlations.columns = ["Pollutant", "Correlation"]
            fig_corr = px.bar(correlations, x="Correlation", y="Pollutant", orientation="h", color="Correlation", color_continuous_scale="Teal", range_color=[-1, 1])
            fig_corr.update_layout(height=370, coloraxis_showscale=False, margin=dict(l=0, r=10, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff")
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Correlation indicates association in the selected data, not causation.")

with tab_compare:
    st.subheader("City performance scorecard")
    summary = filtered.groupby("City").agg(
        Average_AQI=("AQI", "mean"), Peak_AQI=("AQI", "max"),
        Best_AQI=("AQI", "min"), Readings=("AQI", "count")
    ).reset_index().sort_values("Average_AQI")
    summary["Typical condition"] = summary["Average_AQI"].apply(aqi_category)
    display_summary = summary.rename(columns={"Average_AQI": "Average AQI", "Peak_AQI": "Peak AQI", "Best_AQI": "Best AQI"})
    st.dataframe(display_summary, use_container_width=True, hide_index=True, column_config={
        "Average AQI": st.column_config.NumberColumn(format="%.0f"), "Peak AQI": st.column_config.NumberColumn(format="%.0f"), "Best AQI": st.column_config.NumberColumn(format="%.0f"),
    })
    fig_rank = px.bar(summary, x="Average_AQI", y="City", orientation="h", color="Typical condition", color_discrete_map=AQI_COLORS, text="Average_AQI")
    fig_rank.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_rank.update_layout(height=max(330, len(summary) * 55), xaxis_title="Average AQI (lower is better)", yaxis_title="", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff")
    st.plotly_chart(fig_rank, use_container_width=True)

st.divider()
st.markdown("<p class='source-note'>Data source: India Air Quality Data (2015–2020). Dashboard designed for exploratory analysis; consult local authorities for health decisions.</p>", unsafe_allow_html=True)
