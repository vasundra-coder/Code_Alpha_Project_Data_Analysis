"""Generate submission-ready EDA artefacts in output/.

Run: python generate_outputs.py
Uses dataset/city_day.csv when available; otherwise uses a reproducible demo set.
"""
import os
from pathlib import Path

# Render image files without requiring a desktop GUI (works locally and in CI).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("output") / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = Path("dataset/city_day.csv")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)
COLORS = ["#0b6980", "#22a783", "#f0ad4e", "#e76f51", "#7b61a8", "#457b9d"]


def category(aqi):
    if pd.isna(aqi): return "Unavailable"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


def demo_data():
    rng = np.random.default_rng(42)
    days = pd.date_range("2018-01-01", "2020-12-31", freq="D")
    baseline = {"Ahmedabad": 145, "Bengaluru": 82, "Chennai": 98, "Delhi": 205,
                "Hyderabad": 118, "Kolkata": 165, "Mumbai": 105, "Pune": 92}
    records = []
    for city, base in baseline.items():
        seasonal = 35 * np.cos(2 * np.pi * (days.dayofyear - 20) / 365.25)
        aqi = np.clip(base + seasonal + np.linspace(8, -5, len(days)) + rng.normal(0, 24, len(days)), 25, 480)
        records.append(pd.DataFrame({
            "City": city, "Date": days, "AQI": np.round(aqi),
            "PM2.5": np.clip(aqi * rng.normal(.42, .05, len(days)), 5, None),
            "PM10": np.clip(aqi * rng.normal(.65, .07, len(days)), 8, None),
            "NO2": np.clip(aqi * rng.normal(.16, .03, len(days)), 2, None),
            "SO2": np.clip(aqi * rng.normal(.06, .02, len(days)), 1, None),
            "CO": np.clip(aqi * rng.normal(.010, .003, len(days)), .1, None),
            "O3": np.clip(35 + 20 * np.sin(2 * np.pi * days.dayofyear / 365.25) + rng.normal(0, 8, len(days)), 2, None),
        }))
    return pd.concat(records, ignore_index=True)


def load_data():
    if not DATA_PATH.exists():
        return demo_data(), True
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for col in ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["City", "Date"]).sort_values("Date"), False


def save_chart(filename, title):
    plt.title(title, loc="left", fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    sns.set_theme(style="whitegrid", font_scale=1.0)
    data, demo = load_data()
    data["AQI Category"] = data["AQI"].apply(category)
    data["Month"] = data["Date"].dt.to_period("M").dt.to_timestamp()
    city_palette = dict(zip(sorted(data["City"].unique()), sns.color_palette("tab10", n_colors=data["City"].nunique())))

    # Cleaned data and numerical summary.
    data.to_csv(OUTPUT / "cleaned_air_quality_data.csv", index=False)
    numeric = data.select_dtypes("number")
    numeric.describe().T.round(2).to_csv(OUTPUT / "descriptive_statistics.csv")
    city_summary = data.groupby("City").agg(
        Average_AQI=("AQI", "mean"), Peak_AQI=("AQI", "max"),
        Minimum_AQI=("AQI", "min"), Observations=("AQI", "count")
    ).sort_values("Average_AQI").round(2)
    city_summary.to_csv(OUTPUT / "city_aqi_summary.csv")
    numeric.corr().round(3).to_csv(OUTPUT / "correlation_matrix.csv")

    # 01: monthly AQI lines.
    monthly = data.groupby(["Month", "City"], as_index=False)["AQI"].mean()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, x="Month", y="AQI", hue="City", palette=city_palette, linewidth=2)
    plt.ylabel("Average AQI")
    plt.xlabel("")
    save_chart("01_monthly_aqi_trends.png", "Monthly AQI trends by city")

    # 02: city ranking.
    plt.figure(figsize=(10, 6))
    rank = city_summary.reset_index()
    sns.barplot(data=rank, y="City", x="Average_AQI", hue="City", palette=city_palette, legend=False)
    plt.xlabel("Average AQI (lower is better)")
    plt.ylabel("")
    save_chart("02_city_aqi_comparison.png", "Average AQI by city")

    # 03: air-quality distribution.
    order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    distribution = data["AQI Category"].value_counts().reindex(order, fill_value=0).reset_index()
    distribution.columns = ["AQI Category", "Days"]
    distribution.to_csv(OUTPUT / "aqi_category_distribution.csv", index=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=distribution, x="AQI Category", y="Days", hue="AQI Category", palette=["#22c55e", "#84cc16", "#facc15", "#fb923c", "#ef4444", "#a855f7"], legend=False)
    plt.xticks(rotation=20, ha="right")
    save_chart("03_aqi_category_distribution.png", "How often air quality reaches each health category")

    # 04: particulate matter comparison.
    pollutants = [c for c in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"] if c in data and data[c].notna().any()]
    if pollutants:
        pollutant_avg = data.groupby("City")[pollutants].mean().reset_index().melt("City", var_name="Pollutant", value_name="Average concentration")
        pollutant_avg.to_csv(OUTPUT / "city_pollutant_averages.csv", index=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=pollutant_avg, x="City", y="Average concentration", hue="Pollutant", palette="viridis")
        plt.xticks(rotation=25, ha="right")
        save_chart("04_pollutant_profile.png", "Average pollutant profile by city")

    # 05: correlation heatmap.
    corr_cols = [c for c in ["AQI", *pollutants] if c in data]
    if len(corr_cols) > 1:
        plt.figure(figsize=(9, 7))
        sns.heatmap(data[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, square=True, cbar_kws={"shrink": .8})
        save_chart("05_pollutant_correlation_heatmap.png", "How pollutants move with AQI")

    # 06: Chennai-focused seasonal view when city exists.
    focus = "Chennai" if "Chennai" in set(data["City"]) else data["City"].iloc[0]
    focus_monthly = data[data["City"] == focus].groupby("Month", as_index=False)["AQI"].mean()
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=focus_monthly, x="Month", y="AQI", color="#0b6980", linewidth=2.5)
    plt.fill_between(focus_monthly["Month"], focus_monthly["AQI"], color="#16a39a", alpha=.18)
    plt.ylabel("Average AQI")
    plt.xlabel("")
    save_chart("06_chennai_aqi_trend.png", f"{focus}: AQI pattern over time")

    worst = city_summary["Average_AQI"].idxmax()
    best = city_summary["Average_AQI"].idxmin()
    report = f"""GLOBAL CLIMATE CHANGE & AIR QUALITY ANALYSIS\n{'=' * 48}\n\nDataset mode: {'DEMO DATA (replace dataset/city_day.csv with the official file)' if demo else 'OFFICIAL city_day.csv'}\nPeriod analysed: {data['Date'].min():%d %b %Y} to {data['Date'].max():%d %b %Y}\nCities: {data['City'].nunique()}\nValid AQI observations: {data['AQI'].notna().sum():,}\n\nKEY FINDINGS\n1. {worst} has the highest average AQI ({city_summary.loc[worst, 'Average_AQI']:.1f}) in the analysed period.\n2. {best} has the lowest average AQI ({city_summary.loc[best, 'Average_AQI']:.1f}), though daily variation remains important.\n3. The city comparison, health-category distribution, and monthly trends make temporal and geographic differences immediately visible.\n4. Pollutant correlations are descriptive associations; they should not be interpreted as causal evidence.\n\nOUTPUT FILES\n- 01–06 PNG files: presentation-ready visualizations\n- cleaned_air_quality_data.csv: prepared analysis data\n- descriptive_statistics.csv, city_aqi_summary.csv, correlation_matrix.csv: supporting tables\n\nGenerated by generate_outputs.py.\n"""
    (OUTPUT / "analysis_report.txt").write_text(report, encoding="utf-8")
    print(f"Created EDA artefacts in {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
