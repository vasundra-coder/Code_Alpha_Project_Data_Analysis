"""Exploratory analysis for the Sleep Health and Lifestyle dataset.

Run from the project root with:
    python dataset/output/mental_health_analysis.py
"""

from pathlib import Path
import os
import warnings
from math import erfc, sqrt

# Enables reliable chart export on a terminal, server, or CI.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="deep")

DATA_PATH = PROJECT_ROOT / "dataset" / "Sleep_health_and_lifestyle_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

REQUIRED_COLUMNS = {
    "Person ID", "Gender", "Age", "Occupation", "Sleep Duration",
    "Quality of Sleep", "Physical Activity Level", "Stress Level",
    "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps", "Sleep Disorder",
}


def save_plot(filename: str) -> None:
    """Save the current chart consistently and free its resources."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def correlation_test(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Return Pearson r and a two-sided normal-approximation p-value."""
    r = float(x.corr(y))
    if len(x) < 4 or abs(r) >= 1:
        return r, 0.0 if abs(r) == 1 else float("nan")
    z_score = abs(r) * sqrt((len(x) - 2) / (1 - r**2))
    return r, erfc(z_score / sqrt(2))


def welch_test(group_one: pd.Series, group_two: pd.Series) -> tuple[float, float]:
    """Return Welch's t statistic and a two-sided normal-approximation p-value."""
    difference = group_one.mean() - group_two.mean()
    standard_error = sqrt(group_one.var(ddof=1) / len(group_one) + group_two.var(ddof=1) / len(group_two))
    t_statistic = float(difference / standard_error)
    return t_statistic, erfc(abs(t_statistic) / sqrt(2))


def main() -> None:
    print("=" * 60)
    print("MENTAL HEALTH AND LIFESTYLE DATA ANALYSIS")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, na_values=["None", "", "N/A", "NA", "null", "NULL"])
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    print(f"\nDataset loaded successfully: {len(df)} records, {len(df.columns)} columns")
    print("\nMissing values before cleaning:\n", df.isna().sum())

    df = df.drop_duplicates().copy()
    df.columns = df.columns.str.strip()

    # Missing sleep-disorder values mean no recorded disorder, not an imputed diagnosis.
    for column in df.columns:
        if column == "Sleep Disorder":
            continue
        if df[column].dtype == "object" and df[column].isna().any():
            df[column] = df[column].fillna(df[column].mode().iat[0])
        elif df[column].dtype != "object" and df[column].isna().any():
            df[column] = df[column].fillna(df[column].median())

    df = df.drop(columns="Person ID")
    df["Age Group"] = pd.cut(
        df["Age"], bins=[0, 25, 35, 45, 55, np.inf],
        labels=["25 or younger", "26–35", "36–45", "46–55", "56+"],
        include_lowest=True,
    )
    df.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
    print(f"\nCleaning completed. Saved cleaned_dataset.csv to {OUTPUT_DIR}")

    numeric_columns = ["Sleep Duration", "Quality of Sleep", "Stress Level", "Physical Activity Level", "Daily Steps"]
    summary = df[numeric_columns].agg(["mean", "median", "min", "max", "std"]).T.round(2)
    summary.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")
    print("\nDescriptive statistics:\n", summary)

    age_analysis = df.groupby("Age Group", observed=False)["Sleep Duration"].mean().round(2)
    age_analysis.to_csv(OUTPUT_DIR / "age_group_analysis.csv", header=["Average Sleep Duration"])
    occupation_analysis = df.groupby("Occupation").agg(
        **{
            "Average Sleep Duration": ("Sleep Duration", "mean"),
            "Average Sleep Quality": ("Quality of Sleep", "mean"),
            "Average Stress Level": ("Stress Level", "mean"),
            "Average Daily Steps": ("Daily Steps", "mean"),
        }
    ).round(2).sort_values("Average Sleep Duration", ascending=False)
    occupation_analysis.to_csv(OUTPUT_DIR / "occupation_analysis.csv")

    correlation = df.select_dtypes(include=np.number).corr().round(3)
    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    r_sleep, p_sleep = correlation_test(df["Sleep Duration"], df["Quality of Sleep"])
    r_stress, p_stress = correlation_test(df["Stress Level"], df["Quality of Sleep"])
    disorder_scores = df.loc[df["Sleep Disorder"].notna(), "Quality of Sleep"]
    no_disorder_scores = df.loc[df["Sleep Disorder"].isna(), "Quality of Sleep"]
    t_stat, p_disorder = welch_test(disorder_scores, no_disorder_scores)

    plt.figure(figsize=(9, 5)); sns.histplot(df["Age"], bins=15, kde=True)
    plt.title("Age Distribution"); plt.xlabel("Age"); plt.ylabel("People"); save_plot("01_age_distribution.png")
    plt.figure(figsize=(9, 5)); sns.histplot(df["Sleep Duration"], bins=15, kde=True)
    plt.title("Sleep Duration Distribution"); plt.xlabel("Sleep duration (hours)"); plt.ylabel("People"); save_plot("02_sleep_duration.png")
    plt.figure(figsize=(9, 5)); sns.boxplot(data=df, x="Stress Level", y="Quality of Sleep")
    plt.title("Stress Level vs Sleep Quality"); save_plot("03_stress_sleep_quality.png")
    plt.figure(figsize=(9, 5)); sns.scatterplot(data=df, x="Physical Activity Level", y="Quality of Sleep", hue="Gender")
    plt.title("Physical Activity vs Sleep Quality"); save_plot("04_activity_sleep.png")
    plt.figure(figsize=(12, 6)); sns.barplot(data=df, x="Occupation", y="Sleep Duration", errorbar=None)
    plt.title("Average Sleep Duration by Occupation"); plt.xticks(rotation=45, ha="right"); save_plot("05_occupation_sleep.png")
    plt.figure(figsize=(12, 8)); sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, center=0)
    plt.title("Lifestyle Data Correlation Heatmap"); save_plot("06_correlation_heatmap.png")
    disorder_labels = df["Sleep Disorder"].fillna("No recorded disorder")
    plt.figure(figsize=(8, 5)); sns.countplot(x=disorder_labels, order=disorder_labels.value_counts().index)
    plt.title("Sleep Disorder Distribution"); plt.xlabel("Sleep disorder"); plt.ylabel("People"); save_plot("07_sleep_disorder.png")
    plt.figure(figsize=(9, 5)); sns.barplot(data=df, x="Age Group", y="Sleep Duration", errorbar=None)
    plt.title("Age Group vs Average Sleep Duration"); plt.xlabel("Age group"); save_plot("08_age_group_sleep.png")

    pair_values = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
    strongest_pair = pair_values.abs().idxmax()
    strongest_value = correlation.loc[strongest_pair[0], strongest_pair[1]]
    report = f"""MENTAL HEALTH AND LIFESTYLE ANALYSIS
{'=' * 50}

Total records: {len(df)}
Average sleep duration: {df['Sleep Duration'].mean():.2f} hours
Average sleep quality: {df['Quality of Sleep'].mean():.2f}
Average stress level: {df['Stress Level'].mean():.2f}

Hypothesis tests
Sleep duration vs quality: r={r_sleep:.3f}, p={p_sleep:.5f}
Stress level vs quality: r={r_stress:.3f}, p={p_stress:.5f}
Sleep quality (recorded disorder vs no recorded disorder): t={t_stat:.3f}, p={p_disorder:.5f}

Highest average sleep duration occupation: {occupation_analysis.index[0]}
Most common sleep-disorder category: {disorder_labels.value_counts().idxmax()}
Strongest numerical correlation: {strongest_pair[0]} and {strongest_pair[1]} ({strongest_value:.3f})

Occupation analysis
{occupation_analysis.to_string()}
"""
    (OUTPUT_DIR / "analysis_report.txt").write_text(report, encoding="utf-8")
    print("\nAnalysis complete. Created 8 charts, 5 CSV files, and analysis_report.txt in output/.")


if __name__ == "__main__":
    main()
