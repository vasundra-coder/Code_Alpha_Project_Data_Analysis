"""CodeAlpha Task 4: AI & Technology Social Media Sentiment Analysis.

Run with: python sentiment_analysis.py
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATASET_DIR, OUTPUT_DIR = ROOT / "dataset", ROOT / "output"
LABELS, RANDOM_STATE = ["Negative", "Neutral", "Positive"], 42


def clean_text(value: object) -> str:
    """Remove URLs, mentions and punctuation from a social-media post."""
    text = str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def demo_data() -> pd.DataFrame:
    """A balanced fallback that makes the project immediately runnable."""
    base = [
        ("OpenAI", "Positive", "AI tools make my work faster and more creative"),
        ("Microsoft", "Positive", "The new cloud features are excellent and reliable"),
        ("Google", "Positive", "This technology update is useful and impressive"),
        ("Nvidia", "Positive", "The graphics performance is amazing for developers"),
        ("Tesla", "Positive", "Innovation in autonomous systems looks promising"),
        ("OpenAI", "Negative", "The service is slow and the latest update is disappointing"),
        ("Microsoft", "Negative", "I hate the confusing interface and constant errors"),
        ("Google", "Negative", "Privacy concerns make this product frustrating"),
        ("Nvidia", "Negative", "The hardware price is too high and support is poor"),
        ("Tesla", "Negative", "This release has serious bugs and unreliable results"),
        ("OpenAI", "Neutral", "The company announced a model update today"),
        ("Microsoft", "Neutral", "The software is available for Windows users"),
        ("Google", "Neutral", "The event starts at ten o clock tomorrow"),
        ("Nvidia", "Neutral", "The product specification lists sixteen gigabytes"),
        ("Tesla", "Neutral", "The team published a technical documentation page"),
    ]
    rows = [(entity, label, post + suffix) for entity, label, post in base
            for suffix in ("", " according to users", " in the latest review")]
    return pd.DataFrame(rows, columns=["Entity", "Sentiment", "Tweet Text"])


def load_dataset(dataset_path: str | None) -> tuple[pd.DataFrame, str]:
    candidates = [Path(dataset_path)] if dataset_path else [DATASET_DIR / "twitter_training.csv", DATASET_DIR / "twitter_validation.csv"]
    frames, names = [], []
    for file in candidates:
        if file.exists():
            frames.append(pd.read_csv(file, header=None, names=["Tweet ID", "Entity", "Sentiment", "Tweet Text"]))
            names.append(file.name)
    if not frames:
        return demo_data(), "built-in demonstration data (add Kaggle CSV files for full analysis)"
    return pd.concat(frames, ignore_index=True), ", ".join(names)


def vader_label(text: str, analyzer: SentimentIntensityAnalyzer) -> str:
    score = analyzer.polarity_scores(text)["compound"]
    return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"


def save_wordcloud(text: str, title: str, filename: str) -> None:
    if not text.strip():
        return
    cloud = WordCloud(width=1200, height=650, background_color="white", colormap="viridis").generate(text)
    plt.figure(figsize=(12, 6)); plt.imshow(cloud, interpolation="bilinear"); plt.axis("off"); plt.title(title, fontsize=16)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight"); plt.close()


def create_visuals(df: pd.DataFrame, y_test: pd.Series, predictions: pd.Series, metrics: dict[str, float]) -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    counts = df["Sentiment"].value_counts().reindex(LABELS, fill_value=0)
    plt.figure(figsize=(8, 5)); sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False)
    plt.title("Sentiment Distribution"); plt.ylabel("Number of Posts"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_sentiment_distribution.png", dpi=180); plt.close()
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    plt.figure(figsize=(7, 5)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Logistic Regression Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_confusion_matrix.png", dpi=180); plt.close()
    plt.figure(figsize=(8, 5)); sns.barplot(x=list(metrics), y=list(metrics.values()), hue=list(metrics), legend=False)
    plt.ylim(0, 1); plt.title("Model Performance (Weighted)"); plt.ylabel("Score"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_model_performance.png", dpi=180); plt.close()
    save_wordcloud(" ".join(df["Clean_Text"]), "Most Frequent Words: All Posts", "04_wordcloud.png")
    for label in LABELS:
        save_wordcloud(" ".join(df.loc[df["Sentiment"] == label, "Clean_Text"]), f"Most Frequent Words: {label} Posts", f"05_{label.lower()}_wordcloud.png")
    top = df["Entity"].value_counts().head(10).index
    entity = pd.crosstab(df.loc[df["Entity"].isin(top), "Entity"], df.loc[df["Entity"].isin(top), "Sentiment"]).reindex(columns=LABELS, fill_value=0)
    entity.plot(kind="bar", stacked=True, figsize=(11, 6), color=["#e74c3c", "#95a5a6", "#2ecc71"])
    plt.title("Sentiment by Entity (Top 10)"); plt.xlabel("Entity"); plt.ylabel("Posts"); plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_entity_sentiment.png", dpi=180); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Social media sentiment analysis")
    parser.add_argument("--dataset", help="Path to one Kaggle-format CSV file")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("\nSOCIAL MEDIA SENTIMENT ANALYSIS\n")
    df, source = load_dataset(args.dataset)
    print(f"DATASET LOADED SUCCESSFULLY: {source}\n\nSTARTING DATA CLEANING...")
    df = df.dropna(subset=["Entity", "Sentiment", "Tweet Text"]).drop_duplicates().copy()
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.title()
    df = df[df["Sentiment"].isin(LABELS)].copy()  # Excludes Irrelevant.
    df["Clean_Text"] = df["Tweet Text"].map(clean_text)
    df = df[df["Clean_Text"].str.len() > 2].copy()
    if df["Sentiment"].nunique() < 2:
        raise ValueError("At least two sentiment categories are required.")
    print(f"TEXT PREPROCESSING COMPLETED ({len(df):,} usable posts)\n\nRUNNING VADER SENTIMENT ANALYSIS...")
    analyzer = SentimentIntensityAnalyzer()
    df["VADER_Sentiment"] = df["Clean_Text"].map(lambda post: vader_label(post, analyzer))
    print("\nTRAINING MACHINE LEARNING MODEL...")
    train_idx, test_idx = train_test_split(df.index, test_size=0.25, random_state=RANDOM_STATE, stratify=df["Sentiment"])
    x_train, x_test = df.loc[train_idx, "Clean_Text"], df.loc[test_idx, "Clean_Text"]
    y_train, y_test = df.loc[train_idx, "Sentiment"], df.loc[test_idx, "Sentiment"]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(vectorizer.fit_transform(x_train), y_train)
    predictions = pd.Series(model.predict(vectorizer.transform(x_test)), index=test_idx, name="ML_Prediction")
    metrics = {"Accuracy": accuracy_score(y_test, predictions), "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0), "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0), "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0)}
    vader_accuracy = accuracy_score(y_test, df.loc[test_idx, "VADER_Sentiment"])
    print("MODEL TRAINING COMPLETED!\n\nMODEL PERFORMANCE")
    for name, value in metrics.items(): print(f"{name}: {value:.3f}")
    print(f"VADER Test Accuracy: {vader_accuracy:.3f}")
    df.to_csv(OUTPUT_DIR / "cleaned_sentiment_dataset.csv", index=False)
    df["Sentiment"].value_counts().reindex(LABELS, fill_value=0).rename_axis("Sentiment").reset_index(name="Count").to_csv(OUTPUT_DIR / "sentiment_distribution.csv", index=False)
    result = df.loc[test_idx, ["Entity", "Tweet Text", "Sentiment", "VADER_Sentiment"]].copy(); result["ML_Prediction"] = predictions; result.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)
    pd.crosstab(df["Entity"], df["Sentiment"]).reindex(columns=LABELS, fill_value=0).to_csv(OUTPUT_DIR / "entity_sentiment_analysis.csv")
    report = classification_report(y_test, predictions, labels=LABELS, zero_division=0)
    (OUTPUT_DIR / "sentiment_report.txt").write_text("SOCIAL MEDIA SENTIMENT ANALYSIS REPORT\n\n" + f"Dataset source: {source}\nRecords analysed: {len(df)}\n\n" + "LOGISTIC REGRESSION METRICS\n" + "\n".join(f"{name}: {value:.3f}" for name, value in metrics.items()) + f"\nVADER Test Accuracy: {vader_accuracy:.3f}\n\nCLASSIFICATION REPORT\n{report}", encoding="utf-8")
    create_visuals(df, y_test, predictions, metrics)
    print(f"\nFiles saved to: {OUTPUT_DIR}\nSENTIMENT ANALYSIS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
