# Social Media Sentiment Analysis

CodeAlpha Data Analytics Internship — Task 4.

This project categorises AI and technology social-media posts as **Positive**,
**Negative**, or **Neutral**. It compares VADER with a TF-IDF + Logistic
Regression classifier and generates charts, word clouds, CSV exports, and a
metrics report.

## Run it

```powershell
python -m pip install -r requirements.txt
python sentiment_analysis.py
```

If the Kaggle data has not been added yet, the program uses a small built-in
demo dataset so every output can be tested immediately.

## Full dataset

Download [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis), then put `twitter_training.csv`
and optionally `twitter_validation.csv` in `dataset/`. The headerless source
format is handled automatically and `Irrelevant` posts are excluded.

## Outputs

The `output/` folder receives cleaned data, predictions, entity analysis,
sentiment distribution, a written evaluation report, a confusion matrix,
performance chart, distribution chart, word clouds, and an entity chart.

## Tools used

Python, Pandas, VADER, scikit-learn, TF-IDF, Logistic Regression, Matplotlib,
Seaborn, and WordCloud.
