# AirLens — India Air Quality Intelligence Dashboard

A polished, interactive Streamlit dashboard for exploring India’s daily air-quality data from 2015–2020. Compare cities, examine AQI trends, investigate pollutant relationships, and identify changing environmental risks.

## Features

- City and date-range filters for focused analysis
- KPI overview of AQI, coverage, and readings
- Interactive India map with AQI category legend
- Daily, weekly, and monthly AQI trend comparison
- Pollutant concentration trends and AQI correlation analysis
- City scorecard with best, average, and peak AQI
- Automatic demo mode so the project opens even before the source dataset is added

## Project structure

```text
CodeAlpha_ClimateDashboard/
├── app.py
├── generate_outputs.py    # Creates EDA charts and CSV summaries
├── requirements.txt
├── README.md
├── dataset/
│   └── city_day.csv       # Downloaded source dataset (not committed)
└── output/                # Optional exports or screenshots
```

## Run locally

1. Install Python 3.10 or later.
2. In this project folder, create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

   Or, after setup, simply double-click `START_DASHBOARD.bat`.

The dashboard will open at `http://localhost:8501`.

## Generate EDA output files

To create the numbered PNG charts, cleaned data, summary tables, and analysis report used in a project submission:

```powershell
python generate_outputs.py
```

All artefacts are saved to `output/`. The generator uses the real dataset when it is present and otherwise produces a reproducible demo analysis.

## Add the real dataset

Download the **Air Quality Data in India (2015–2020)** dataset and place its `city_day.csv` file here:

```text
dataset/city_day.csv
```

The app automatically detects the file and switches from demo data to the original dataset. It expects `City`, `Date`, `AQI`, and (where available) pollutant columns such as `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, and `O3`.

## AQI categories

The dashboard uses India’s National Air Quality Index bands: Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), and Severe (401–500).

## Suggested presentation talking points

- Air quality is not uniform: city-level comparisons reveal different local risk profiles.
- Monthly aggregation makes seasonal patterns visible and reduces daily noise.
- Pollutant correlations are exploratory signals, not proof of causation.
- The tool helps users transform raw records into a practical, location-aware environmental story.
