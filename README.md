# TimeSeries Forecaster AI
# Automatic Time Series Forecasting Tool using ARIMA and GARCH Models
# Author: Jouini Saief Eddine

## Project Description  
This project implements a **time series forecasting system** based on:  
- **ARIMA** to model temporal trends and structure,  
- **GARCH** to model volatility and dynamic variance,  
- and **Poetry** as a modern build and dependency management tool.

The goal is to demonstrate full project automation with Poetry while applying advanced statistical models for financial time series prediction.

## Project Structure  

TimeSeries-Forecaster-AI/
├── data/
│ └── gld_price_data.csv ← Dataset used for modeling
├── src/
│ ├── init.py
│ ├── preprocess.py ← Data loading and preprocessing
│ ├── models.py ← ARIMA and GARCH model implementation
│ ├── visualize.py ← Forecast visualization
│ └── main.py ← Main entry point
├── tests/ ← Empty for now 
├── pyproject.toml ← Poetry configuration and metadata
└── README.md

## How to build
### Step 1 — Install Poetry
Install Poetry globally (once on your machine):
```
pip install poetry```

Check that Poetry is installed:

```poetry --version```
### Step 2 — Create the Project

Move to your working directory and Create the project folder:
```
mkdir TimeSeries-Forecaster-AI
cd TimeSeries-Forecaster-AI```

Initialize a Poetry project:
```
poetry init```

When prompted, add the dependencies manually later (see below).

### Step 3 — Add Dependencies

Install all required libraries through Poetry:

```poetry add pandas numpy matplotlib seaborn statsmodels arch```

This automatically updates pyproject.toml with the proper dependency versions and creates a virtual environment.

### Step 4 — Project Implementation

Inside the src/ folder, create the following files:

| File           | Description                                                |
|----------------|------------------------------------------------------------|
| preprocess.py  | Reads the CSV, sets Date as index, cleans missing values   |
| models.py      | Defines `fit_arima()`, `fit_garch()`, and `forecast_combined()` |
| visualize.py   | Displays forecast results using Matplotlib                 |
| main.py        | Combines all modules and runs the pipeline                 |

### Step 5 — Build the Project

Once the code is ready, build the package automatically with Poetry:
```
poetry build```

This step compiles your code and generates distributable packages containing:
project metadata (name, version, license, author),all dependencies declared in pyproject.toml.

### Step 6 — Run the Project

To execute the project inside the Poetry environment:
```
poetry run python src/main.py```

This will:
Load the dataset data/gld_price_data.csv,
Fit an ARIMA(2,0,3) model on the gold price series,
Apply a GARCH(1,1) model on residuals,
Forecast the next 20 values,
Display a Matplotlib window with real (blue) and predicted (red) series,
Print ARIMA and GARCH summaries in the terminal.

Example terminal output:

ARIMA Summary:
SARIMAX Results
Dep. Variable: GLD
...

GARCH Summary:
Mean Model: ARIMA(2,0,3)
Volatility Model: GARCH(1,1)
...

