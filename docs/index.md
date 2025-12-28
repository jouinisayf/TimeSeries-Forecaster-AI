# TimeSeries-Forecaster-AI  
Automatic ARIMA + GARCH forecasting pipeline using Python & Poetry

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Build](https://img.shields.io/badge/Build-Poetry-informational.svg)

## Overview

TimeSeries-Forecaster-AI is a lightweight but complete forecasting pipeline combining:

- **ARIMA models** (trend + autocorrelation)
- **GARCH models** (volatility modeling)
- **Poetry** as build & dependency manager
- **Logging**, **unit tests**, and **debugging support**

This project demonstrates a modern and automated Python workflow suitable for real-world time-series analysis.

## Project Structure
```
TimeSeries-Forecaster-AI/
├── data/
│ └── gld_price_data.csv
├── src/
│ ├── preprocess.py # Data loading & cleaning
│ ├── models.py # ARIMA/GARCH models
│ ├── visualize.py # Forecast plot
│ └── main.py # Pipeline entry point
├── tests/ # Pytest unit tests
├── pyproject.toml # Poetry configuration
├── LICENSE # MIT License
└── README.md # Project documentation
```
## Features

- Load & clean time-series financial data  
- Train ARIMA(p,d,q) models  
- Train GARCH(p,q) volatility models  
- Combined forecasting  
- Logging at all levels (INFO/DEBUG/WARNING/ERROR)  
- Unit tests using Pytest  
- Debugger-friendly code  
- Poetry-based environment

## Installation

### 1. Install Poetry
```
pip install poetry
```
### 2. Install dependencies
```
poetry install
```
### 3. Usage
Run the full forecasting pipeline:
```
poetry run python src/main.py
```
This will:
Load the dataset
Train ARIMA(2,0,3)
Train GARCH(1,1)
Forecast 20 future values
Display plots
Print statistical summaries


## License
This project is released under the MIT License.




