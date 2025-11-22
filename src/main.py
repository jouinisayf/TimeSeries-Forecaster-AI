import logging
from preprocess import load_time_series
from models import forecast_combined
from visualize import plot_forecast

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger("TimeSeriesForecaster")


def main():
    logger.info("Starting forecasting pipeline")

    series = load_time_series("data/gld_price_data.csv", column="GLD")
    logger.info("Data successfully loaded")

    forecast, arima_res, garch_res = forecast_combined(series, (2, 0, 3), (1, 1), 20)

    logger.info("Models trained successfully")
    logger.debug(f"Forecast array: {forecast}")

    print("ARIMA Summary:")
    print(arima_res.summary())
    print("\nGARCH Summary:")
    print(garch_res.summary())

    logger.info("Plotting forecast")
    plot_forecast(series, forecast, 20)

    logger.info("Pipeline executed successfully")


if __name__ == "__main__":
    main()
