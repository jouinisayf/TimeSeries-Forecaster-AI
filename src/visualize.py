import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("TimeSeriesForecaster")


def plot_forecast(series, forecast, horizon):
    logger.info("Displaying forecast plot")

    plt.figure(figsize=(10, 5))
    plt.plot(series, label="Original")
    plt.plot(range(len(series), len(series) + horizon), forecast, label="Forecast", color="red")
    plt.legend()
    plt.title("ARIMA + GARCH Forecast")
    plt.show()

    logger.debug("Forecast plot displayed")
