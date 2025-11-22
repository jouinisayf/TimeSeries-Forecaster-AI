from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import logging

logger = logging.getLogger("TimeSeriesForecaster")


def fit_arima(series, order=(2, 0, 3)):
    logger.info(f"Training ARIMA model with order={order}")
    try:
        model = ARIMA(series, order=order)
        result = model.fit()
        logger.debug(f"ARIMA fitted parameters: {result.params}")
        return result
    except Exception as e:
        logger.error(f"ARIMA training failed: {str(e)}")
        raise


def fit_garch(residuals, order=(1, 1)):
    logger.info(f"Training GARCH model with order={order}")
    try:
        model = arch_model(residuals, vol="Garch", p=order[0], q=order[1])
        result = model.fit(disp="off")
        logger.debug(f"GARCH fitted parameters: {result.params}")
        return result
    except Exception as e:
        logger.error(f"GARCH training failed: {str(e)}")
        raise


def forecast_combined(series, arima_order=(2, 0, 3), garch_order=(1, 1), horizon=20):
    logger.info("Starting combined ARIMA + GARCH forecasting")
    arima_res = fit_arima(series, arima_order)
    garch_res = fit_garch(arima_res.resid, garch_order)
    forecast = arima_res.forecast(steps=horizon)

    logger.debug(f"Generated forecast for next {horizon} points: {forecast}")
    logger.info("Combined forecast completed")
    return forecast, arima_res, garch_res
