from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import logging

logger = logging.getLogger("TimeSeriesForecaster")


def fit_arima(series, order=(2, 0, 3)):
    """
    Train an ARIMA model on a time series.

    Parameters
    ----------
    series : pandas.Series
        Cleaned time series on which the ARIMA model will be fitted.
    order : tuple(int, int, int), optional
        ARIMA(p, d, q) order. Default is (2, 0, 3).

    Returns
    -------
    statsmodels.tsa.arima.model.ARIMAResults
        Fitted ARIMA model results.

    Raises
    ------
    Exception
        If the ARIMA fitting process fails.

    Notes
    -----
    Logging levels used:
        - INFO: model initialization
        - DEBUG: fitted parameters
        - ERROR: failure message before raising the exception
    """
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
    """
    Train a GARCH model on ARIMA residuals.

    Parameters
    ----------
    residuals : pandas.Series or ndarray
        Residuals from the ARIMA model.
    order : tuple(int, int), optional
        GARCH(p, q) order. Default is (1, 1).

    Returns
    -------
    arch.univariate.base.ARCHModelResult
        Fitted GARCH model results.

    Raises
    ------
    Exception
        If the GARCH fitting process fails.

    Notes
    -----
    Logging levels used:
        - INFO: model initialization
        - DEBUG: fitted parameters
        - ERROR: failure message before raising the exception
    """
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
    """
    Perform combined ARIMA + GARCH forecasting.

    The process:
    1. Train an ARIMA model on the series.
    2. Train a GARCH model using ARIMA residuals.
    3. Forecast future values using ARIMA.

    Parameters
    ----------
    series : pandas.Series
        Input time series to forecast.
    arima_order : tuple(int, int, int), optional
        ARIMA(p, d, q) order. Default is (2, 0, 3).
    garch_order : tuple(int, int), optional
        GARCH(p, q) order. Default is (1, 1).
    horizon : int, optional
        Number of future time steps to forecast (default is 20).

    Returns
    -------
    tuple
        (forecast, arima_results, garch_results)

        forecast : numpy.ndarray or pandas.Series
            Forecasted values for the specified horizon.

        arima_results : ARIMAResults
            Fitted ARIMA model.

        garch_results : ARCHModelResult
            Fitted GARCH model.

    Notes
    -----
    Logging levels used:
        - INFO: pipeline steps
        - DEBUG: forecast values
    """
    logger.info("Starting combined ARIMA + GARCH forecasting")
    arima_res = fit_arima(series, arima_order)
    garch_res = fit_garch(arima_res.resid, garch_order)
    forecast = arima_res.forecast(steps=horizon)

    logger.debug(f"Generated forecast for next {horizon} points: {forecast}")
    logger.info("Combined forecast completed")
    return forecast, arima_res, garch_res
