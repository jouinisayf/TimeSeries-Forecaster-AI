from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


def fit_arima(series, order=(2, 0, 3)):
    model = ARIMA(series, order=order)
    result = model.fit()
    return result


def fit_garch(residuals, order=(1, 1)):
    model = arch_model(residuals, vol="Garch", p=order[0], q=order[1])
    result = model.fit(disp="off")
    return result


def forecast_combined(series, arima_order=(2, 0, 3), garch_order=(1, 1), horizon=20):
    arima_res = fit_arima(series, arima_order)
    garch_res = fit_garch(arima_res.resid, garch_order)
    forecast = arima_res.forecast(steps=horizon)
    return forecast, arima_res, garch_res
