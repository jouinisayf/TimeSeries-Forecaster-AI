from preprocess import load_time_series
from models import forecast_combined
from visualize import plot_forecast

def main():
    series = load_time_series("data/gld_price_data.csv", column="GLD")
    forecast, arima_res, garch_res = forecast_combined(series, (2,0,3), (1,1), 20)

    print("ARIMA Summary:")
    print(arima_res.summary())
    print("\nGARCH Summary:")
    print(garch_res.summary())

    plot_forecast(series, forecast, 20)

if __name__ == "__main__":
    main()

