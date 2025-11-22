import pandas as pd


def load_time_series(path, column="GLD"):
    df = pd.read_csv(path, sep=",", parse_dates=["Date"], dayfirst=False)
    df.set_index("Date", inplace=True)
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])
    return df[column]
