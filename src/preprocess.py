import pandas as pd
import os
import logging

logger = logging.getLogger("TimeSeriesForecaster")


def load_time_series(path, column="GLD"):
    logger.info(f"Loading dataset from {path}")

    if not os.path.exists(path):
        logger.critical(f"Dataset not found at path: {path}")
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=",", parse_dates=["Date"], dayfirst=False)
    logger.debug(f"Raw DataFrame head: {df.head()}")

    df.set_index("Date", inplace=True)
    before_drop = df[column].isna().sum()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])
    after_drop = df[column].isna().sum()

    if before_drop != after_drop:
        logger.warning(f"{before_drop - after_drop} missing values removed in column {column}")

    logger.info(f"Dataset loaded successfully: {len(df)} rows")
    return df[column]
