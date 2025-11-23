import pandas as pd
import os
import logging

logger = logging.getLogger("TimeSeriesForecaster")


def load_time_series(path, column="GLD"):
    """
    Load and clean a financial time series from a CSV file.

    The function:
    - loads a CSV file containing a date column and financial indicators
    - converts the selected column to numeric
    - sets the "Date" column as index
    - removes invalid or missing values
    - logs detailed steps for debugging and traceability

    Parameters
    ----------
    path : str
        Path to the CSV dataset.
    column : str, optional
        Name of the column to extract as a time series (default is "GLD").

    Returns
    -------
    pandas.Series
        A cleaned time series indexed by date.

    Raises
    ------
    FileNotFoundError
        If the provided file path does not exist.
    ValueError
        If the column does not exist in the dataset.

    Notes
    -----
    Logging Levels used:
        - INFO: loading steps and final dataset size
        - DEBUG: head of raw DataFrame before cleaning
        - WARNING: number of removed missing values
        - CRITICAL: missing file error before raising exception
    """
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
