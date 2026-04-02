"""
Data loading and preprocessing for BTC price history.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Tuple
import warnings


class BTCDataLoader:
    """Loads and preprocesses historical BTC price data."""

    def __init__(
        self,
        ticker: str = "BTC-USD",
        start_date: str = "2013-01-01",
        data_dir: str = "data"
    ):
        """
        Initialize data loader.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker symbol
        start_date : str
            Start date in YYYY-MM-DD format
        data_dir : str
            Directory for storing raw and processed data
        """
        self.ticker = ticker
        self.start_date = start_date
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical BTC price data from Yahoo Finance.

        Parameters
        ----------
        force_refresh : bool
            If True, re-download data even if cached version exists

        Returns
        -------
        pd.DataFrame
            Raw OHLCV data with DatetimeIndex
        """
        cache_path = self.raw_dir / f"{self.ticker}_{self.start_date}_raw.parquet"

        # Check cache
        if cache_path.exists() and not force_refresh:
            print(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)

        # Download fresh data
        print(f"Downloading {self.ticker} data from {self.start_date}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress yfinance warnings
            df = yf.download(
                self.ticker,
                start=self.start_date,
                progress=False
            )

        if df.empty:
            raise ValueError(f"No data returned for {self.ticker} starting {self.start_date}")

        # Save to cache
        df.to_parquet(cache_path)
        print(f"Saved raw data to {cache_path}")

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw OHLCV data.

        - Compute log returns
        - Handle missing values
        - Add useful derived columns

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data

        Returns
        -------
        pd.DataFrame
            Processed data with log returns and derived features
        """
        df = df.copy()

        # Use Close price for returns calculation
        # Handle multi-level columns from yfinance if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Ensure we have Close column
        if 'Close' not in df.columns:
            raise ValueError("Close column not found in data")

        # Compute log returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Simple returns for reference
        df['simple_return'] = df['Close'].pct_change()

        # Drop first row (NaN returns)
        df = df.dropna(subset=['log_return'])

        # Check for any remaining NaN or inf values
        if df['log_return'].isnull().any():
            print(f"Warning: Found {df['log_return'].isnull().sum()} NaN values in log returns. Forward filling...")
            df['log_return'] = df['log_return'].fillna(method='ffill')

        if np.isinf(df['log_return']).any():
            print(f"Warning: Found {np.isinf(df['log_return']).sum()} inf values in log returns. Capping...")
            # Cap extreme values at 99.9th percentile
            upper = df['log_return'].quantile(0.999)
            lower = df['log_return'].quantile(0.001)
            df['log_return'] = df['log_return'].clip(lower, upper)

        return df

    def load_processed_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load processed data (fetch + preprocess), using cache when available.

        Parameters
        ----------
        force_refresh : bool
            If True, re-process data even if cached

        Returns
        -------
        pd.DataFrame
            Processed data with log returns
        """
        cache_path = self.processed_dir / f"{self.ticker}_{self.start_date}_processed.parquet"

        # Check cache
        if cache_path.exists() and not force_refresh:
            print(f"Loading processed data from {cache_path}")
            return pd.read_parquet(cache_path)

        # Fetch and process
        raw_df = self.fetch_data(force_refresh=force_refresh)
        processed_df = self.preprocess(raw_df)

        # Save to cache
        processed_df.to_parquet(cache_path)
        print(f"Saved processed data to {cache_path}")

        return processed_df

    def get_train_test_split(
        self,
        train_pct: float = 0.7,
        force_refresh: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data and split into training and testing sets.

        Parameters
        ----------
        train_pct : float
            Fraction of data to use for training (rest for testing)
        force_refresh : bool
            If True, re-download and re-process data

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_df, test_df)
        """
        df = self.load_processed_data(force_refresh=force_refresh)

        # Split at the train_pct point
        split_idx = int(len(df) * train_pct)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"\nData split:")
        print(f"  Training: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} days)")
        print(f"  Testing:  {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} days)")

        return train_df, test_df

    def get_returns_array(
        self,
        df: pd.DataFrame | None = None,
        force_refresh: bool = False
    ) -> np.ndarray:
        """
        Get log returns as a numpy array.

        Parameters
        ----------
        df : pd.DataFrame, optional
            If provided, extract returns from this dataframe.
            If None, load all processed data.
        force_refresh : bool
            If True and df is None, re-download data

        Returns
        -------
        np.ndarray
            1D array of log returns
        """
        if df is None:
            df = self.load_processed_data(force_refresh=force_refresh)

        return df['log_return'].values


def get_price_array(df: pd.DataFrame) -> np.ndarray:
    """
    Extract price array from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'Close' column

    Returns
    -------
    np.ndarray
        1D array of closing prices
    """
    return df['Close'].values
