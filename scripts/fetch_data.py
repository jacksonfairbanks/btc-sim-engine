#!/usr/bin/env python
"""Fetch and cache historical BTC price data."""
import click
from data.loader import BTCDataLoader


@click.command()
@click.option("--ticker", default="BTC-USD", help="Yahoo Finance ticker.")
@click.option("--start-date", default="2013-01-01", help="Start date (YYYY-MM-DD).")
@click.option("--force", is_flag=True, help="Force re-download even if cached.")
def main(ticker: str, start_date: str, force: bool):
    """Download and preprocess historical BTC data."""
    loader = BTCDataLoader(ticker=ticker, start_date=start_date)

    print(f"Fetching {ticker} from {start_date}...")
    df = loader.load_processed_data(force_refresh=force)

    print(f"\nData summary:")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Total days: {len(df)}")
    print(f"  Log return stats:")
    print(f"    Mean:     {df['log_return'].mean():.6f}")
    print(f"    Std:      {df['log_return'].std():.6f}")
    print(f"    Skew:     {df['log_return'].skew():.4f}")
    print(f"    Kurtosis: {df['log_return'].kurtosis():.4f}")
    print(f"    Min:      {df['log_return'].min():.6f}")
    print(f"    Max:      {df['log_return'].max():.6f}")


if __name__ == "__main__":
    main()
