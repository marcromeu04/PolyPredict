"""
Data preprocessing utilities for insider trading detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Utilities for preprocessing trading data"""

    @staticmethod
    def clean_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate trades data

        Args:
            trades_df: Raw trades DataFrame

        Returns:
            Cleaned DataFrame
        """
        if trades_df.empty:
            return trades_df

        df = trades_df.copy()

        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df)} duplicate trades")

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Remove invalid timestamps
        if 'timestamp' in df.columns:
            df = df[df['timestamp'].notna()]

        # Ensure numeric columns
        numeric_cols = ['size', 'price', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with missing critical data
        critical_cols = ['maker', 'size']
        for col in critical_cols:
            if col in df.columns:
                df = df[df[col].notna()]

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Cleaned trades: {len(df)} valid trades")

        return df

    @staticmethod
    def aggregate_by_trader(
        trades_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate trades by trader

        Args:
            trades_df: Trades DataFrame

        Returns:
            Aggregated DataFrame with trader statistics
        """
        if trades_df.empty or 'maker' not in trades_df.columns:
            return pd.DataFrame()

        agg_dict = {
            'size': ['sum', 'mean', 'std', 'count'],
        }

        if 'price' in trades_df.columns:
            agg_dict['price'] = ['mean', 'std']

        if 'timestamp' in trades_df.columns:
            agg_dict['timestamp'] = ['min', 'max']

        trader_stats = trades_df.groupby('maker').agg(agg_dict)

        # Flatten column names
        trader_stats.columns = ['_'.join(col).strip() for col in trader_stats.columns]

        # Calculate additional metrics
        trader_stats['total_volume'] = trader_stats.get('size_sum', 0)
        trader_stats['num_trades'] = trader_stats.get('size_count', 0)
        trader_stats['avg_trade_size'] = trader_stats.get('size_mean', 0)

        # Calculate trading period
        if 'timestamp_min' in trader_stats.columns and 'timestamp_max' in trader_stats.columns:
            trader_stats['trading_period_hours'] = (
                (trader_stats['timestamp_max'] - trader_stats['timestamp_min'])
                .dt.total_seconds() / 3600
            )

        return trader_stats.reset_index()

    @staticmethod
    def resample_timeseries(
        trades_df: pd.DataFrame,
        freq: str = '5T',
        agg_func: str = 'sum'
    ) -> pd.DataFrame:
        """
        Resample trading data to regular time intervals

        Args:
            trades_df: Trades DataFrame with timestamp
            freq: Resampling frequency (e.g., '1T', '5T', '1H')
            agg_func: Aggregation function

        Returns:
            Resampled DataFrame
        """
        if trades_df.empty or 'timestamp' not in trades_df.columns:
            return pd.DataFrame()

        df = trades_df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set timestamp as index
        df = df.set_index('timestamp')

        # Resample
        resampled = df.resample(freq).agg({
            'size': agg_func,
            'maker': 'count'
        })

        resampled.columns = ['volume', 'trade_count']

        return resampled.reset_index()

    @staticmethod
    def calculate_market_metrics(
        trades_df: pd.DataFrame,
        window_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Calculate market-level metrics

        Args:
            trades_df: Trades DataFrame
            window_minutes: Rolling window size in minutes

        Returns:
            DataFrame with market metrics
        """
        if trades_df.empty:
            return pd.DataFrame()

        df = trades_df.copy()

        # Ensure timestamp
        if 'timestamp' not in df.columns:
            return pd.DataFrame()

        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        df = df.set_index('timestamp').sort_index()

        # Resample to 1-minute intervals
        resampled = df.resample('1T').agg({
            'size': 'sum',
            'maker': 'nunique'
        })

        metrics = pd.DataFrame()
        metrics['volume'] = resampled['size']
        metrics['unique_traders'] = resampled['maker']

        # Rolling statistics
        window = f'{window_minutes}T'
        metrics['volume_rolling_mean'] = metrics['volume'].rolling(
            window=window_minutes, min_periods=1
        ).mean()
        metrics['volume_rolling_std'] = metrics['volume'].rolling(
            window=window_minutes, min_periods=1
        ).std()

        # Calculate volatility
        if 'price' in df.columns:
            price_resample = df['price'].resample('1T').mean()
            metrics['price'] = price_resample
            metrics['price_volatility'] = metrics['price'].rolling(
                window=window_minutes, min_periods=1
            ).std()

        return metrics.reset_index()

    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers in a series

        Args:
            series: Data series
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection

        Returns:
            Boolean series indicating outliers
        """
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (series < lower) | (series > upper)

        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def create_time_features(
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Create time-based features

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with additional time features
        """
        if df.empty or timestamp_col not in df.columns:
            return df

        df = df.copy()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract features
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['is_weekend'] = df[timestamp_col].dt.dayofweek >= 5
        df['is_business_hours'] = df['hour'].between(9, 17)

        # Time since first trade
        if len(df) > 0:
            first_timestamp = df[timestamp_col].min()
            df['hours_since_start'] = (
                (df[timestamp_col] - first_timestamp).dt.total_seconds() / 3600
            )

        return df


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()

    # Create sample data
    sample_trades = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5T'),
        'maker': [f'0x{i%10:040x}' for i in range(100)],
        'size': np.random.exponential(100, 100),
        'price': np.random.uniform(0.4, 0.6, 100)
    })

    # Test cleaning
    clean_df = preprocessor.clean_trades(sample_trades)
    print(f"Cleaned: {len(clean_df)} trades")

    # Test aggregation
    trader_stats = preprocessor.aggregate_by_trader(clean_df)
    print(f"\nTrader stats:\n{trader_stats.head()}")

    # Test time features
    with_features = preprocessor.create_time_features(clean_df)
    print(f"\nTime features: {list(with_features.columns)}")
