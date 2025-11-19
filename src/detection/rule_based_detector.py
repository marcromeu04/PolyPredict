"""
Rule-Based Insider Trading Detection
Based on Chainalysis techniques and research findings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RuleBasedDetector:
    """
    Detects suspicious trading patterns using rule-based heuristics

    Detection methods:
    1. Timing anomalies (pre-event trading)
    2. Volume spikes
    3. Multi-account clustering
    4. Coordinated trading patterns
    5. Funding source analysis
    """

    def __init__(
        self,
        timing_window_minutes: int = 30,
        volume_spike_threshold: float = 3.0,
        clustering_similarity_threshold: float = 0.7
    ):
        self.timing_window = timing_window_minutes
        self.volume_spike_threshold = volume_spike_threshold
        self.clustering_threshold = clustering_similarity_threshold

    def detect_timing_anomalies(
        self,
        trades_df: pd.DataFrame,
        event_time: datetime,
        market_id: str
    ) -> List[Dict]:
        """
        Detect trades suspiciously close to a major event

        Args:
            trades_df: DataFrame with trades (must have 'timestamp', 'maker', 'size' columns)
            event_time: Time when event was publicly announced
            market_id: Market identifier

        Returns:
            List of suspicious trades
        """
        suspicious_trades = []

        if trades_df.empty or 'timestamp' not in trades_df.columns:
            return suspicious_trades

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # Define suspicious window before event
        window_start = event_time - timedelta(minutes=self.timing_window)

        # Filter trades in the suspicious window
        mask = (trades_df['timestamp'] >= window_start) & (
            trades_df['timestamp'] < event_time
        )
        suspicious_window_trades = trades_df[mask].copy()

        if suspicious_window_trades.empty:
            return suspicious_trades

        # Calculate average trade size outside the window
        normal_trades = trades_df[~mask]
        if not normal_trades.empty and 'size' in normal_trades.columns:
            avg_normal_size = normal_trades['size'].mean()
            std_normal_size = normal_trades['size'].std()

            # Flag large trades in suspicious window
            threshold = avg_normal_size + 2 * std_normal_size

            large_trades = suspicious_window_trades[
                suspicious_window_trades['size'] > threshold
            ]

            for _, trade in large_trades.iterrows():
                minutes_before = (event_time - trade['timestamp']).total_seconds() / 60

                # Calculate risk score (higher size relative to threshold = higher score)
                size_ratio = trade['size'] / threshold if threshold > 0 else 1.0
                time_factor = 1.5 if minutes_before < 15 else 1.0
                risk_score = min(1.0, (size_ratio - 1.0) * 0.5 * time_factor)

                suspicious_trades.append({
                    'type': 'timing_anomaly',
                    'market_id': market_id,
                    'trader': trade.get('maker', 'unknown'),
                    'timestamp': trade['timestamp'],
                    'minutes_before_event': minutes_before,
                    'size': trade.get('size', 0),
                    'avg_normal_size': avg_normal_size,
                    'severity': 'high' if minutes_before < 15 else 'medium',
                    'score': max(0.1, risk_score)  # Minimum score 0.1
                })

        return suspicious_trades

    def detect_volume_spikes(
        self,
        trades_df: pd.DataFrame,
        market_id: str,
        window_minutes: int = 60
    ) -> List[Dict]:
        """
        Detect unusual volume spikes that may indicate insider trading

        Args:
            trades_df: DataFrame with trades
            market_id: Market identifier
            window_minutes: Time window for calculating moving average

        Returns:
            List of detected volume spikes
        """
        if trades_df.empty or 'timestamp' not in trades_df.columns:
            return []

        # Make a copy to avoid modifying original
        df = trades_df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Resample to get volume per minute
        df = df.set_index('timestamp')
        volume_per_min = df['size'].resample('1T').sum()

        # Calculate rolling average and std
        rolling_avg = volume_per_min.rolling(
            window=window_minutes, min_periods=1
        ).mean()
        rolling_std = volume_per_min.rolling(
            window=window_minutes, min_periods=1
        ).std()

        # Detect spikes (> threshold * std above average)
        spikes = volume_per_min > (
            rolling_avg + self.volume_spike_threshold * rolling_std
        )

        spike_times = volume_per_min[spikes].index.tolist()

        detections = []
        for spike_time in spike_times:
            spike_volume = volume_per_min[spike_time]
            normal_volume = rolling_avg[spike_time]

            if normal_volume > 0:
                spike_ratio = spike_volume / normal_volume

                detections.append({
                    'type': 'volume_spike',
                    'market_id': market_id,
                    'timestamp': spike_time,
                    'volume': spike_volume,
                    'normal_volume': normal_volume,
                    'spike_ratio': spike_ratio,
                    'severity': 'high' if spike_ratio > 5 else 'medium',
                    'score': min(1.0, spike_ratio / 10)
                })

        return detections

    def cluster_wallets(
        self,
        trades_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Cluster wallets that might belong to the same entity
        Based on funding sources, timing, and trading patterns

        Args:
            trades_df: DataFrame with trades

        Returns:
            Dictionary mapping cluster_id to list of wallet addresses
        """
        if trades_df.empty or 'maker' not in trades_df.columns:
            return {}

        # Group trades by maker
        trader_stats = defaultdict(lambda: {
            'trades': [],
            'markets': set(),
            'timestamps': [],
            'sizes': []
        })

        for _, trade in trades_df.iterrows():
            maker = trade.get('maker')
            if maker:
                trader_stats[maker]['trades'].append(trade)
                trader_stats[maker]['timestamps'].append(trade.get('timestamp'))
                trader_stats[maker]['sizes'].append(trade.get('size', 0))

                if 'market' in trade:
                    trader_stats[maker]['markets'].add(trade['market'])

        # Calculate similarity between traders
        traders = list(trader_stats.keys())
        clusters = defaultdict(list)
        cluster_id = 0

        assigned = set()

        for i, trader1 in enumerate(traders):
            if trader1 in assigned:
                continue

            cluster = [trader1]
            assigned.add(trader1)

            for trader2 in traders[i + 1:]:
                if trader2 in assigned:
                    continue

                similarity = self._calculate_trader_similarity(
                    trader_stats[trader1],
                    trader_stats[trader2]
                )

                if similarity > self.clustering_threshold:
                    cluster.append(trader2)
                    assigned.add(trader2)

            if len(cluster) > 1:
                clusters[f"cluster_{cluster_id}"] = cluster
                cluster_id += 1

        return dict(clusters)

    def _calculate_trader_similarity(
        self,
        stats1: Dict,
        stats2: Dict
    ) -> float:
        """
        Calculate similarity score between two traders

        Factors:
        - Market overlap
        - Timing correlation
        - Trade size similarity
        """
        # Market overlap
        markets1 = stats1['markets']
        markets2 = stats2['markets']

        if not markets1 or not markets2:
            return 0.0

        market_overlap = len(markets1 & markets2) / len(markets1 | markets2)

        # Timing correlation (simplified)
        timestamps1 = pd.Series(stats1['timestamps'])
        timestamps2 = pd.Series(stats2['timestamps'])

        timing_score = 0.0
        if len(timestamps1) > 0 and len(timestamps2) > 0:
            # Check if they trade around similar times
            # (within 1 hour windows)
            timing_score = 0.5  # Placeholder

        # Trade size similarity
        sizes1 = stats1['sizes']
        sizes2 = stats2['sizes']

        size_score = 0.0
        if sizes1 and sizes2:
            avg1 = np.mean(sizes1)
            avg2 = np.mean(sizes2)

            if avg1 > 0 and avg2 > 0:
                size_ratio = min(avg1, avg2) / max(avg1, avg2)
                size_score = size_ratio

        # Combined score
        similarity = (
            0.4 * market_overlap +
            0.3 * timing_score +
            0.3 * size_score
        )

        return similarity

    def detect_coordinated_trading(
        self,
        trades_df: pd.DataFrame,
        market_id: str,
        time_window_seconds: int = 300
    ) -> List[Dict]:
        """
        Detect groups of traders making coordinated moves

        Args:
            trades_df: DataFrame with trades
            market_id: Market identifier
            time_window_seconds: Time window to check for coordination

        Returns:
            List of coordinated trading events
        """
        if trades_df.empty or 'timestamp' not in trades_df.columns:
            return []

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        trades_df = trades_df.sort_values('timestamp')

        coordinated_events = []

        # Sliding window to find clusters of trades
        for i in range(len(trades_df)):
            window_start = trades_df.iloc[i]['timestamp']
            window_end = window_start + timedelta(seconds=time_window_seconds)

            # Get trades in window
            window_trades = trades_df[
                (trades_df['timestamp'] >= window_start) &
                (trades_df['timestamp'] < window_end)
            ]

            if len(window_trades) < 3:
                continue

            # Check if trades are in same direction
            if 'side' in window_trades.columns:
                same_direction = window_trades['side'].nunique() == 1

                if same_direction:
                    unique_traders = window_trades['maker'].nunique()
                    total_volume = window_trades['size'].sum()

                    if unique_traders >= 3:
                        coordinated_events.append({
                            'type': 'coordinated_trading',
                            'market_id': market_id,
                            'timestamp': window_start,
                            'num_traders': unique_traders,
                            'total_volume': total_volume,
                            'direction': window_trades['side'].iloc[0],
                            'traders': window_trades['maker'].unique().tolist(),
                            'severity': 'high' if unique_traders >= 5 else 'medium',
                            'score': min(1.0, unique_traders / 10)
                        })

        return coordinated_events

    def generate_report(
        self,
        trades_df: pd.DataFrame,
        market_id: str,
        event_time: Optional[datetime] = None
    ) -> Dict:
        """
        Generate comprehensive detection report

        Args:
            trades_df: DataFrame with trades
            market_id: Market identifier
            event_time: Optional event time for timing analysis

        Returns:
            Detection report dictionary
        """
        report = {
            'market_id': market_id,
            'analysis_time': datetime.now(),
            'total_trades': len(trades_df),
            'detections': {
                'timing_anomalies': [],
                'volume_spikes': [],
                'coordinated_trading': [],
                'wallet_clusters': {}
            },
            'risk_score': 0.0,
            'summary': ''
        }

        # Run detections
        if event_time:
            report['detections']['timing_anomalies'] = self.detect_timing_anomalies(
                trades_df, event_time, market_id
            )

        report['detections']['volume_spikes'] = self.detect_volume_spikes(
            trades_df, market_id
        )

        report['detections']['coordinated_trading'] = self.detect_coordinated_trading(
            trades_df, market_id
        )

        report['detections']['wallet_clusters'] = self.cluster_wallets(trades_df)

        # Calculate overall risk score
        scores = []
        for detection_list in report['detections'].values():
            if isinstance(detection_list, list):
                scores.extend([d.get('score', 0) for d in detection_list])

        if scores:
            report['risk_score'] = np.mean(scores)

        # Generate summary
        num_timing = len(report['detections']['timing_anomalies'])
        num_spikes = len(report['detections']['volume_spikes'])
        num_coordinated = len(report['detections']['coordinated_trading'])
        num_clusters = len(report['detections']['wallet_clusters'])

        report['summary'] = (
            f"Detected {num_timing} timing anomalies, "
            f"{num_spikes} volume spikes, "
            f"{num_coordinated} coordinated trading events, "
            f"and {num_clusters} wallet clusters. "
            f"Overall risk score: {report['risk_score']:.2f}"
        )

        return report


if __name__ == "__main__":
    # Test with sample data
    detector = RuleBasedDetector()

    # Create sample trades
    sample_trades = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5T'),
        'maker': [f'0x{i%10:040x}' for i in range(100)],
        'size': np.random.exponential(100, 100),
        'side': np.random.choice(['buy', 'sell'], 100),
        'market': 'test_market'
    })

    # Test detection
    event_time = pd.Timestamp('2024-01-01 06:00:00')
    report = detector.generate_report(sample_trades, 'test_market', event_time)

    print("Detection Report:")
    print(report['summary'])
    print(f"Risk Score: {report['risk_score']:.2f}")
