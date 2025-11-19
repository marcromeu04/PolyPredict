"""
Integration tests for PolyPredict system
"""

import sys
sys.path.append('..')

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_collection.polymarket_client import PolymarketClient
from src.detection.rule_based_detector import RuleBasedDetector
from src.models.insider_ml_model import InsiderMLModel
from src.utils.data_preprocessor import DataPreprocessor


class TestPolymarketClient:
    """Test Polymarket API client"""

    def test_client_initialization(self):
        """Test client can be initialized"""
        client = PolymarketClient()
        assert client is not None
        assert client.clob_api is not None

    def test_get_markets(self):
        """Test fetching markets"""
        client = PolymarketClient()
        markets = client.get_markets(active=True)

        # Should return a list (may be empty)
        assert isinstance(markets, list)
        print(f"✓ Fetched {len(markets)} markets")


class TestRuleBasedDetector:
    """Test rule-based detection"""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trading data"""
        n_trades = 200
        base_time = datetime.now() - timedelta(days=1)

        trades = pd.DataFrame({
            'timestamp': [base_time + timedelta(minutes=i*5) for i in range(n_trades)],
            'maker': [f'0x{i%20:040x}' for i in range(n_trades)],
            'size': np.random.exponential(100, n_trades),
            'price': np.random.uniform(0.3, 0.7, n_trades),
            'side': np.random.choice(['buy', 'sell'], n_trades),
            'market': 'test_market'
        })

        return trades

    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = RuleBasedDetector()
        assert detector is not None
        assert detector.timing_window == 30

    def test_timing_anomaly_detection(self, sample_trades):
        """Test timing anomaly detection"""
        detector = RuleBasedDetector()
        event_time = sample_trades['timestamp'].max() - timedelta(hours=1)

        anomalies = detector.detect_timing_anomalies(
            sample_trades, event_time, 'test_market'
        )

        assert isinstance(anomalies, list)
        print(f"✓ Detected {len(anomalies)} timing anomalies")

    def test_volume_spike_detection(self, sample_trades):
        """Test volume spike detection"""
        detector = RuleBasedDetector()

        spikes = detector.detect_volume_spikes(
            sample_trades, 'test_market'
        )

        assert isinstance(spikes, list)
        print(f"✓ Detected {len(spikes)} volume spikes")

    def test_wallet_clustering(self, sample_trades):
        """Test wallet clustering"""
        detector = RuleBasedDetector()

        clusters = detector.cluster_wallets(sample_trades)

        assert isinstance(clusters, dict)
        print(f"✓ Found {len(clusters)} wallet clusters")

    def test_generate_report(self, sample_trades):
        """Test full report generation"""
        detector = RuleBasedDetector()
        event_time = sample_trades['timestamp'].max() - timedelta(hours=1)

        report = detector.generate_report(
            sample_trades, 'test_market', event_time
        )

        assert 'market_id' in report
        assert 'detections' in report
        assert 'risk_score' in report
        assert 'summary' in report

        print(f"✓ Generated report: {report['summary']}")
        print(f"✓ Risk score: {report['risk_score']:.2f}")


class TestInsiderMLModel:
    """Test ML model"""

    def test_model_initialization(self):
        """Test model initialization"""
        model = InsiderMLModel()
        assert model is not None
        assert model.sequence_length == 50

    def test_feature_preparation(self):
        """Test feature extraction"""
        model = InsiderMLModel()

        # Create sample trades
        trades = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='5T'),
            'size': np.random.exponential(100, 100),
            'price': np.random.uniform(0.4, 0.6, 100)
        })

        features = model.prepare_features(trades)

        assert not features.empty
        assert len(features) == len(trades)
        print(f"✓ Extracted {features.shape[1]} features")

    def test_sequence_creation(self):
        """Test sequence creation for LSTM"""
        model = InsiderMLModel(sequence_length=10)

        # Sample features
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        sequences, labels = model.create_sequences(X, y)

        assert sequences.shape[1] == 10  # sequence length
        assert sequences.shape[2] == 5   # features
        assert len(sequences) == len(labels)
        print(f"✓ Created {len(sequences)} sequences")

    def test_model_training(self):
        """Test model training"""
        model = InsiderMLModel(sequence_length=10)

        # Create synthetic data
        n_samples = 200
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        sequences, labels = model.create_sequences(X, y)

        # Train LSTM
        model.train_lstm(sequences, labels, epochs=2, batch_size=16)

        assert model.lstm_model is not None
        print("✓ LSTM training successful")

        # Train Random Forest
        model.train_random_forest(sequences, labels)

        assert model.rf_model is not None
        print("✓ Random Forest training successful")

    def test_prediction(self):
        """Test making predictions"""
        model = InsiderMLModel(sequence_length=10)

        # Create and train on small dataset
        X = np.random.randn(200, 5)
        y = np.random.randint(0, 2, 200)

        sequences, labels = model.create_sequences(X, y)

        # Quick training
        model.train_lstm(sequences, labels, epochs=2)
        model.train_random_forest(sequences, labels)

        # Make predictions
        predictions = model.predict(sequences[:10], use_ensemble=True)

        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)
        print(f"✓ Made predictions: {predictions[:3]}")


class TestDataPreprocessor:
    """Test data preprocessing utilities"""

    def test_clean_trades(self):
        """Test trade cleaning"""
        preprocessor = DataPreprocessor()

        # Create messy data
        trades = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-02', 'invalid', '2024-01-03'],
            'maker': ['0x1', '0x2', '0x3', '0x4'],
            'size': [100, 200, None, 300]
        })

        clean = preprocessor.clean_trades(trades)

        # Should remove invalid timestamp and null size
        assert len(clean) < len(trades)
        print(f"✓ Cleaned {len(trades)} -> {len(clean)} trades")

    def test_aggregate_by_trader(self):
        """Test trader aggregation"""
        preprocessor = DataPreprocessor()

        trades = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='5T'),
            'maker': [f'0x{i%5}' for i in range(100)],
            'size': np.random.exponential(100, 100)
        })

        agg = preprocessor.aggregate_by_trader(trades)

        assert not agg.empty
        assert len(agg) == 5  # 5 unique traders
        print(f"✓ Aggregated {len(trades)} trades to {len(agg)} traders")

    def test_time_features(self):
        """Test time feature creation"""
        preprocessor = DataPreprocessor()

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H')
        })

        with_features = preprocessor.create_time_features(df)

        assert 'hour' in with_features.columns
        assert 'day_of_week' in with_features.columns
        assert 'is_weekend' in with_features.columns
        print(f"✓ Created time features: {list(with_features.columns)}")


def run_integration_test():
    """Run full integration test"""
    print("\n" + "="*60)
    print("POLYPREDICT INTEGRATION TEST")
    print("="*60 + "\n")

    # 1. Create sample data
    print("1. Creating sample trading data...")
    n_trades = 500
    base_time = datetime.now() - timedelta(days=7)

    trades_df = pd.DataFrame({
        'timestamp': [base_time + timedelta(minutes=i*30) for i in range(n_trades)],
        'maker': [f'0x{i%50:040x}' for i in range(n_trades)],
        'size': np.random.exponential(100, n_trades),
        'price': np.random.uniform(0.3, 0.7, n_trades),
        'side': np.random.choice(['buy', 'sell'], n_trades),
        'market': 'integration_test_market'
    })
    print(f"✓ Created {len(trades_df)} trades\n")

    # 2. Test rule-based detection
    print("2. Testing rule-based detection...")
    detector = RuleBasedDetector()
    event_time = trades_df['timestamp'].max() - timedelta(hours=2)

    report = detector.generate_report(trades_df, 'test_market', event_time)
    print(f"✓ {report['summary']}\n")

    # 3. Test ML model
    print("3. Testing ML model...")
    ml_model = InsiderMLModel(sequence_length=20)

    features = ml_model.prepare_features(trades_df)
    print(f"✓ Extracted {features.shape[1]} features")

    # Create sequences
    X = features.values
    y = np.random.randint(0, 2, len(X))  # Random labels for testing

    sequences, labels = ml_model.create_sequences(X, y)
    print(f"✓ Created {len(sequences)} sequences")

    # Quick training
    if len(sequences) > 50:
        ml_model.train_lstm(sequences, labels, epochs=2, batch_size=16)
        print("✓ LSTM training successful")

        ml_model.train_random_forest(sequences, labels, n_estimators=10)
        print("✓ Random Forest training successful")

        # Make predictions
        predictions = ml_model.predict(sequences[:10], use_ensemble=True)
        print(f"✓ Made predictions (sample): {predictions[:3]}\n")

    # 4. Test data preprocessing
    print("4. Testing data preprocessing...")
    preprocessor = DataPreprocessor()

    clean_df = preprocessor.clean_trades(trades_df)
    print(f"✓ Cleaned data: {len(clean_df)} trades")

    trader_stats = preprocessor.aggregate_by_trader(clean_df)
    print(f"✓ Aggregated to {len(trader_stats)} unique traders")

    with_features = preprocessor.create_time_features(clean_df)
    print(f"✓ Added time features\n")

    print("="*60)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    # Run pytest tests
    print("Running pytest tests...\n")
    pytest.main([__file__, '-v'])

    # Run integration test
    print("\n")
    run_integration_test()
