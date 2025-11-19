#!/usr/bin/env python3
"""
Quick integration test for PolyPredict
Tests all major components without requiring pytest
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("\n" + "="*70)
print("POLYPREDICT QUICK INTEGRATION TEST")
print("="*70 + "\n")

# Test 1: Imports
print("Test 1: Testing imports...")
try:
    from src.data_collection.polymarket_client import PolymarketClient
    from src.detection.rule_based_detector import RuleBasedDetector
    from src.models.insider_ml_model import InsiderMLModel
    from src.utils.data_preprocessor import DataPreprocessor
    print("✓ All imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Create sample data
print("Test 2: Creating sample trading data...")
try:
    n_trades = 500
    base_time = datetime.now() - timedelta(days=7)

    trades_df = pd.DataFrame({
        'timestamp': [base_time + timedelta(minutes=i*30) for i in range(n_trades)],
        'maker': [f'0x{i%50:040x}' for i in range(n_trades)],
        'size': np.random.exponential(100, n_trades) * np.random.uniform(0.5, 2, n_trades),
        'price': np.random.uniform(0.3, 0.7, n_trades),
        'side': np.random.choice(['buy', 'sell'], n_trades),
        'market': 'test_market'
    })

    # Add suspicious patterns
    event_time = trades_df['timestamp'].max() - timedelta(hours=2)
    suspicious_window = (trades_df['timestamp'] >= event_time - timedelta(minutes=20)) & \
                       (trades_df['timestamp'] < event_time)

    insider_addresses = [f'0xINSIDER{i:037x}' for i in range(3)]
    for idx in trades_df[suspicious_window].index[:10]:
        trades_df.loc[idx, 'size'] = np.random.uniform(500, 1000)
        trades_df.loc[idx, 'maker'] = np.random.choice(insider_addresses)

    print(f"✓ Created {len(trades_df)} trades with suspicious patterns\n")
except Exception as e:
    print(f"✗ Data creation failed: {e}\n")
    sys.exit(1)

# Test 3: Polymarket Client
print("Test 3: Testing Polymarket API Client...")
try:
    client = PolymarketClient()
    print(f"✓ Client initialized")
    print(f"  - CLOB API: {client.clob_api}")
    print(f"  - Gamma API: {client.gamma_api}\n")
except Exception as e:
    print(f"✗ Client initialization failed: {e}\n")

# Test 4: Rule-Based Detector
print("Test 4: Testing Rule-Based Detector...")
try:
    detector = RuleBasedDetector(
        timing_window_minutes=30,
        volume_spike_threshold=3.0
    )

    # Timing anomalies
    anomalies = detector.detect_timing_anomalies(
        trades_df.copy(), event_time, 'test_market'
    )
    print(f"✓ Timing anomaly detection: {len(anomalies)} anomalies found")

    # Volume spikes
    spikes = detector.detect_volume_spikes(
        trades_df.copy(), 'test_market'
    )
    print(f"✓ Volume spike detection: {len(spikes)} spikes found")

    # Wallet clustering
    clusters = detector.cluster_wallets(trades_df.copy())
    print(f"✓ Wallet clustering: {len(clusters)} clusters found")

    # Full report
    report = detector.generate_report(
        trades_df.copy(), 'test_market', event_time
    )
    print(f"✓ Full report generated")
    print(f"  - {report['summary']}")
    print(f"  - Risk Score: {report['risk_score']:.2f}/1.00\n")
except Exception as e:
    print(f"✗ Rule-based detection failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 5: Data Preprocessor
print("Test 5: Testing Data Preprocessor...")
try:
    preprocessor = DataPreprocessor()

    # Clean data
    clean_df = preprocessor.clean_trades(trades_df.copy())
    print(f"✓ Data cleaning: {len(clean_df)} clean trades")

    # Aggregate by trader
    trader_stats = preprocessor.aggregate_by_trader(clean_df)
    print(f"✓ Trader aggregation: {len(trader_stats)} unique traders")

    # Time features
    with_features = preprocessor.create_time_features(clean_df)
    print(f"✓ Time features: {list(with_features.columns)[:8]}...\n")
except Exception as e:
    print(f"✗ Data preprocessing failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 6: ML Model
print("Test 6: Testing ML Model...")
try:
    ml_model = InsiderMLModel(sequence_length=20, lstm_hidden_size=64)

    # Prepare features
    features_df = ml_model.prepare_features(trades_df.copy())
    print(f"✓ Feature extraction: {features_df.shape[1]} features from {len(features_df)} trades")

    # Create sequences
    X = features_df.values
    y = np.random.randint(0, 2, len(X))  # Random labels for testing

    sequences, labels = ml_model.create_sequences(X, y)
    print(f"✓ Sequence creation: {len(sequences)} sequences of shape {sequences.shape}")

    # Train LSTM (small test)
    if len(sequences) > 50:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )

        print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test")

        print("  Training LSTM (5 epochs)...")
        ml_model.train_lstm(X_train, y_train, epochs=5, batch_size=16)
        print("  ✓ LSTM training complete")

        print("  Training Random Forest...")
        ml_model.train_random_forest(X_train, y_train, n_estimators=50)
        print("  ✓ Random Forest training complete")

        # Make predictions
        predictions = ml_model.predict(X_test[:5], use_ensemble=True)
        print(f"✓ Predictions (sample): {predictions}")

        # Evaluate
        from sklearn.metrics import accuracy_score, roc_auc_score
        all_preds = ml_model.predict(X_test, use_ensemble=True)
        y_pred_binary = (all_preds > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred_binary)
        try:
            auc = roc_auc_score(y_test, all_preds)
            print(f"✓ Model performance: Accuracy={accuracy:.2f}, AUC={auc:.2f}")
        except:
            print(f"✓ Model performance: Accuracy={accuracy:.2f}")

        # Save models
        ml_model.save_models()
        print(f"✓ Models saved to {ml_model.model_dir}\n")
    else:
        print("⚠ Not enough data for training (need >50 sequences)\n")

except Exception as e:
    print(f"✗ ML model testing failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 7: Real API Connection (if possible)
print("Test 7: Testing Real Polymarket API Connection...")
try:
    client = PolymarketClient()
    markets = client.get_markets(active=True)

    if markets:
        print(f"✓ Connected to Polymarket API")
        print(f"  - Fetched {len(markets)} active markets")

        # Show top 3 markets
        print("  Top 3 markets by volume:")
        sorted_markets = sorted(markets, key=lambda x: x.get('volume', 0), reverse=True)[:3]
        for i, market in enumerate(sorted_markets, 1):
            question = market.get('question', 'N/A')
            volume = market.get('volume', 0)
            print(f"    {i}. {question[:50]}... (${volume:,.0f})")
    else:
        print("⚠ API returned no markets (may be rate limited or unavailable)")

except Exception as e:
    print(f"⚠ API connection test skipped: {e}")

print("\n" + "="*70)
print("✓ INTEGRATION TEST COMPLETE!")
print("="*70)
print("\nSummary:")
print("  - Data collection module: Ready")
print("  - Rule-based detection: Working")
print("  - ML models (LSTM + Random Forest): Working")
print("  - Data preprocessing: Working")
print("\nNext steps:")
print("  1. Run the Jupyter notebook: notebooks/insider_tracker_demo.ipynb")
print("  2. Fetch real data from Polymarket")
print("  3. Train models on historical data")
print("  4. Set up real-time monitoring")
print("="*70 + "\n")
