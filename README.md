# PolyPredict - Polymarket Insider Trading Tracker

## Overview

PolyPredict is an advanced insider trading detection system for Polymarket prediction markets. It combines rule-based detection with machine learning to identify suspicious trading patterns in real-time.

## Features

### 🔍 Detection Methods

1. **Rule-Based Detection**
   - Wallet clustering analysis
   - Timing anomaly detection
   - Volume spike identification
   - Multi-account pattern recognition

2. **Machine Learning Detection**
   - LSTM neural networks for time series analysis
   - Random Forest classification
   - Anomaly detection with dynamic thresholding
   - Pre-training on historical data

3. **Real-Time Monitoring**
   - Live data ingestion via Polymarket APIs
   - WebSocket streaming for instant updates
   - Alert system for suspicious activity

## Architecture

```
PolyPredict/
├── src/
│   ├── data_collection/     # Polymarket API integrations
│   ├── detection/           # Rule-based & ML detection engines
│   ├── models/              # Pre-trained ML models
│   └── utils/               # Helper functions
├── notebooks/               # Analysis & visualization notebooks
├── data/                    # Historical data & datasets
├── tests/                   # Unit & integration tests
└── config/                  # Configuration files
```

## Data Sources

- **Polymarket CLOB API** - Order book & trade data
- **Gamma API** - Market metadata
- **Data API** - Position tracking
- **Bitquery GraphQL** - On-chain analysis

## Detection Signals

### High-Risk Indicators
- Large position changes 5-30 min before major events
- Multiple wallets with coordinated trading patterns
- Unusual funding source similarities
- Directional trading with high confidence pre-event

### Medium-Risk Indicators
- Volume spikes outside normal patterns
- Price impact disproportionate to market depth
- Rapid position accumulation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See `notebooks/insider_tracker_demo.ipynb` for full examples.

## State of the Art

Based on research from:
- Chainalysis wallet clustering techniques
- EPJ Data Science ML insider detection (2024)
- LSTM-based anomaly detection methods
- Random Forest classification for trading patterns

## License

MIT
