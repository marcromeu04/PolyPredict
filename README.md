# PolyPredict - Advanced Insider Trading Detection for Polymarket

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**PolyPredict** is a comprehensive insider trading detection system for Polymarket prediction markets. It combines rule-based heuristics with state-of-the-art machine learning models and integrates **5 major data sources** to identify suspicious trading patterns in real-time.

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

## 📡 Data Sources (5 Integrated APIs)

PolyPredict aggregates data from multiple sources for comprehensive analysis:

1. **Polymarket Official APIs**
   - CLOB API - Real-time order book & trades
   - Gamma API - Market information
   - Data API - User positions
   - WebSocket - Live streaming

2. **The Graph Subgraph**
   - GraphQL API for on-chain indexed data
   - Historical trades & volume
   - Liquidity tracking
   - 100k free queries/month

3. **Polygon Blockchain**
   - Direct on-chain data via PolygonScan API
   - Contract interaction analysis
   - Transaction verification
   - Gas usage patterns

4. **News & Events APIs**
   - NewsAPI for market-related news
   - Twitter/X API for social sentiment
   - Event timing correlation
   - Insider trading timing detection

5. **Unified Data Client**
   - Aggregates all sources
   - Cross-source validation
   - Automatic deduplication
   - Anomaly detection across sources

See [docs/API_INTEGRATIONS.md](docs/API_INTEGRATIONS.md) for detailed API documentation.

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
