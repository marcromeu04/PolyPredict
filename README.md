# PolyPredict - Advanced Insider Trading Detection for Polymarket

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**PolyPredict** is a comprehensive insider trading detection system for Polymarket prediction markets. It combines rule-based heuristics with state-of-the-art machine learning models and integrates **5 major data sources** to identify suspicious trading patterns in real-time.

> ⚠️ **REAL DATA ONLY**: PolyPredict works exclusively with real data from Polymarket and other APIs. No synthetic or mock data is used. See [docs/GETTING_REAL_DATA.md](docs/GETTING_REAL_DATA.md) for API configuration.

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

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/PolyPredict.git
cd PolyPredict

# Install required packages
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional but Recommended)

PolyPredict works **without any API keys** for basic functionality (Polymarket API only), but adding API keys significantly improves data quality:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

**Recommended APIs** (all have free tiers):
- **The Graph**: 100k queries/month free - Get at [thegraph.com/studio](https://thegraph.com/studio/)
- **PolygonScan**: Free tier - Get at [polygonscan.com/apis](https://polygonscan.com/apis)
- **NewsAPI**: 100 requests/day free - Get at [newsapi.org](https://newsapi.org/)

See [docs/GETTING_REAL_DATA.md](docs/GETTING_REAL_DATA.md) for detailed API setup guide.

### 3. Verify Real Data Access

```bash
# Test that real data is accessible
python test_real_data.py
```

This will verify connectivity to Polymarket and show which optional APIs are configured.

## Usage

### Quick Start (Command Line)

```bash
# Analyze the highest-volume market automatically
python run_tracker.py

# Analyze a specific market
python run_tracker.py --market-id <market_id>

# Get more historical data (14 days instead of 7)
python run_tracker.py --days 14
```

### Interactive Analysis (Jupyter Notebook)

```bash
# Launch the interactive notebook
jupyter notebook notebooks/insider_tracker_demo.ipynb
```

The notebook includes:
- Real-time data fetching from all 5 sources
- Interactive visualizations
- Step-by-step detection analysis
- Comprehensive dashboard
- Result export functionality

**Important**: The notebook will error if no real data is available for the selected market. This is by design - select a market with active trading.

### Example Output

```
Fetching comprehensive data for market 0x123abc...
  → Fetching from Polymarket API...
  → Fetching trade history from Polymarket...
  → Fetching from The Graph subgraph...
  → Fetching on-chain data from Polygon...
  → Fetching news and events...
✓ Comprehensive data fetch complete

Found 1,247 trades from 89 unique traders
Detected 3 high-risk patterns:
  - Timing anomaly: Large trade 8 min before event
  - Coordinated trading: 4 wallets with similar patterns
  - Volume spike: 300% above baseline
```

See [docs/GETTING_REAL_DATA.md](docs/GETTING_REAL_DATA.md) for troubleshooting if you encounter "No data found" errors.

## State of the Art

Based on research from:
- Chainalysis wallet clustering techniques
- EPJ Data Science ML insider detection (2024)
- LSTM-based anomaly detection methods
- Random Forest classification for trading patterns

## License

MIT
