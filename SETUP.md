# PolyPredict Setup Guide

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys (optional for basic usage):

```bash
# Polymarket API (optional)
POLYMARKET_API_KEY=your_api_key_here

# Ethereum RPC (optional, for on-chain analysis)
ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID

# Bitquery API (optional, for GraphQL data)
BITQUERY_API_KEY=your_bitquery_key_here
```

## Quick Start

### Run the Quick Test

```bash
python test_quick.py
```

This will test all components and create a sample analysis.

### Run the Tracker

Analyze the highest-volume market:

```bash
python run_tracker.py
```

Analyze a specific market:

```bash
python run_tracker.py --market-id MARKET_ID_HERE --days 14
```

### Use the Jupyter Notebook

The comprehensive notebook includes:
- Data collection from Polymarket
- Rule-based detection
- ML model training and evaluation
- Visualizations and dashboard

```bash
jupyter notebook notebooks/insider_tracker_demo.ipynb
```

## Architecture

### Data Collection (`src/data_collection/`)

- **polymarket_client.py**: Interfaces with Polymarket APIs
  - CLOB API for trade data
  - Gamma API for market info
  - Data API for positions
  - WebSocket support for real-time data

### Detection (`src/detection/`)

- **rule_based_detector.py**: Heuristic detection methods
  - Timing anomaly detection
  - Volume spike identification
  - Wallet clustering
  - Coordinated trading detection

### Models (`src/models/`)

- **insider_ml_model.py**: Machine learning models
  - LSTM neural network for time series
  - Random Forest classifier
  - Ensemble methods
  - Feature engineering pipeline

### Utilities (`src/utils/`)

- **data_preprocessor.py**: Data cleaning and transformation
  - Trade validation
  - Trader aggregation
  - Time feature extraction
  - Outlier detection

## Detection Methods

### Rule-Based Detection

Based on research from Chainalysis and academic papers:

1. **Timing Anomalies**
   - Detects large trades immediately before events
   - Configurable time window (default: 30 minutes)
   - Compares to normal trading patterns

2. **Volume Spikes**
   - Identifies unusual trading volume
   - Uses rolling statistics
   - Dynamic threshold adjustment

3. **Wallet Clustering**
   - Groups related wallets
   - Based on trading patterns, timing, and funding
   - Similarity threshold configurable

4. **Coordinated Trading**
   - Detects multiple traders acting in concert
   - Time-window based analysis
   - Directional consistency checks

### Machine Learning Detection

Advanced pattern recognition:

1. **LSTM Neural Network**
   - Processes time series sequences
   - Attention mechanism
   - Learns temporal patterns
   - Dynamic thresholding

2. **Random Forest**
   - Feature-based classification
   - Handles non-linear relationships
   - Provides feature importance

3. **Ensemble Methods**
   - Combines LSTM and Random Forest
   - Improved accuracy and robustness
   - Probability calibration

## Training ML Models

### 1. Collect Historical Data

```python
from src.data_collection.polymarket_client import PolymarketClient

client = PolymarketClient()
markets = client.get_markets(active=True)

# Get top market
market = markets[0]
trades = client.get_historical_data(market['condition_id'], days_back=30)
```

### 2. Label Data

Label suspicious trades based on:
- Known insider trading events
- Rule-based detections
- Manual verification

```python
# Create labels (0 = normal, 1 = suspicious)
labels = create_labels_from_events(trades, known_events)
```

### 3. Train Models

```python
from src.models.insider_ml_model import InsiderMLModel

model = InsiderMLModel()

# Prepare features
features = model.prepare_features(trades)

# Create sequences
sequences, seq_labels = model.create_sequences(features.values, labels)

# Train
model.train_lstm(sequences, seq_labels, epochs=50)
model.train_random_forest(sequences, seq_labels)

# Save
model.save_models()
```

### 4. Evaluate

```python
from sklearn.metrics import classification_report, roc_auc_score

predictions = model.predict(test_sequences)
print(classification_report(test_labels, predictions > 0.5))
print(f"ROC-AUC: {roc_auc_score(test_labels, predictions):.4f}")
```

## Real-Time Monitoring

### WebSocket Streaming

```python
import asyncio
from src.data_collection.polymarket_client import PolymarketClient

async def handle_trade(trade_data):
    # Process incoming trade
    print(f"New trade: {trade_data}")

    # Run detection
    # ...

client = PolymarketClient()
asyncio.run(client.stream_trades_async(market_id, handle_trade))
```

## Configuration

Edit `config/config.yaml` to customize:

- Detection thresholds
- ML model hyperparameters
- Data collection settings
- Alert configurations

## Outputs

### Detection Reports

JSON reports saved to `data/results/`:

```json
{
  "market_id": "0x123...",
  "analysis_time": "2024-01-01T12:00:00",
  "total_trades": 1000,
  "detections": {
    "timing_anomalies": [...],
    "volume_spikes": [...],
    "coordinated_trading": [...],
    "wallet_clusters": {...}
  },
  "risk_score": 0.75,
  "summary": "Detected 5 timing anomalies..."
}
```

### Trained Models

Models saved to `data/models/`:

- `lstm_model.pt` - PyTorch LSTM weights
- `rf_model.pkl` - Random Forest pickle
- `scaler.pkl` - Feature scaler

## API Rate Limits

Polymarket APIs have rate limits:

- Free tier: ~1000 requests/hour
- Premium: Higher limits available

The client implements:
- Automatic rate limiting
- Request caching (15-min TTL)
- Retry logic with backoff

## Troubleshooting

### No markets returned

- Check internet connection
- Verify API is accessible
- May be rate limited (wait and retry)

### Import errors

```bash
pip install -r requirements.txt
```

### CUDA/GPU errors

CPU-only mode:

```python
# In insider_ml_model.py
self.device = torch.device('cpu')
```

### Memory errors

Reduce batch size or sequence length:

```python
model = InsiderMLModel(
    sequence_length=20,  # Reduced from 50
    batch_size=16  # Reduced from 32
)
```

## Next Steps

1. **Collect more data**: Run data collection over longer periods
2. **Label training data**: Identify known insider trading cases
3. **Train models**: Use labeled data for supervised learning
4. **Deploy monitoring**: Set up real-time detection
5. **Create alerts**: Integrate with Slack/email/etc.
6. **Validate results**: Compare with known events

## References

- [Polymarket Documentation](https://docs.polymarket.com/)
- [Chainalysis Wallet Clustering](https://www.chainalysis.com/)
- EPJ Data Science - ML Insider Detection (2024)
- LSTM Anomaly Detection Methods

## Support

For issues and questions:
- Check the documentation
- Review example notebook
- Open an issue on GitHub
