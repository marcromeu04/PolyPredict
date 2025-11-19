# PolyPredict API Integrations

## Overview

PolyPredict integrates with **5 major data sources** to provide comprehensive insider trading detection:

1. **Polymarket Official APIs** - Real-time trading data
2. **The Graph Subgraph** - On-chain indexed data via GraphQL
3. **Polygon Blockchain** - Direct blockchain access
4. **News APIs** - Event detection and timing analysis
5. **Unified Client** - Aggregates all sources

---

## 1. Polymarket Official APIs

### Endpoints

- **CLOB API**: `https://clob.polymarket.com`
- **Gamma API**: `https://gamma-api.polymarket.com`
- **Data API**: `https://data-api.polymarket.com`

### Features

- ✅ Market information
- ✅ Historical trades
- ✅ Order book data
- ✅ User positions
- ✅ WebSocket streaming (real-time)

### Setup

```python
from src.data_collection.polymarket_client import PolymarketClient

client = PolymarketClient()

# Get markets
markets = client.get_markets(active=True)

# Get historical trades
trades = client.get_historical_data(market_id, days_back=7)

# Get user trades
user_trades = client.get_user_trades(user_address)
```

### Rate Limits

- Free: ~1000 requests/hour
- Premium: Higher limits (requires API key)

### API Key (Optional)

Set in `.env`:
```bash
POLYMARKET_API_KEY=your_api_key_here
```

---

## 2. The Graph Subgraph

### Overview

The Graph provides a GraphQL API for querying Polymarket's on-chain data.

- **Endpoint**: `https://gateway.thegraph.com/api/{api-key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp`
- **Free Tier**: 100,000 queries/month
- **Documentation**: https://thegraph.com/docs/en/subgraphs/guides/polymarket/

### Features

- ✅ On-chain trade history
- ✅ User positions and liquidity
- ✅ Market volume statistics
- ✅ Top traders analysis
- ✅ Historical volume data

### Setup

```python
from src.data_collection.thegraph_client import TheGraphClient

client = TheGraphClient()

# Get market info from subgraph
market_info = client.get_market_info(market_id)

# Get trades
trades = client.get_trades(market_id, first=1000)

# Get user positions
positions = client.get_user_positions(user_address)

# Get volume history
volume_df = client.get_market_volume_history(market_id, days=30)
```

### GraphQL Queries

Example query:
```graphql
query GetMarket($marketId: ID!) {
    fixedProductMarketMaker(id: $marketId) {
        id
        collateralVolume
        usdVolume
        fpmmTrades(first: 100) {
            creationTimestamp
            collateralAmount
            creator {
                id
            }
        }
    }
}
```

### API Key

Get free API key at https://thegraph.com/studio/

Set in `.env`:
```bash
THEGRAPH_API_KEY=your_api_key_here
```

---

## 3. Polygon Blockchain

### Overview

Direct access to Polymarket smart contracts on Polygon blockchain.

### Contract Addresses

- **CTF Exchange**: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- **Conditional Tokens**: `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`
- **UMA CTF Adapter**: `0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74`

### Features

- ✅ Raw transaction data
- ✅ Event logs
- ✅ Wallet activity tracking
- ✅ Gas usage analysis
- ✅ Token transfers

### Setup

```python
from src.data_collection.polygon_client import PolygonClient

client = PolygonClient()

# Get on-chain transactions
onchain_df = client.get_polymarket_trades_onchain(days_back=7)

# Get user activity
activity = client.get_user_onchain_activity(user_address, days_back=30)

# Get wallet balance
balance = client.get_wallet_balance(address)
```

### PolygonScan API

Get free API key at: https://polygonscan.com/apis

Set in `.env`:
```bash
POLYGONSCAN_API_KEY=your_api_key_here
```

### RPC Providers

Options for `POLYGON_RPC_URL`:
- **Free**: `https://polygon-rpc.com`
- **Alchemy**: `https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY`
- **Infura**: `https://polygon-mainnet.infura.io/v3/YOUR_KEY`
- **QuickNode**: `https://your-endpoint.matic.quiknode.pro/YOUR_KEY/`

---

## 4. News & Events API

### Overview

Fetches news articles and events related to prediction markets for detecting insider trading timing.

### Supported Sources

- **NewsAPI**: General news articles
- **Twitter/X API**: Social media mentions
- **RSS Feeds**: Custom sources

### Features

- ✅ Market-related news fetching
- ✅ Event detection
- ✅ Timing correlation with trades
- ✅ Major event identification

### Setup

```python
from src.data_collection.news_events_client import NewsEventsClient, EventDetector

news_client = NewsEventsClient()

# Get news for market
news_df = news_client.get_market_related_news(market_question, days_back=7)

# Detect timing anomalies
suspicious = news_client.detect_announcement_timing(
    market_question,
    trades_df,
    news_df
)

# Detect major events
event_detector = EventDetector()
major_events = event_detector.detect_major_events(market_question, days_back=30)
```

### NewsAPI

- **Free Tier**: 100 requests/day
- **Get API Key**: https://newsapi.org/

Set in `.env`:
```bash
NEWSAPI_KEY=your_newsapi_key
```

### Twitter/X API

- **Get API Key**: https://developer.twitter.com/

Set in `.env`:
```bash
TWITTER_BEARER_TOKEN=your_twitter_token
```

---

## 5. Unified Data Client

### Overview

The **Unified Data Client** aggregates data from all sources into a single interface.

### Features

- ✅ Fetches from all APIs simultaneously
- ✅ Merges and deduplicates data
- ✅ Cross-source validation
- ✅ Anomaly detection across sources

### Setup

```python
from src.data_collection.unified_data_client import UnifiedDataClient

client = UnifiedDataClient()

# Get comprehensive data from ALL sources
comprehensive_data = client.get_comprehensive_market_data(
    market_id,
    days_back=7,
    include_onchain=True,
    include_news=True
)

# Access data by source
polymarket_data = comprehensive_data['sources']['polymarket']
thegraph_data = comprehensive_data['sources']['thegraph']
polygon_data = comprehensive_data['sources']['polygon']
news_data = comprehensive_data['sources']['news']

# Get merged trade data
merged_trades_df = client.merge_trade_data(market_id, days_back=7)

# Detect cross-source anomalies
anomalies = client.detect_cross_source_anomalies(market_id, days_back=7)
```

### Trader Profiling

```python
# Comprehensive trader profile from all sources
profile = client.get_trader_comprehensive_profile(
    trader_address,
    days_back=30
)

print(f"Trades: {profile['sources']['polymarket']['num_trades']}")
print(f"On-chain txs: {profile['sources']['polygon']['activity']}")
print(f"Balance: {profile['sources']['polygon']['balance_matic']} MATIC")
```

---

## Data Source Comparison

| Feature | Polymarket API | The Graph | Polygon | News |
|---------|---------------|-----------|---------|------|
| **Real-time** | ✅ WebSocket | ❌ | ❌ | ✅ |
| **Historical** | ✅ Limited | ✅ Full | ✅ Full | ✅ |
| **Free Tier** | ✅ 1k/hour | ✅ 100k/month | ✅ Limited | ✅ 100/day |
| **Requires Key** | ❌ Optional | ❌ Optional | ✅ Recommended | ✅ Yes |
| **Data Latency** | ~Seconds | ~Minutes | ~Blocks | ~Hours |
| **Best For** | Current data | Analytics | Verification | Events |

---

## Best Practices

### 1. Use Multiple Sources

Always fetch from multiple sources to validate data:

```python
unified_client = UnifiedDataClient()
data = unified_client.get_comprehensive_market_data(market_id)

# Compare trade counts across sources
pm_trades = data['sources']['polymarket']['num_trades']
tg_trades = data['sources']['thegraph']['num_trades']

if abs(pm_trades - tg_trades) / max(pm_trades, tg_trades) > 0.2:
    print("⚠️  Warning: Large discrepancy between sources!")
```

### 2. Handle Rate Limits

```python
import time

try:
    trades = client.get_trades(market_id)
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        print("Rate limited, waiting...")
        time.sleep(60)
        trades = client.get_trades(market_id)
```

### 3. Cache Data

```python
import pickle
from pathlib import Path

cache_file = Path("data/cache/market_data.pkl")

if cache_file.exists():
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
else:
    data = client.get_comprehensive_market_data(market_id)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
```

### 4. Error Handling

Always handle API errors gracefully:

```python
try:
    trades = client.get_trades(market_id)
    if not trades:
        print("No trades found, trying alternate source...")
        trades_df = thegraph_client.get_trades(market_id)
except Exception as e:
    logger.error(f"Error fetching trades: {e}")
    trades = []
```

---

## Complete Integration Example

```python
from src.data_collection.unified_data_client import UnifiedDataClient
from src.detection.rule_based_detector import RuleBasedDetector
from datetime import datetime, timedelta

# Initialize clients
unified_client = UnifiedDataClient()
detector = RuleBasedDetector()

# Get comprehensive data
market_id = "your_market_id_here"
data = unified_client.get_comprehensive_market_data(
    market_id,
    days_back=7,
    include_onchain=True,
    include_news=True
)

# Merge trades from all sources
trades_df = unified_client.merge_trade_data(market_id, days_back=7)

# Run detection
event_time = datetime.now() - timedelta(hours=2)
report = detector.generate_report(trades_df, market_id, event_time)

# Correlate with news
if 'news' in data['sources'] and not data['sources']['news']['articles'].empty:
    news_df = data['sources']['news']['articles']
    news_client = unified_client.news

    timing_anomalies = news_client.detect_announcement_timing(
        data['market_question'],
        trades_df,
        news_df
    )

    print(f"Found {len(timing_anomalies)} suspicious trading events before news")

# Cross-source validation
anomalies = unified_client.detect_cross_source_anomalies(market_id, days_back=7)
print(f"Cross-source anomalies: {anomalies}")
```

---

## Troubleshooting

### No data returned

1. Check API keys are set in `.env`
2. Verify market ID is correct
3. Check rate limits
4. Try alternative data source

### Slow performance

1. Reduce `days_back` parameter
2. Disable on-chain fetching: `include_onchain=False`
3. Use pagination
4. Cache results

### Inconsistent data between sources

This is normal! Reasons:
- Different indexing times
- API vs on-chain lag
- Different aggregation methods

Always use `detect_cross_source_anomalies()` to identify issues.

---

## API Keys Summary

Required `.env` configuration:

```bash
# Optional but recommended
THEGRAPH_API_KEY=          # Free: 100k queries/month
POLYGONSCAN_API_KEY=       # Free tier available
NEWSAPI_KEY=               # Free: 100 requests/day

# Optional
POLYMARKET_API_KEY=        # Only for premium features
TWITTER_BEARER_TOKEN=      # For social media monitoring
```

Get API keys:
- **The Graph**: https://thegraph.com/studio/
- **PolygonScan**: https://polygonscan.com/apis
- **NewsAPI**: https://newsapi.org/
- **Twitter**: https://developer.twitter.com/
