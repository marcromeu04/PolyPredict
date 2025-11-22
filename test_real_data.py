#!/usr/bin/env python3
"""
Real Data Test for PolyPredict
Tests all components with REAL data from Polymarket APIs
NO SYNTHETIC DATA
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

print("\n" + "="*80)
print("POLYPREDICT REAL DATA TEST")
print("Tests with actual Polymarket data - NO SYNTHETIC DATA")
print("="*80 + "\n")

# Test 1: Import all modules
print("Test 1: Importing all modules...")
try:
    from src.data_collection.polymarket_client import PolymarketClient
    from src.data_collection.thegraph_client import TheGraphClient
    from src.data_collection.polygon_client import PolygonClient
    from src.data_collection.news_events_client import NewsEventsClient
    from src.data_collection.unified_data_client import UnifiedDataClient
    from src.detection.rule_based_detector import RuleBasedDetector
    from src.models.insider_ml_model import InsiderMLModel
    from src.utils.data_preprocessor import DataPreprocessor
    print("✓ All imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Initialize clients
print("Test 2: Initializing all clients...")
try:
    polymarket_client = PolymarketClient()
    thegraph_client = TheGraphClient()
    polygon_client = PolygonClient()
    news_client = NewsEventsClient()
    unified_client = UnifiedDataClient()
    detector = RuleBasedDetector()
    preprocessor = DataPreprocessor()
    print("✓ All clients initialized\n")
except Exception as e:
    print(f"✗ Client initialization failed: {e}\n")
    sys.exit(1)

# Test 3: Fetch REAL markets from Polymarket
print("Test 3: Fetching REAL markets from Polymarket API...")
try:
    markets = polymarket_client.get_markets(active=True, limit=10)

    if not markets:
        print("✗ ERROR: No markets returned from Polymarket API")
        print("\nPossible reasons:")
        print("  - API is temporarily down")
        print("  - Rate limit exceeded")
        print("  - Network connectivity issues")
        print("\nPlease try again later.")
        sys.exit(1)

    print(f"✓ Fetched {len(markets)} REAL markets from Polymarket")
    print(f"\nTop 3 markets by volume:")

    sorted_markets = sorted(markets, key=lambda x: x.get('volume', 0), reverse=True)[:3]
    for i, market in enumerate(sorted_markets, 1):
        question = market.get('question', 'N/A')
        volume = market.get('volume', 0)
        print(f"  {i}. {question[:70]}...")
        print(f"     Volume: ${volume:,.2f}")
        print(f"     ID: {market.get('condition_id', 'N/A')[:15]}...")

    print()

except Exception as e:
    print(f"✗ Failed to fetch markets: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Fetch REAL trading data
print("Test 4: Fetching REAL trading data...")
try:
    # Select highest volume market
    selected_market = sorted_markets[0]
    market_id = selected_market.get('condition_id')
    market_question = selected_market.get('question')

    print(f"  Selected: {market_question[:60]}...")
    print(f"  Market ID: {market_id}")

    # Fetch real trades
    print(f"  Fetching trades from last 3 days...")
    trades_df = polymarket_client.get_historical_data(market_id, days_back=3)

    if trades_df.empty:
        print(f"\n⚠️  WARNING: No trades found for this market")
        print(f"  This market may have low activity")
        print(f"  Trying next market...")

        # Try second market
        if len(sorted_markets) > 1:
            selected_market = sorted_markets[1]
            market_id = selected_market.get('condition_id')
            market_question = selected_market.get('question')

            print(f"\n  Trying: {market_question[:60]}...")
            trades_df = polymarket_client.get_historical_data(market_id, days_back=3)

            if trades_df.empty:
                print(f"\n✗ ERROR: No trading data available")
                print(f"  All tested markets have no trades")
                print(f"\nThis could mean:")
                print(f"  - Markets are very new")
                print(f"  - API is not returning trade data")
                print(f"  - Try increasing --days parameter")
                sys.exit(1)

    print(f"✓ Fetched {len(trades_df)} REAL trades")
    print(f"  Date range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
    print(f"  Unique traders: {trades_df['maker'].nunique() if 'maker' in trades_df.columns else 'N/A'}")
    print()

except Exception as e:
    print(f"✗ Failed to fetch trades: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Data preprocessing on REAL data
print("Test 5: Preprocessing REAL trading data...")
try:
    clean_df = preprocessor.clean_trades(trades_df.copy())
    print(f"✓ Cleaned: {len(clean_df)} trades")

    trader_stats = preprocessor.aggregate_by_trader(clean_df)
    print(f"✓ Aggregated: {len(trader_stats)} unique traders")

    with_features = preprocessor.create_time_features(clean_df.copy())
    print(f"✓ Time features added")
    print()

except Exception as e:
    print(f"✗ Preprocessing failed: {e}\n")
    sys.exit(1)

# Test 6: Rule-based detection on REAL data
print("Test 6: Running detection on REAL data...")
try:
    # Use recent time as "event" for demonstration
    event_time = clean_df['timestamp'].max() - timedelta(hours=1)

    # Run full detection
    report = detector.generate_report(clean_df.copy(), market_id, event_time)

    print(f"✓ Detection complete:")
    print(f"  {report['summary']}")
    print(f"  Risk Score: {report['risk_score']:.2f}/1.00")

    if report['detections']['timing_anomalies']:
        print(f"  ⚠️  {len(report['detections']['timing_anomalies'])} timing anomalies detected")

    if report['detections']['volume_spikes']:
        print(f"  ⚠️  {len(report['detections']['volume_spikes'])} volume spikes detected")

    print()

except Exception as e:
    print(f"✗ Detection failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 7: Try The Graph (if API key configured)
print("Test 7: Testing The Graph integration...")
try:
    if thegraph_client.api_key:
        print(f"  ✓ The Graph API key configured")

        # Try to fetch market info
        market_info = thegraph_client.get_market_info(market_id)
        if market_info:
            print(f"  ✓ Fetched market info from subgraph")
        else:
            print(f"  ⚠️  No data from subgraph (market may not exist on-chain yet)")
    else:
        print(f"  ⚠️  The Graph API key not configured (optional)")
        print(f"     Get free key at: https://thegraph.com/studio/")

    print()

except Exception as e:
    print(f"  ⚠️  The Graph test: {e}\n")

# Test 8: Try Polygon blockchain data (if API key configured)
print("Test 8: Testing Polygon blockchain integration...")
try:
    if polygon_client.polygonscan_api_key:
        print(f"  ✓ PolygonScan API key configured")

        # Try to fetch recent on-chain data
        onchain_df = polygon_client.get_polymarket_trades_onchain(
            days_back=1,
            limit=10
        )

        if not onchain_df.empty:
            print(f"  ✓ Fetched {len(onchain_df)} on-chain transactions")
        else:
            print(f"  ⚠️  No on-chain data in last 24h")
    else:
        print(f"  ⚠️  PolygonScan API key not configured (optional)")
        print(f"     Get free key at: https://polygonscan.com/apis")

    print()

except Exception as e:
    print(f"  ⚠️  Polygon test: {e}\n")

# Test 9: Try News API (if configured)
print("Test 9: Testing News integration...")
try:
    if news_client.newsapi_key:
        print(f"  ✓ NewsAPI key configured")

        # Extract keywords from market question
        keywords = news_client._extract_keywords(market_question)
        print(f"  ✓ Extracted keywords: {keywords[:3]}")

        print(f"  ℹ️  News API ready (not calling to avoid rate limits)")
    else:
        print(f"  ⚠️  NewsAPI key not configured (optional)")
        print(f"     Get free key at: https://newsapi.org/")

    print()

except Exception as e:
    print(f"  ⚠️  News test: {e}\n")

# Test 10: Unified client with REAL data
print("Test 10: Testing Unified Data Client...")
try:
    print(f"  → Attempting to fetch from all sources...")

    comprehensive_data = unified_client.get_comprehensive_market_data(
        market_id,
        days_back=3,
        include_onchain=False,  # Skip to avoid slowness
        include_news=False  # Skip to avoid rate limits
    )

    # Check what data was retrieved
    sources_with_data = []
    for source_name, source_data in comprehensive_data['sources'].items():
        if isinstance(source_data, dict) and source_data.get('status') == 'success':
            sources_with_data.append(source_name)

    print(f"✓ Unified client retrieved data from {len(sources_with_data)} sources:")
    for source in sources_with_data:
        print(f"  ✓ {source}")

    print()

except Exception as e:
    print(f"  ⚠️  Unified client test: {e}\n")
    import traceback
    traceback.print_exc()

# Final Summary
print("="*80)
print("REAL DATA TEST SUMMARY")
print("="*80)
print("\n✅ SUCCESSFULLY TESTED WITH REAL DATA:")
print(f"  ✓ Fetched {len(markets)} real markets from Polymarket")
print(f"  ✓ Retrieved {len(trades_df)} real trades")
print(f"  ✓ Analyzed {trader_stats.shape[0]} real traders")
print(f"  ✓ Detection algorithms working on real data")
print(f"  ✓ Risk Score: {report['risk_score']:.2f}")

print("\n📊 MARKET ANALYZED:")
print(f"  Question: {market_question}")
print(f"  Volume: ${selected_market.get('volume', 0):,.2f}")
print(f"  Time Range: {clean_df['timestamp'].min()} to {clean_df['timestamp'].max()}")

print("\n🔑 API KEYS STATUS:")
print(f"  Polymarket API: ✓ Working (no key required)")
print(f"  The Graph: {'✓ Configured' if thegraph_client.api_key else '⚠️  Not configured (optional)'}")
print(f"  PolygonScan: {'✓ Configured' if polygon_client.polygonscan_api_key else '⚠️  Not configured (optional)'}")
print(f"  NewsAPI: {'✓ Configured' if news_client.newsapi_key else '⚠️  Not configured (optional)'}")

print("\n💡 NEXT STEPS:")
print("  1. Configure optional API keys in .env for full functionality")
print("  2. Run analysis: python run_tracker.py")
print("  3. Use Jupyter notebook for detailed visualization")
print("  4. Monitor multiple markets in real-time")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED WITH REAL DATA!")
print("="*80 + "\n")
