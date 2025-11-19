"""
Unified Data Client
Aggregates data from all sources: Polymarket APIs, The Graph, Polygon, News
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .polymarket_client import PolymarketClient
from .thegraph_client import TheGraphClient
from .polygon_client import PolygonClient
from .news_events_client import NewsEventsClient, EventDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedDataClient:
    """
    Unified client that aggregates data from multiple sources for comprehensive analysis

    Data Sources:
    1. Polymarket APIs (CLOB, Gamma, Data) - Official API data
    2. The Graph Subgraph - On-chain indexed data via GraphQL
    3. Polygon Blockchain - Direct on-chain data
    4. News & Events - External event detection
    """

    def __init__(self):
        """Initialize all data clients"""
        logger.info("Initializing Unified Data Client...")

        # Initialize all clients
        self.polymarket = PolymarketClient()
        self.thegraph = TheGraphClient()
        self.polygon = PolygonClient()
        self.news = NewsEventsClient()
        self.event_detector = EventDetector()

        logger.info("✓ All data clients initialized")

    def get_comprehensive_market_data(
        self,
        market_id: str,
        days_back: int = 7,
        include_onchain: bool = True,
        include_news: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive data for a market from all sources

        Args:
            market_id: Market/condition ID
            days_back: Days of historical data
            include_onchain: Whether to fetch on-chain data
            include_news: Whether to fetch news data

        Returns:
            Dictionary with data from all sources
        """
        logger.info(f"Fetching comprehensive data for market {market_id}")

        result = {
            'market_id': market_id,
            'fetched_at': datetime.now(),
            'sources': {}
        }

        # 1. Get market info from Polymarket API
        logger.info("  → Fetching from Polymarket API...")
        try:
            market_info = self.polymarket.get_market_by_id(market_id)
            result['sources']['polymarket'] = {
                'market_info': market_info,
                'status': 'success' if market_info else 'no_data'
            }
            result['market_question'] = market_info.get('question') if market_info else None
        except Exception as e:
            logger.error(f"Error fetching from Polymarket: {e}")
            result['sources']['polymarket'] = {'status': 'error', 'error': str(e)}

        # 2. Get historical trades from Polymarket
        logger.info("  → Fetching trade history from Polymarket...")
        try:
            trades_df = self.polymarket.get_historical_data(market_id, days_back)
            result['sources']['polymarket']['trades'] = trades_df
            result['sources']['polymarket']['num_trades'] = len(trades_df)
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            result['sources']['polymarket']['trades'] = pd.DataFrame()

        # 3. Get data from The Graph subgraph
        logger.info("  → Fetching from The Graph subgraph...")
        try:
            subgraph_market = self.thegraph.get_market_info(market_id)
            subgraph_trades = self.thegraph.get_trades(market_id, first=1000)
            liquidity = self.thegraph.get_market_liquidity(market_id)

            result['sources']['thegraph'] = {
                'market_info': subgraph_market,
                'trades': subgraph_trades,
                'liquidity': liquidity,
                'num_trades': len(subgraph_trades) if subgraph_trades else 0,
                'status': 'success' if subgraph_market or subgraph_trades else 'no_data'
            }
        except Exception as e:
            logger.error(f"Error fetching from The Graph: {e}")
            result['sources']['thegraph'] = {'status': 'error', 'error': str(e)}

        # 4. Get on-chain data from Polygon (optional, can be slow)
        if include_onchain:
            logger.info("  → Fetching on-chain data from Polygon...")
            try:
                onchain_df = self.polygon.get_polymarket_trades_onchain(
                    days_back=min(days_back, 3),  # Limit to avoid slowness
                    limit=500
                )

                result['sources']['polygon'] = {
                    'transactions': onchain_df,
                    'num_transactions': len(onchain_df),
                    'status': 'success' if not onchain_df.empty else 'no_data'
                }
            except Exception as e:
                logger.error(f"Error fetching on-chain data: {e}")
                result['sources']['polygon'] = {'status': 'error', 'error': str(e)}

        # 5. Get news and events (if market question available)
        if include_news and result.get('market_question'):
            logger.info("  → Fetching news and events...")
            try:
                news_df = self.news.get_market_related_news(
                    result['market_question'],
                    days_back
                )

                major_events = self.event_detector.detect_major_events(
                    result['market_question'],
                    days_back
                )

                result['sources']['news'] = {
                    'articles': news_df,
                    'num_articles': len(news_df),
                    'major_events': major_events,
                    'num_major_events': len(major_events),
                    'status': 'success' if not news_df.empty else 'no_data'
                }
            except Exception as e:
                logger.error(f"Error fetching news: {e}")
                result['sources']['news'] = {'status': 'error', 'error': str(e)}

        logger.info("✓ Comprehensive data fetch complete")

        return result

    def get_trader_comprehensive_profile(
        self,
        trader_address: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get comprehensive profile for a trader from all sources

        Args:
            trader_address: Ethereum address
            days_back: Days to analyze

        Returns:
            Complete trader profile
        """
        logger.info(f"Building comprehensive profile for trader {trader_address}")

        profile = {
            'address': trader_address,
            'analyzed_at': datetime.now(),
            'sources': {}
        }

        # 1. Polymarket API data
        logger.info("  → Fetching from Polymarket API...")
        try:
            trades = self.polymarket.get_user_trades(trader_address, limit=1000)
            positions = self.polymarket.get_user_positions(trader_address)

            profile['sources']['polymarket'] = {
                'trades': trades,
                'positions': positions,
                'num_trades': len(trades),
                'num_positions': len(positions),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error fetching from Polymarket: {e}")
            profile['sources']['polymarket'] = {'status': 'error', 'error': str(e)}

        # 2. The Graph subgraph data
        logger.info("  → Fetching from The Graph...")
        try:
            subgraph_positions = self.thegraph.get_user_positions(trader_address)

            profile['sources']['thegraph'] = {
                'positions': subgraph_positions,
                'status': 'success' if subgraph_positions else 'no_data'
            }
        except Exception as e:
            logger.error(f"Error fetching from The Graph: {e}")
            profile['sources']['thegraph'] = {'status': 'error', 'error': str(e)}

        # 3. On-chain activity from Polygon
        logger.info("  → Fetching on-chain activity...")
        try:
            onchain_activity = self.polygon.get_user_onchain_activity(
                trader_address,
                days_back
            )

            balance = self.polygon.get_wallet_balance(trader_address)

            profile['sources']['polygon'] = {
                'activity': onchain_activity,
                'balance_matic': balance,
                'status': 'success' if onchain_activity else 'no_data'
            }
        except Exception as e:
            logger.error(f"Error fetching on-chain activity: {e}")
            profile['sources']['polygon'] = {'status': 'error', 'error': str(e)}

        logger.info("✓ Trader profile complete")

        return profile

    def merge_trade_data(self, market_id: str, days_back: int = 7) -> pd.DataFrame:
        """
        Merge trade data from multiple sources into unified DataFrame

        Args:
            market_id: Market ID
            days_back: Days of data

        Returns:
            Unified DataFrame with trades from all sources
        """
        all_trades = []

        # Get trades from Polymarket API
        try:
            pm_trades = self.polymarket.get_historical_data(market_id, days_back)
            if not pm_trades.empty:
                pm_trades['source'] = 'polymarket_api'
                all_trades.append(pm_trades)
        except Exception as e:
            logger.warning(f"Could not fetch Polymarket trades: {e}")

        # Get trades from The Graph
        try:
            tg_trades = self.thegraph.get_trades(market_id, first=1000)
            if tg_trades:
                tg_df = pd.DataFrame(tg_trades)
                tg_df['source'] = 'thegraph'

                # Convert timestamp
                if 'creationTimestamp' in tg_df.columns:
                    tg_df['timestamp'] = pd.to_datetime(
                        tg_df['creationTimestamp'].astype(int),
                        unit='s'
                    )

                # Rename columns to match
                if 'creator' in tg_df.columns:
                    tg_df['maker'] = tg_df['creator'].apply(
                        lambda x: x['id'] if isinstance(x, dict) else x
                    )

                if 'collateralAmount' in tg_df.columns:
                    tg_df['size'] = pd.to_numeric(tg_df['collateralAmount'], errors='coerce')

                all_trades.append(tg_df)
        except Exception as e:
            logger.warning(f"Could not fetch The Graph trades: {e}")

        if not all_trades:
            logger.warning("No trades found from any source")
            return pd.DataFrame()

        # Merge all dataframes
        merged = pd.concat(all_trades, ignore_index=True, sort=False)

        # Remove duplicates (same trade from different sources)
        # Try to deduplicate by timestamp + maker + size
        if all(['timestamp', 'maker', 'size'] in merged.columns for col in ['timestamp', 'maker', 'size']):
            merged['dedup_key'] = (
                merged['timestamp'].astype(str) + '_' +
                merged['maker'].astype(str) + '_' +
                merged['size'].astype(str)
            )
            merged = merged.drop_duplicates(subset=['dedup_key'], keep='first')
            merged = merged.drop('dedup_key', axis=1)

        # Sort by timestamp
        if 'timestamp' in merged.columns:
            merged = merged.sort_values('timestamp', ascending=False)

        logger.info(f"Merged {len(merged)} unique trades from {merged['source'].nunique()} sources")

        return merged

    def detect_cross_source_anomalies(
        self,
        market_id: str,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Compare data across sources to detect anomalies and inconsistencies

        Args:
            market_id: Market ID
            days_back: Days to analyze

        Returns:
            Dictionary with detected anomalies
        """
        logger.info("Detecting cross-source anomalies...")

        anomalies = {
            'market_id': market_id,
            'detected': [],
            'warnings': []
        }

        # Get data from multiple sources
        data = self.get_comprehensive_market_data(
            market_id,
            days_back,
            include_onchain=True,
            include_news=False
        )

        # Compare trade counts
        pm_trades = data['sources'].get('polymarket', {}).get('num_trades', 0)
        tg_trades = data['sources'].get('thegraph', {}).get('num_trades', 0)

        if pm_trades > 0 and tg_trades > 0:
            discrepancy = abs(pm_trades - tg_trades) / max(pm_trades, tg_trades)

            if discrepancy > 0.2:  # More than 20% difference
                anomalies['warnings'].append({
                    'type': 'trade_count_discrepancy',
                    'polymarket_count': pm_trades,
                    'thegraph_count': tg_trades,
                    'discrepancy_pct': discrepancy * 100,
                    'severity': 'medium'
                })

        # Check for missing on-chain data
        polygon_txs = data['sources'].get('polygon', {}).get('num_transactions', 0)

        if polygon_txs == 0 and pm_trades > 10:
            anomalies['warnings'].append({
                'type': 'missing_onchain_data',
                'message': 'No on-chain transactions found despite API showing trades',
                'severity': 'low'
            })

        logger.info(f"Found {len(anomalies['detected'])} anomalies and {len(anomalies['warnings'])} warnings")

        return anomalies


if __name__ == "__main__":
    # Test the unified client
    client = UnifiedDataClient()

    print("\n" + "="*70)
    print("UNIFIED DATA CLIENT TEST")
    print("="*70 + "\n")

    # Note: This would need a real market ID to work
    print("Unified client initialized successfully")
    print("\nAvailable data sources:")
    print("  ✓ Polymarket APIs (CLOB, Gamma, Data)")
    print("  ✓ The Graph Subgraph")
    print("  ✓ Polygon Blockchain")
    print("  ✓ News & Events")
    print("\nTo use: client.get_comprehensive_market_data(market_id)")
