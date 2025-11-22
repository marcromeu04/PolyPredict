#!/usr/bin/env python3
"""
PolyPredict - Main execution script
Runs the insider trading tracker on Polymarket data
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from src.data_collection.polymarket_client import PolymarketClient
from src.detection.rule_based_detector import RuleBasedDetector
from src.models.insider_ml_model import InsiderMLModel
from src.utils.data_preprocessor import DataPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_tracker(
    market_id: str = None,
    days_back: int = 7,
    use_ml: bool = True,
    save_results: bool = True
):
    """
    Run insider trading tracker

    Args:
        market_id: Specific market to analyze (None = highest volume)
        days_back: Number of days of historical data
        use_ml: Whether to use ML models
        save_results: Whether to save detection results
    """
    logger.info("="*70)
    logger.info("POLYPREDICT INSIDER TRADING TRACKER")
    logger.info("="*70)

    # Initialize components
    logger.info("\nInitializing components...")
    client = PolymarketClient()
    detector = RuleBasedDetector()
    preprocessor = DataPreprocessor()

    if use_ml:
        ml_model = InsiderMLModel()

    # Fetch markets
    logger.info("\nFetching Polymarket markets...")
    markets = client.get_markets(active=True)

    if not markets:
        logger.error("❌ ERROR: No markets found from Polymarket API")
        logger.error("\nPossible reasons:")
        logger.error("  1. API is currently unavailable")
        logger.error("  2. Rate limit exceeded")
        logger.error("  3. Network connectivity issues")
        logger.error("\nPlease try again later or check your internet connection.")
        sys.exit(1)

    logger.info(f"✓ Found {len(markets)} active markets")

    # Select market
    if market_id is None:
        # Get highest volume market
        selected_market = sorted(
            markets,
            key=lambda x: x.get('volume', 0),
            reverse=True
        )[0]
        market_id = selected_market.get('condition_id')
        market_question = selected_market.get('question', 'Unknown')
        market_volume = selected_market.get('volume', 0)
    else:
        # Try to get market info
        market_info = client.get_market_by_id(market_id)
        if market_info:
            market_question = market_info.get('question', market_id)
            market_volume = market_info.get('volume', 0)
        else:
            market_question = market_id
            market_volume = 0

    logger.info(f"\n📊 Selected Market:")
    logger.info(f"  Question: {market_question}")
    logger.info(f"  Market ID: {market_id}")
    logger.info(f"  Volume: ${market_volume:,.2f}")

    # Fetch trading data
    logger.info(f"\n📥 Fetching {days_back} days of trading data...")
    trades_df = client.get_historical_data(market_id, days_back)

    if trades_df.empty:
        logger.error(f"\n❌ ERROR: No trading data found for market {market_id}")
        logger.error("\nPossible reasons:")
        logger.error("  1. Market has no trades yet")
        logger.error("  2. Market ID is incorrect")
        logger.error("  3. API returned no data")
        logger.error("\nTry selecting a different market with:")
        logger.error("  python run_tracker.py --market-id <different_market_id>")
        logger.error("\nOr let the system auto-select:")
        logger.error("  python run_tracker.py")
        sys.exit(1)

    logger.info(f"✓ Loaded {len(trades_df)} real trades from Polymarket")

    # Preprocess data
    logger.info("\nPreprocessing data...")
    trades_df = preprocessor.clean_trades(trades_df)
    logger.info(f"  - Cleaned: {len(trades_df)} valid trades")

    trader_stats = preprocessor.aggregate_by_trader(trades_df)
    logger.info(f"  - Found {len(trader_stats)} unique traders")

    # Run rule-based detection
    logger.info("\n" + "="*70)
    logger.info("RULE-BASED DETECTION")
    logger.info("="*70)

    event_time = trades_df['timestamp'].max() - timedelta(hours=2)
    report = detector.generate_report(trades_df, market_id, event_time)

    logger.info(f"\n{report['summary']}")
    logger.info(f"\nOverall Risk Score: {report['risk_score']:.2f}/1.00")

    # Display detections
    timing_anomalies = report['detections']['timing_anomalies']
    volume_spikes = report['detections']['volume_spikes']
    coordinated = report['detections']['coordinated_trading']
    clusters = report['detections']['wallet_clusters']

    logger.info(f"\nDetailed Results:")
    logger.info(f"  - Timing Anomalies: {len(timing_anomalies)}")
    if timing_anomalies:
        for anomaly in timing_anomalies[:3]:
            logger.info(f"    • {anomaly['trader'][:15]}... "
                       f"({anomaly['minutes_before_event']:.1f} min before, "
                       f"size: ${anomaly['size']:.2f})")

    logger.info(f"  - Volume Spikes: {len(volume_spikes)}")
    if volume_spikes:
        for spike in volume_spikes[:3]:
            logger.info(f"    • {spike['timestamp']} "
                       f"(ratio: {spike['spike_ratio']:.2f}x normal)")

    logger.info(f"  - Coordinated Trading Events: {len(coordinated)}")
    if coordinated:
        for coord in coordinated[:3]:
            logger.info(f"    • {coord['num_traders']} traders, "
                       f"volume: ${coord['total_volume']:.2f}")

    logger.info(f"  - Wallet Clusters: {len(clusters)}")
    if clusters:
        for cluster_id, wallets in list(clusters.items())[:3]:
            logger.info(f"    • {cluster_id}: {len(wallets)} wallets")

    # Run ML detection if enabled
    if use_ml:
        logger.info("\n" + "="*70)
        logger.info("MACHINE LEARNING DETECTION")
        logger.info("="*70)

        logger.info("\nPreparing features...")
        features_df = ml_model.prepare_features(trades_df)
        logger.info(f"  Extracted {features_df.shape[1]} features")

        # For demonstration, create sequences and show capability
        X = features_df.values
        sequences, _ = ml_model.create_sequences(X)

        logger.info(f"  Created {len(sequences)} sequences")
        logger.info("\n⚠  Note: ML models need to be trained on labeled data")
        logger.info("   Run the Jupyter notebook to train and evaluate models")

    # Save results
    if save_results:
        logger.info("\n" + "="*70)
        logger.info("SAVING RESULTS")
        logger.info("="*70)

        output_dir = Path('data/results')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save report
        import json
        report_copy = report.copy()
        report_copy['analysis_time'] = report_copy['analysis_time'].isoformat()

        report_file = output_dir / f'detection_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report_copy, f, indent=2, default=str)

        logger.info(f"\n✓ Report saved: {report_file}")

        # Save trades
        trades_file = output_dir / f'trades_{timestamp}.csv'
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"✓ Trades saved: {trades_file}")

    logger.info("\n" + "="*70)
    logger.info("✓ TRACKING COMPLETE")
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='PolyPredict - Insider Trading Tracker for Polymarket'
    )

    parser.add_argument(
        '--market-id',
        type=str,
        default=None,
        help='Market ID to analyze (default: highest volume market)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days of historical data (default: 7)'
    )

    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Disable ML detection (faster)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )

    args = parser.parse_args()

    try:
        run_tracker(
            market_id=args.market_id,
            days_back=args.days,
            use_ml=not args.no_ml,
            save_results=not args.no_save
        )
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
