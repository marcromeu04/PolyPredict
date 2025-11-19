"""
Polymarket Data Collection Client
Handles API calls to Polymarket CLOB, Gamma, and Data APIs
"""

import os
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class PolymarketClient:
    """Client for fetching data from Polymarket APIs"""

    def __init__(self):
        self.clob_api = os.getenv("POLYMARKET_CLOB_API", "https://clob.polymarket.com")
        self.gamma_api = os.getenv("POLYMARKET_GAMMA_API", "https://gamma-api.polymarket.com")
        self.data_api = os.getenv("POLYMARKET_DATA_API", "https://data-api.polymarket.com")
        self.api_key = os.getenv("POLYMARKET_API_KEY")

        # Request timeout settings
        self.timeout = 30

        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 0.1  # 100ms between requests

        self.session = None

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers with authentication if available"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PolyPredict/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _rate_limit(self, endpoint: str):
        """Simple rate limiting"""
        import time
        current_time = time.time()
        last_time = self.last_request_time.get(endpoint, 0)
        time_since_last = current_time - last_time

        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time[endpoint] = time.time()

    def get_markets(self, active: bool = True, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Fetch all markets from Polymarket

        Args:
            active: If True, only return active markets
            limit: Maximum number of markets to return
            offset: Pagination offset

        Returns:
            List of market dictionaries
        """
        try:
            self._rate_limit("markets")
            url = f"{self.gamma_api}/markets"
            params = {
                "active": str(active).lower(),
                "limit": limit,
                "offset": offset
            }
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            markets = response.json()
            logger.info(f"Fetched {len(markets)} markets (offset={offset})")
            return markets

        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching markets")
            return []
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit hit, waiting...")
                import time
                time.sleep(5)
                return []
            logger.error(f"HTTP error fetching markets: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def get_market_by_id(self, market_id: str) -> Optional[Dict]:
        """
        Get specific market details

        Args:
            market_id: Market condition ID

        Returns:
            Market data dictionary or None
        """
        try:
            url = f"{self.gamma_api}/markets/{market_id}"
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            return None

    def get_trades(self, market_id: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades for a market

        Args:
            market_id: Market condition ID
            limit: Maximum number of trades to fetch

        Returns:
            List of trade dictionaries
        """
        try:
            url = f"{self.clob_api}/trades"
            params = {
                "market": market_id,
                "limit": limit
            }
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()

            trades = response.json()
            logger.info(f"Fetched {len(trades)} trades for market {market_id}")
            return trades

        except Exception as e:
            logger.error(f"Error fetching trades for {market_id}: {e}")
            return []

    def get_orderbook(self, token_id: str) -> Dict:
        """
        Get current orderbook for a token

        Args:
            token_id: Token ID

        Returns:
            Orderbook data with bids and asks
        """
        try:
            url = f"{self.clob_api}/book"
            params = {"token_id": token_id}
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching orderbook for {token_id}: {e}")
            return {"bids": [], "asks": []}

    def get_user_trades(self, user_address: str, limit: int = 100) -> List[Dict]:
        """
        Get trades for a specific user/wallet

        Args:
            user_address: Ethereum address of user
            limit: Maximum number of trades

        Returns:
            List of user's trades
        """
        try:
            url = f"{self.clob_api}/trades"
            params = {
                "maker": user_address,
                "limit": limit
            }
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching trades for user {user_address}: {e}")
            return []

    def get_user_positions(self, user_address: str) -> List[Dict]:
        """
        Get current positions for a user

        Args:
            user_address: Ethereum address

        Returns:
            List of positions
        """
        try:
            url = f"{self.data_api}/positions"
            params = {"user": user_address}
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching positions for {user_address}: {e}")
            return []

    async def stream_trades_async(self, market_id: str, callback):
        """
        Stream trades in real-time using WebSocket

        Args:
            market_id: Market to monitor
            callback: Function to call with each trade
        """
        ws_url = f"wss://ws-subscriptions-clob.polymarket.com/ws/market/{market_id}"

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                logger.info(f"Connected to WebSocket for market {market_id}")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        await callback(data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break

    def get_historical_data(
        self,
        market_id: str,
        days_back: int = 30,
        max_trades: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch historical trading data for a market

        Args:
            market_id: Market ID
            days_back: Number of days to look back
            max_trades: Maximum number of trades to fetch (safety limit)

        Returns:
            DataFrame with historical trades
        """
        try:
            # Fetch trades in batches
            all_trades = []
            batch_size = 500  # Reduced from 1000 to be more conservative
            cutoff_date = datetime.now() - timedelta(days=days_back)

            # Get initial batch
            trades = self.get_trades(market_id, limit=batch_size)

            if not trades:
                logger.warning(f"No trades found for market {market_id}")
                return pd.DataFrame()

            all_trades.extend(trades)

            # Try to get older trades using before parameter if API supports it
            # Note: This assumes trades have 'id' or 'timestamp' field
            while len(all_trades) < max_trades:
                if not trades or len(trades) < batch_size:
                    break

                # Get last trade timestamp/id for pagination
                # Polymarket API uses 'next_cursor' for pagination
                # This is a simplified version - real API might differ
                last_trade = trades[-1]

                # Check if we've gone back far enough in time
                if 'timestamp' in last_trade:
                    trade_time = pd.to_datetime(last_trade['timestamp'], unit='s')
                    if trade_time < cutoff_date:
                        break

                # Rate limiting
                import time
                time.sleep(0.5)

                # Fetch next batch - note: actual pagination may differ
                trades = self.get_trades(market_id, limit=batch_size)
                if trades:
                    all_trades.extend(trades)
                else:
                    break

            if not all_trades:
                return pd.DataFrame()

            df = pd.DataFrame(all_trades)

            # Remove duplicates based on trade ID if present
            if 'id' in df.columns:
                df = df.drop_duplicates(subset=['id'])

            # Convert timestamp if present
            if 'timestamp' in df.columns:
                # Try different timestamp formats
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                except:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except:
                        logger.warning("Could not parse timestamps")

                # Filter by date range
                if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df = df[df['timestamp'] >= cutoff_date]
                    df = df.sort_values('timestamp', ascending=False)

            logger.info(f"Fetched {len(df)} historical trades for market {market_id}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()


if __name__ == "__main__":
    # Test the client
    client = PolymarketClient()

    # Fetch some markets
    markets = client.get_markets(active=True)
    print(f"Found {len(markets)} active markets")

    if markets:
        # Get details for first market
        first_market = markets[0]
        print(f"\nFirst market: {first_market.get('question', 'N/A')}")

        # Get trades if market has condition_id
        if 'condition_id' in first_market:
            trades = client.get_trades(first_market['condition_id'], limit=10)
            print(f"Recent trades: {len(trades)}")
