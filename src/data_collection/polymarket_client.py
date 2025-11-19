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

        self.session = None

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers with authentication if available"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_markets(self, active: bool = True) -> List[Dict]:
        """
        Fetch all markets from Polymarket

        Args:
            active: If True, only return active markets

        Returns:
            List of market dictionaries
        """
        try:
            url = f"{self.gamma_api}/markets"
            params = {"active": str(active).lower()}
            response = requests.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()

            markets = response.json()
            logger.info(f"Fetched {len(markets)} markets")
            return markets

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
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical trading data for a market

        Args:
            market_id: Market ID
            days_back: Number of days to look back

        Returns:
            DataFrame with historical trades
        """
        try:
            # Fetch trades in batches
            all_trades = []
            offset = 0
            batch_size = 1000

            while True:
                trades = self.get_trades(market_id, limit=batch_size)
                if not trades:
                    break

                all_trades.extend(trades)

                # Check if we've gone back far enough
                if len(trades) < batch_size:
                    break

                offset += batch_size

                # Simple rate limiting
                import time
                time.sleep(0.5)

            if not all_trades:
                return pd.DataFrame()

            df = pd.DataFrame(all_trades)

            # Convert timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

                # Filter by date range
                cutoff_date = datetime.now() - timedelta(days=days_back)
                df = df[df['timestamp'] >= cutoff_date]

            logger.info(f"Fetched {len(df)} historical trades for market {market_id}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
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
