"""
The Graph API Client for Polymarket Subgraph
Provides access to on-chain data via GraphQL queries
"""

import os
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class TheGraphClient:
    """
    Client for querying Polymarket subgraph on The Graph

    Provides access to:
    - Markets data
    - Trades history
    - User positions
    - Liquidity data
    - Volume statistics
    """

    def __init__(self):
        # The Graph API endpoint for Polymarket subgraph
        # Default to free hosted service endpoint
        self.api_key = os.getenv("THEGRAPH_API_KEY", "")

        if self.api_key:
            # Decentralized network endpoint (requires API key)
            self.endpoint = f"https://gateway.thegraph.com/api/{self.api_key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp"
        else:
            # Free hosted service (may have rate limits)
            self.endpoint = "https://api.thegraph.com/subgraphs/name/polymarket/polymarket-subgraph"
            logger.warning("No The Graph API key set. Using free hosted service (rate limited)")

        self.timeout = 30

    def _query(self, query: str, variables: Optional[Dict] = None) -> Optional[Dict]:
        """
        Execute a GraphQL query

        Args:
            query: GraphQL query string
            variables: Optional query variables

        Returns:
            Query result or None on error
        """
        try:
            payload = {"query": query}
            if variables:
                payload["variables"] = variables

            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            if "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")
                return None

            return result.get("data")

        except requests.exceptions.Timeout:
            logger.error("The Graph query timeout")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error querying The Graph: {e}")
            return None
        except Exception as e:
            logger.error(f"Error querying The Graph: {e}")
            return None

    def get_market_info(self, market_id: str) -> Optional[Dict]:
        """
        Get detailed market information from subgraph

        Args:
            market_id: Market/condition ID

        Returns:
            Market information dictionary
        """
        query = """
        query GetMarket($marketId: ID!) {
            fixedProductMarketMaker(id: $marketId) {
                id
                creator
                collateralToken
                fee
                collateralVolume
                scaledCollateralVolume
                usdVolume
                outcomeTokenAmounts
                outcomeSlotCount
                liquidityMeasure
                createTimestamp
                lastActiveDay
                runningDailyVolume
                creationTransactionHash
                fpmmPoolMemberships {
                    amount
                    funder {
                        id
                    }
                }
            }
        }
        """

        variables = {"marketId": market_id.lower()}
        result = self._query(query, variables)

        if result and "fixedProductMarketMaker" in result:
            return result["fixedProductMarketMaker"]

        return None

    def get_trades(
        self,
        market_id: Optional[str] = None,
        user_address: Optional[str] = None,
        first: int = 100,
        skip: int = 0,
        start_timestamp: Optional[int] = None
    ) -> List[Dict]:
        """
        Get trades from subgraph

        Args:
            market_id: Filter by market ID
            user_address: Filter by user address
            first: Number of results to return
            skip: Number of results to skip (pagination)
            start_timestamp: Filter trades after this timestamp

        Returns:
            List of trade dictionaries
        """
        # Build where clause
        where_conditions = []
        if market_id:
            where_conditions.append(f'fpmm: "{market_id.lower()}"')
        if user_address:
            where_conditions.append(f'creator: "{user_address.lower()}"')
        if start_timestamp:
            where_conditions.append(f'creationTimestamp_gte: "{start_timestamp}"')

        where_clause = ", ".join(where_conditions) if where_conditions else ""

        query = f"""
        query GetTrades {{
            fpmmTrades(
                first: {first},
                skip: {skip},
                orderBy: creationTimestamp,
                orderDirection: desc
                {f', where: {{{where_clause}}}' if where_clause else ''}
            ) {{
                id
                type
                creator {{
                    id
                }}
                creationTimestamp
                collateralToken
                collateralAmount
                collateralAmountUSD
                feeAmount
                outcomeIndex
                outcomeTokensTraded
                transactionHash
                fpmm {{
                    id
                    outcomeSlotCount
                }}
            }}
        }}
        """

        result = self._query(query)

        if result and "fpmmTrades" in result:
            return result["fpmmTrades"]

        return []

    def get_user_positions(self, user_address: str, first: int = 100) -> List[Dict]:
        """
        Get user's positions across all markets

        Args:
            user_address: Ethereum address
            first: Number of results

        Returns:
            List of position dictionaries
        """
        query = """
        query GetUserPositions($user: String!, $first: Int!) {
            account(id: $user) {
                id
                fpmmPoolMemberships(first: $first) {
                    amount
                    fpmm {
                        id
                        collateralToken
                        outcomeSlotCount
                        outcomeTokenAmounts
                        collateralVolume
                        usdVolume
                    }
                }
                fpmmParticipations(first: $first) {
                    fpmm {
                        id
                    }
                    creationTimestamp
                    collateralTokenAmount
                    fee
                }
            }
        }
        """

        variables = {
            "user": user_address.lower(),
            "first": first
        }

        result = self._query(query, variables)

        if result and "account" in result:
            return result["account"]

        return {}

    def get_market_volume_history(
        self,
        market_id: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get daily volume history for a market

        Args:
            market_id: Market ID
            days: Number of days to look back

        Returns:
            DataFrame with daily volume data
        """
        start_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())

        query = f"""
        query GetVolume {{
            fixedProductMarketMaker(id: "{market_id.lower()}") {{
                id
                usdVolume
                collateralVolume
                scaledCollateralVolume
                runningDailyVolume
                lastActiveDay
                fpmmTrades(
                    where: {{ creationTimestamp_gte: "{start_timestamp}" }},
                    orderBy: creationTimestamp,
                    orderDirection: asc,
                    first: 1000
                ) {{
                    creationTimestamp
                    collateralAmountUSD
                    collateralAmount
                }}
            }}
        }}
        """

        result = self._query(query)

        if not result or "fixedProductMarketMaker" not in result:
            return pd.DataFrame()

        market_data = result["fixedProductMarketMaker"]
        if not market_data or "fpmmTrades" not in market_data:
            return pd.DataFrame()

        trades = market_data["fpmmTrades"]
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['creationTimestamp'], unit='s')
        df['date'] = df['timestamp'].dt.date

        # Aggregate by day
        daily_volume = df.groupby('date').agg({
            'collateralAmountUSD': 'sum',
            'collateralAmount': 'sum'
        }).reset_index()

        daily_volume.columns = ['date', 'usd_volume', 'collateral_volume']

        return daily_volume

    def get_top_traders(
        self,
        market_id: Optional[str] = None,
        limit: int = 50,
        min_trades: int = 5
    ) -> List[Dict]:
        """
        Get top traders by volume

        Args:
            market_id: Optional market ID to filter by
            limit: Number of top traders to return
            min_trades: Minimum number of trades to be included

        Returns:
            List of trader statistics
        """
        where_clause = f'fpmm: "{market_id.lower()}"' if market_id else ""

        query = f"""
        query GetTraders {{
            fpmmTrades(
                first: 1000,
                orderBy: creationTimestamp,
                orderDirection: desc
                {f', where: {{{where_clause}}}' if where_clause else ''}
            ) {{
                creator {{
                    id
                }}
                collateralAmountUSD
                creationTimestamp
            }}
        }}
        """

        result = self._query(query)

        if not result or "fpmmTrades" not in result:
            return []

        trades = result["fpmmTrades"]

        # Aggregate by trader
        trader_stats = {}
        for trade in trades:
            trader_id = trade['creator']['id']
            usd_amount = float(trade.get('collateralAmountUSD', 0))

            if trader_id not in trader_stats:
                trader_stats[trader_id] = {
                    'address': trader_id,
                    'total_volume_usd': 0,
                    'trade_count': 0,
                    'first_trade': None,
                    'last_trade': None
                }

            trader_stats[trader_id]['total_volume_usd'] += usd_amount
            trader_stats[trader_id]['trade_count'] += 1

            timestamp = int(trade['creationTimestamp'])
            if trader_stats[trader_id]['first_trade'] is None or timestamp < trader_stats[trader_id]['first_trade']:
                trader_stats[trader_id]['first_trade'] = timestamp
            if trader_stats[trader_id]['last_trade'] is None or timestamp > trader_stats[trader_id]['last_trade']:
                trader_stats[trader_id]['last_trade'] = timestamp

        # Filter and sort
        top_traders = [
            stats for stats in trader_stats.values()
            if stats['trade_count'] >= min_trades
        ]

        top_traders.sort(key=lambda x: x['total_volume_usd'], reverse=True)

        return top_traders[:limit]

    def get_market_liquidity(self, market_id: str) -> Optional[Dict]:
        """
        Get current liquidity information for a market

        Args:
            market_id: Market ID

        Returns:
            Liquidity information dictionary
        """
        query = """
        query GetLiquidity($marketId: ID!) {
            fixedProductMarketMaker(id: $marketId) {
                id
                liquidityMeasure
                collateralToken
                outcomeTokenAmounts
                fpmmPoolMemberships(first: 100, orderBy: amount, orderDirection: desc) {
                    amount
                    funder {
                        id
                    }
                }
            }
        }
        """

        variables = {"marketId": market_id.lower()}
        result = self._query(query, variables)

        if result and "fixedProductMarketMaker" in result:
            market_data = result["fixedProductMarketMaker"]

            # Calculate total liquidity
            total_liquidity = sum(
                float(membership['amount'])
                for membership in market_data.get('fpmmPoolMemberships', [])
            )

            return {
                'market_id': market_data['id'],
                'liquidity_measure': market_data.get('liquidityMeasure'),
                'total_liquidity': total_liquidity,
                'num_liquidity_providers': len(market_data.get('fpmmPoolMemberships', [])),
                'outcome_token_amounts': market_data.get('outcomeTokenAmounts', []),
                'top_providers': market_data.get('fpmmPoolMemberships', [])[:10]
            }

        return None


if __name__ == "__main__":
    # Test the client
    client = TheGraphClient()

    # Test market info
    print("Testing The Graph client...")
    print(f"Endpoint: {client.endpoint}")

    # Note: This will only work with a valid market ID
    # You would get this from the Polymarket API first
