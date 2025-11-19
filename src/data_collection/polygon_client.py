"""
Polygon Blockchain Client for On-Chain Data
Direct access to Polymarket contracts via Web3 and PolygonScan API
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


class PolygonClient:
    """
    Client for accessing Polymarket on-chain data from Polygon

    Uses:
    - PolygonScan API for transaction history
    - Web3 for direct contract interaction (optional)
    """

    # Polymarket contract addresses on Polygon
    CONTRACTS = {
        'CTF_EXCHANGE': '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E',  # Main exchange
        'CONDITIONAL_TOKENS': '0x4D97DCd97eC945f40cF65F87097ACe5EA0476045',  # CTF
        'UMA_ADAPTER': '0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74',  # UMA oracle adapter
        'NEG_RISK_ADAPTER': '0x87d1fF9ebD6b28cB0780f8fA7B5763b5FDDCc072',  # Neg risk adapter
        'NEG_RISK_FEE_MODULE': '0x78769d50bE1763ed1ca0d5e878d93F05aabFf29e'
    }

    def __init__(self):
        self.polygonscan_api_key = os.getenv("POLYGONSCAN_API_KEY", "")
        self.polygon_rpc = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")

        # PolygonScan API endpoint
        self.api_base = "https://api.polygonscan.com/api"

        self.timeout = 30

        # Try to initialize Web3 if available
        self.web3 = None
        try:
            from web3 import Web3
            if self.polygon_rpc:
                self.web3 = Web3(Web3.HTTPProvider(self.polygon_rpc))
                if self.web3.is_connected():
                    logger.info(f"Connected to Polygon RPC: {self.polygon_rpc}")
                else:
                    logger.warning("Web3 RPC connection failed")
                    self.web3 = None
        except ImportError:
            logger.warning("web3 not installed. Install with: pip install web3")
        except Exception as e:
            logger.warning(f"Could not connect to Polygon RPC: {e}")

    def _polygonscan_request(self, params: Dict) -> Optional[Dict]:
        """
        Make request to PolygonScan API

        Args:
            params: Request parameters

        Returns:
            API response or None on error
        """
        try:
            if self.polygonscan_api_key:
                params['apikey'] = self.polygonscan_api_key

            response = requests.get(
                self.api_base,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            if data.get('status') == '1':
                return data.get('result')
            else:
                logger.warning(f"PolygonScan API error: {data.get('message')}")
                return None

        except Exception as e:
            logger.error(f"Error querying PolygonScan: {e}")
            return None

    def get_contract_transactions(
        self,
        contract_address: str,
        start_block: int = 0,
        end_block: int = 999999999,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict]:
        """
        Get transactions for a contract address

        Args:
            contract_address: Contract address
            start_block: Starting block number
            end_block: Ending block number
            page: Page number for pagination
            offset: Number of transactions per page

        Returns:
            List of transaction dictionaries
        """
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': contract_address,
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': offset,
            'sort': 'desc'
        }

        result = self._polygonscan_request(params)
        return result if result else []

    def get_internal_transactions(
        self,
        contract_address: str,
        start_block: int = 0,
        end_block: int = 999999999
    ) -> List[Dict]:
        """
        Get internal transactions for a contract

        Args:
            contract_address: Contract address
            start_block: Starting block
            end_block: Ending block

        Returns:
            List of internal transactions
        """
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': contract_address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': 'desc'
        }

        result = self._polygonscan_request(params)
        return result if result else []

    def get_token_transfers(
        self,
        contract_address: Optional[str] = None,
        address: Optional[str] = None,
        start_block: int = 0,
        end_block: int = 999999999,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict]:
        """
        Get ERC20 token transfer events

        Args:
            contract_address: Token contract address
            address: Wallet address to filter by
            start_block: Starting block
            end_block: Ending block
            page: Page number
            offset: Number of results per page

        Returns:
            List of token transfer events
        """
        params = {
            'module': 'account',
            'action': 'tokentx',
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': offset,
            'sort': 'desc'
        }

        if contract_address:
            params['contractaddress'] = contract_address
        if address:
            params['address'] = address

        result = self._polygonscan_request(params)
        return result if result else []

    def get_event_logs(
        self,
        contract_address: str,
        topic0: Optional[str] = None,
        from_block: int = 0,
        to_block: int = 999999999
    ) -> List[Dict]:
        """
        Get event logs from a contract

        Args:
            contract_address: Contract address
            topic0: Event signature hash (optional)
            from_block: Starting block
            to_block: Ending block

        Returns:
            List of event logs
        """
        params = {
            'module': 'logs',
            'action': 'getLogs',
            'address': contract_address,
            'fromBlock': from_block,
            'toBlock': to_block
        }

        if topic0:
            params['topic0'] = topic0

        result = self._polygonscan_request(params)
        return result if result else []

    def get_polymarket_trades_onchain(
        self,
        days_back: int = 7,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get Polymarket trades from on-chain data

        Args:
            days_back: Number of days to look back
            limit: Maximum number of transactions to fetch

        Returns:
            DataFrame with trade data
        """
        try:
            # Get current block number
            latest_block = self._polygonscan_request({
                'module': 'proxy',
                'action': 'eth_blockNumber'
            })

            if not latest_block:
                logger.error("Could not get latest block number")
                return pd.DataFrame()

            latest_block = int(latest_block, 16)

            # Estimate blocks for time period (Polygon ~2 sec block time)
            blocks_per_day = 43200  # ~86400 / 2
            start_block = max(0, latest_block - (days_back * blocks_per_day))

            logger.info(f"Fetching on-chain data from block {start_block} to {latest_block}")

            # Get transactions from main CTF Exchange
            transactions = self.get_contract_transactions(
                self.CONTRACTS['CTF_EXCHANGE'],
                start_block=start_block,
                end_block=latest_block,
                offset=min(limit, 10000)  # PolygonScan max is 10000
            )

            if not transactions:
                logger.warning("No on-chain transactions found")
                return pd.DataFrame()

            df = pd.DataFrame(transactions)

            # Convert timestamp
            df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timeStamp'], unit='s')

            # Convert values from wei
            df['value_matic'] = pd.to_numeric(df['value'], errors='coerce') / 1e18
            df['gas_used'] = pd.to_numeric(df['gasUsed'], errors='coerce')
            df['gas_price_gwei'] = pd.to_numeric(df['gasPrice'], errors='coerce') / 1e9

            # Filter successful transactions
            df = df[df['txreceipt_status'] == '1']

            logger.info(f"Fetched {len(df)} on-chain transactions")

            return df

        except Exception as e:
            logger.error(f"Error fetching on-chain trades: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def get_user_onchain_activity(
        self,
        user_address: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get comprehensive on-chain activity for a user

        Args:
            user_address: Ethereum address
            days_back: Number of days to look back

        Returns:
            Dictionary with user activity statistics
        """
        try:
            # Get current block
            latest_block = self._polygonscan_request({
                'module': 'proxy',
                'action': 'eth_blockNumber'
            })

            if not latest_block:
                return {}

            latest_block = int(latest_block, 16)
            blocks_per_day = 43200
            start_block = max(0, latest_block - (days_back * blocks_per_day))

            # Get all transactions
            transactions = []
            for contract_name, contract_address in self.CONTRACTS.items():
                txs = self.get_contract_transactions(
                    contract_address,
                    start_block=start_block,
                    end_block=latest_block
                )

                # Filter by user
                user_txs = [
                    tx for tx in txs
                    if tx.get('from', '').lower() == user_address.lower()
                    or tx.get('to', '').lower() == user_address.lower()
                ]

                transactions.extend(user_txs)

            if not transactions:
                return {
                    'address': user_address,
                    'num_transactions': 0,
                    'total_gas_used': 0,
                    'contracts_interacted': []
                }

            df = pd.DataFrame(transactions)

            return {
                'address': user_address,
                'num_transactions': len(df),
                'total_gas_used': df['gasUsed'].astype(float).sum(),
                'avg_gas_price_gwei': df['gasPrice'].astype(float).mean() / 1e9,
                'total_value_matic': df['value'].astype(float).sum() / 1e18,
                'first_tx': pd.to_datetime(df['timeStamp'].astype(int).min(), unit='s'),
                'last_tx': pd.to_datetime(df['timeStamp'].astype(int).max(), unit='s'),
                'contracts_interacted': df['to'].unique().tolist(),
                'unique_contracts': df['to'].nunique()
            }

        except Exception as e:
            logger.error(f"Error getting user on-chain activity: {e}")
            return {}

    def get_wallet_balance(self, address: str) -> Optional[float]:
        """
        Get MATIC balance for an address

        Args:
            address: Ethereum address

        Returns:
            Balance in MATIC or None
        """
        params = {
            'module': 'account',
            'action': 'balance',
            'address': address,
            'tag': 'latest'
        }

        result = self._polygonscan_request(params)

        if result:
            return int(result) / 1e18

        return None


if __name__ == "__main__":
    # Test the client
    client = PolygonClient()

    print("Testing Polygon client...")
    print(f"Polymarket CTF Exchange: {client.CONTRACTS['CTF_EXCHANGE']}")

    # Test fetching recent trades
    print("\nFetching recent on-chain transactions...")
    df = client.get_polymarket_trades_onchain(days_back=1, limit=10)

    if not df.empty:
        print(f"Found {len(df)} on-chain transactions")
        print(df[['timestamp', 'from', 'to', 'value_matic', 'gas_used']].head())
    else:
        print("No data returned (may need PolygonScan API key)")
