"""
News and Events Client
Fetches news and events that could trigger insider trading
"""

import os
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class NewsEventsClient:
    """
    Client for fetching news and events related to prediction markets

    Sources:
    - NewsAPI for general news
    - Twitter/X API for social media events
    - RSS feeds
    - Custom event sources
    """

    def __init__(self):
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")

        self.newsapi_base = "https://newsapi.org/v2"
        self.timeout = 30

        # Cache for events
        self.event_cache = {}

    def get_news(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
        page_size: int = 100
    ) -> List[Dict]:
        """
        Fetch news articles related to a query

        Args:
            query: Search query
            from_date: Start date for news
            to_date: End date for news
            language: Language code
            page_size: Number of results

        Returns:
            List of news article dictionaries
        """
        if not self.newsapi_key:
            logger.warning("NewsAPI key not set. Set NEWSAPI_KEY in .env")
            return []

        try:
            url = f"{self.newsapi_base}/everything"

            params = {
                'q': query,
                'language': language,
                'pageSize': min(page_size, 100),  # NewsAPI max is 100
                'apiKey': self.newsapi_key,
                'sortBy': 'publishedAt'
            }

            if from_date:
                params['from'] = from_date.strftime('%Y-%m-%d')
            if to_date:
                params['to'] = to_date.strftime('%Y-%m-%d')

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                logger.info(f"Fetched {len(articles)} news articles for query: {query}")
                return articles
            else:
                logger.error(f"NewsAPI error: {data.get('message')}")
                return []

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("NewsAPI rate limit exceeded")
            else:
                logger.error(f"HTTP error fetching news: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def get_market_related_news(
        self,
        market_question: str,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Get news related to a specific market question

        Args:
            market_question: The market question/topic
            days_back: Number of days to look back

        Returns:
            DataFrame with news articles and timestamps
        """
        from_date = datetime.now() - timedelta(days=days_back)

        # Extract key terms from market question
        keywords = self._extract_keywords(market_question)

        all_articles = []

        # Query for each keyword
        for keyword in keywords[:3]:  # Limit to top 3 keywords to avoid rate limits
            articles = self.get_news(
                query=keyword,
                from_date=from_date,
                page_size=20
            )
            all_articles.extend(articles)

        if not all_articles:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_articles)

        # Remove duplicates based on URL
        if 'url' in df.columns:
            df = df.drop_duplicates(subset=['url'])

        # Parse published date
        if 'publishedAt' in df.columns:
            df['timestamp'] = pd.to_datetime(df['publishedAt'])

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False)

        logger.info(f"Found {len(df)} unique news articles for market: {market_question[:50]}...")

        return df

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using simple heuristics

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Simple keyword extraction
        # In production, use NLP library like spaCy or NLTK

        # Remove common words
        stop_words = {
            'will', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'as', 'by', 'is', 'be', 'are', 'was', 'were',
            'what', 'when', 'where', 'who', 'which', 'how', 'than'
        }

        # Split and clean
        words = text.lower().split()
        keywords = [
            word.strip('?,.')
            for word in words
            if len(word) > 3 and word.lower() not in stop_words
        ]

        # Return unique keywords
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:5]

    def detect_announcement_timing(
        self,
        market_question: str,
        trades_df: pd.DataFrame,
        news_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Detect suspicious trading before news announcements

        Args:
            market_question: Market question
            trades_df: DataFrame with trades (must have 'timestamp' column)
            news_df: DataFrame with news (must have 'timestamp' column)

        Returns:
            List of suspicious timing events
        """
        if trades_df.empty or news_df.empty:
            return []

        if 'timestamp' not in trades_df.columns or 'timestamp' not in news_df.columns:
            logger.warning("Missing timestamp columns")
            return []

        suspicious_events = []

        # For each news article
        for _, news_row in news_df.iterrows():
            news_time = news_row['timestamp']

            # Look for trades in the 30 minutes before news
            time_window_start = news_time - timedelta(minutes=30)

            trades_before_news = trades_df[
                (trades_df['timestamp'] >= time_window_start) &
                (trades_df['timestamp'] < news_time)
            ]

            if len(trades_before_news) > 0:
                # Calculate volume spike
                avg_volume = trades_df['size'].mean() if 'size' in trades_df.columns else 0
                pre_news_volume = trades_before_news['size'].sum() if 'size' in trades_before_news.columns else 0

                if avg_volume > 0 and pre_news_volume > avg_volume * 2:
                    suspicious_events.append({
                        'news_title': news_row.get('title', 'N/A'),
                        'news_time': news_time,
                        'num_trades_before': len(trades_before_news),
                        'volume_before': pre_news_volume,
                        'avg_normal_volume': avg_volume,
                        'volume_ratio': pre_news_volume / avg_volume if avg_volume > 0 else 0,
                        'traders_involved': trades_before_news['maker'].nunique() if 'maker' in trades_before_news.columns else 0,
                        'minutes_before_news': [
                            int((news_time - t).total_seconds() / 60)
                            for t in trades_before_news['timestamp']
                        ]
                    })

        logger.info(f"Detected {len(suspicious_events)} suspicious trading events before news")

        return suspicious_events

    def get_twitter_mentions(
        self,
        query: str,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Get recent tweets mentioning a topic

        Args:
            query: Search query
            max_results: Maximum number of tweets

        Returns:
            List of tweet dictionaries
        """
        if not self.twitter_bearer_token:
            logger.warning("Twitter API token not set. Set TWITTER_BEARER_TOKEN in .env")
            return []

        try:
            url = "https://api.twitter.com/2/tweets/search/recent"

            headers = {
                'Authorization': f'Bearer {self.twitter_bearer_token}'
            }

            params = {
                'query': query,
                'max_results': min(max_results, 100),
                'tweet.fields': 'created_at,public_metrics,author_id'
            }

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            if 'data' in data:
                tweets = data['data']
                logger.info(f"Fetched {len(tweets)} tweets for query: {query}")
                return tweets
            else:
                logger.warning(f"No tweets found for query: {query}")
                return []

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Twitter API rate limit exceeded")
            else:
                logger.error(f"HTTP error fetching tweets: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")
            return []

    def create_event_timeline(
        self,
        market_question: str,
        trades_df: pd.DataFrame,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Create comprehensive timeline of trades and news events

        Args:
            market_question: Market question
            trades_df: Trading data
            days_back: Days to look back

        Returns:
            DataFrame with combined timeline
        """
        # Get news
        news_df = self.get_market_related_news(market_question, days_back)

        timeline = []

        # Add trades to timeline
        for _, trade in trades_df.iterrows():
            timeline.append({
                'timestamp': trade.get('timestamp'),
                'type': 'trade',
                'data': trade.to_dict()
            })

        # Add news to timeline
        for _, news in news_df.iterrows():
            timeline.append({
                'timestamp': news.get('timestamp'),
                'type': 'news',
                'data': {
                    'title': news.get('title'),
                    'source': news.get('source', {}).get('name'),
                    'url': news.get('url')
                }
            })

        # Convert to DataFrame and sort
        timeline_df = pd.DataFrame(timeline)

        if not timeline_df.empty and 'timestamp' in timeline_df.columns:
            timeline_df = timeline_df.sort_values('timestamp')

        return timeline_df


class EventDetector:
    """
    Detects important events that could trigger insider trading
    """

    def __init__(self):
        self.news_client = NewsEventsClient()

    def detect_major_events(
        self,
        market_question: str,
        days_back: int = 30
    ) -> List[Dict]:
        """
        Detect major events related to a market

        Args:
            market_question: Market question
            days_back: Days to analyze

        Returns:
            List of major events with metadata
        """
        news_df = self.news_client.get_market_related_news(
            market_question,
            days_back
        )

        if news_df.empty:
            return []

        major_events = []

        # Detect clusters of news (indicating major event)
        if 'timestamp' in news_df.columns:
            news_df['date'] = news_df['timestamp'].dt.date

            # Group by date
            daily_counts = news_df.groupby('date').size()

            # Find days with high news volume (potential major events)
            avg_daily_news = daily_counts.mean()
            std_daily_news = daily_counts.std()

            for date, count in daily_counts.items():
                if count > avg_daily_news + 2 * std_daily_news:
                    # This day had unusually high news volume
                    day_news = news_df[news_df['date'] == date]

                    major_events.append({
                        'date': date,
                        'news_count': count,
                        'avg_daily_news': avg_daily_news,
                        'headlines': day_news['title'].tolist()[:5],
                        'first_article_time': day_news['timestamp'].min(),
                        'significance': 'high' if count > avg_daily_news + 3 * std_daily_news else 'medium'
                    })

        logger.info(f"Detected {len(major_events)} major events for market")

        return major_events


if __name__ == "__main__":
    # Test the client
    client = NewsEventsClient()

    print("Testing News & Events client...")

    # Test news fetch
    if client.newsapi_key:
        articles = client.get_news("election", page_size=5)
        print(f"Fetched {len(articles)} news articles")

        if articles:
            print(f"Latest: {articles[0].get('title')}")
    else:
        print("NewsAPI key not configured (optional)")
