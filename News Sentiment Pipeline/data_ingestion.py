import datetime
import logging

import praw
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

from config import NEWSAPI_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

def fetch_newsapi(query: str, from_date: str, to_date: str) -> list[dict]:
    articles = []
    url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&language=en&sortBy=relevancy&apiKey={NEWSAPI_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "ok":
            for article in data["articles"]:
                articles.append({
                    "timestamp": article.get("publishedAt"),
                    "title": article.get("title"),
                    "source": article.get("source", {}).get("name"),
                    "ticker": query.upper()
                })
            logging.info(f"Successfully fetched {len(articles)} articles from NewsAPI for query: {query}.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching from NewsAPI for query '{query}': {e}")
    return articles

def fetch_reddit(subreddit: str, limit: int = 100) -> list[dict]:
    logging.info(f"Attempting to fetch Reddit headlines from r/{subreddit} using PRAW.")
    posts = []
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        sub = reddit.subreddit(subreddit)
        for submission in sub.hot(limit=limit):
            if submission.stickied:
                continue
            posts.append({
                "timestamp": datetime.datetime.fromtimestamp(submission.created_utc).isoformat(),
                "title": submission.title,
                "source": f"Reddit/{subreddit}",
                "ticker": None
            })
        logging.info(f"Successfully fetched {len(posts)} posts from Reddit r/{subreddit}.")
    except praw.exceptions.ClientException as e:
        logging.error(f"PRAW Client Error for r/{subreddit}. Check your Reddit API credentials: {e}")
    except praw.exceptions.APIException as e:
        logging.error(f"Reddit API Error for r/{subreddit}: {e.response.status_code} - {e.message}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during Reddit fetching for r/{subreddit}: {e}")
    return posts

def fetch_finviz(ticker: str) -> list[dict]:
    logging.info(f"Attempting to fetch Finviz headlines for {ticker} via scraping.")
    headlines = []
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find('table', class_='fullview-news-outer')
        if news_table:
            rows = news_table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    timestamp_str = cols[0].get_text(strip=True)
                    title = cols[1].get_text(strip=True)
                    try:
                        if 'AM' in timestamp_str or 'PM' in timestamp_str:
                            parsed_dt = datetime.datetime.strptime(f"{datetime.date.today().strftime('%b-%d-%Y')} {timestamp_str}", "%b-%d-%Y %I:%M%p")
                        else:
                            parsed_dt = datetime.datetime.strptime(timestamp_str, "%b-%d-%Y")
                        timestamp = parsed_dt.isoformat()
                    except ValueError:
                        timestamp = datetime.datetime.now().isoformat()
                    headlines.append({
                        "timestamp": timestamp,
                        "title": title,
                        "source": "Finviz",
                        "ticker": ticker.upper()
                    })
            logging.info(f"Successfully fetched {len(headlines)} headlines from Finviz for {ticker}.")
        else:
            logging.warning(f"Finviz news table not found for {ticker}. Selector might be outdated.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching from Finviz for {ticker}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during Finviz scraping for {ticker}: {e}")
    return headlines


def fetch_yfinance_data(ticker: str, start_date: str, end_date: str, frequency: str = '1d') -> pd.DataFrame:
    logging.info(f"Fetching historical data for {ticker} from {start_date} to {end_date} with frequency {frequency}.")
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=frequency)
        if stock_data.empty:
            logging.warning(f"No data found for {ticker} for the given date range and frequency.")
            return pd.DataFrame()

        stock_data.index = pd.to_datetime(stock_data.index)

        # Handle missing values
        stock_data.ffill(inplace=True)
        stock_data.bfill(inplace=True)

        logging.info(f"Successfully fetched and cleaned data for {ticker}.")
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching data from yfinance for {ticker}: {e}")
        return pd.DataFrame()