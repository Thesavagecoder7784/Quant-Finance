import logging
import os
from datetime import date, timedelta

import nltk
import torch
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

DATABASE_NAME = "news_sentiment.db"

vader_analyzer = SentimentIntensityAnalyzer()

try:
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    logging.info("FinBERT model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading FinBERT model: {e}. Please ensure you have an internet connection or the model is cached.")
    finbert_tokenizer = None
    finbert_model = None

def ensure_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        logging.info("Downloading VADER lexicon...")
        nltk.download('vader_lexicon')
        logging.info("VADER lexicon downloaded.")

def get_default_dates():
    today = date.today()
    seven_days_ago = today - timedelta(days=7)
    return seven_days_ago.isoformat(), today.isoformat()

def get_finbert_models():
    return finbert_tokenizer, finbert_model