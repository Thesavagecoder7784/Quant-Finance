import logging

import pandas as pd

def assign_ticker_to_reddit_headlines(reddit_headlines: list[dict], ticker_keywords_map: dict) -> list[dict]:
    """Assigns a ticker to Reddit headlines based on keyword matching."""
    processed_headlines = []
    for headline in reddit_headlines:
        text = headline.get("title", "")
        if not text:
            continue

        assigned_ticker = None
        for ticker, keywords in ticker_keywords_map.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                assigned_ticker = ticker
                break
        
        headline['ticker'] = assigned_ticker if assigned_ticker else 'MARKET'
        processed_headlines.append(headline)
        
    logging.info(f"Assigned tickers to {len(processed_headlines)} Reddit headlines.")
    return processed_headlines

def process_headlines_in_batch(headlines: list[dict], sentiment_analyzers: dict) -> list[dict]:
    """Processes a batch of headlines using vectorization for efficiency."""
    if not headlines:
        return []

    df = pd.DataFrame(headlines)
    df['text'] = df['title'].fillna('')

    # Vectorized sentiment analysis
    df['vader_sentiment'] = df['text'].apply(sentiment_analyzers['vader'].polarity_scores)
    df['vader_compound'] = df['vader_sentiment'].apply(lambda x: x['compound'])
    df['vader_classification'] = df['vader_compound'].apply(lambda c: 'positive' if c >= 0.05 else ('negative' if c <= -0.05 else 'neutral'))

    # Batch sentiment analysis with FinBERT
    finbert_results = sentiment_analyzers['finbert'](df['text'].tolist())
    df_finbert = pd.DataFrame(finbert_results)
    df['finbert_classification'] = df_finbert['classification']
    df['finbert_positive'] = df_finbert['positive']
    df['finbert_negative'] = df_finbert['negative']
    df['finbert_neutral'] = df_finbert['neutral']

    processed_data = df.to_dict('records')
    logging.info(f"Processed {len(processed_data)} headlines in a batch.")
    return processed_data