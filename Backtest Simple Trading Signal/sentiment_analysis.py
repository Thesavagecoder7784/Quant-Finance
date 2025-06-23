import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize FinBERT model and tokenizer
try:
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    logging.info("FinBERT model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading FinBERT model: {e}. Please ensure you have an internet connection or the model is cached.")
    finbert_tokenizer = None
    finbert_model = None

def analyze_vader_sentiment(text: str) -> dict:
    """
    Analyzes sentiment using NLTK's VADER.
    Returns compound score, and classifies as 'positive', 'negative', or 'neutral'.
    """
    if not text:
        return {"compound": 0.0, "classification": "neutral"}
    score = vader_analyzer.polarity_scores(text)
    compound_score = score['compound']

    if compound_score >= 0.05:
        classification = "positive"
    elif compound_score <= -0.05:
        classification = "negative"
    else:
        classification = "neutral"
    return {"compound": compound_score, "classification": classification}

def analyze_finbert_sentiment(text: str) -> dict:
    """
    Analyzes sentiment using the FinBERT model.
    Returns probabilities for 'positive', 'negative', 'neutral', and the highest confidence classification.
    """
    if not finbert_model or not finbert_tokenizer or not text:
        logging.warning("FinBERT model or tokenizer not loaded, or text is empty. Skipping FinBERT analysis.")
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "classification": "neutral"}

    inputs = finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = finbert_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    sentiment_scores = {label: pred.item() for label, pred in zip(labels, predictions[0])}

    classification = labels[torch.argmax(predictions)]

    return {
        "positive": sentiment_scores.get("positive", 0.0),
        "negative": sentiment_scores.get("negative", 0.0),
        "neutral": sentiment_scores.get("neutral", 0.0),
        "classification": classification
    }