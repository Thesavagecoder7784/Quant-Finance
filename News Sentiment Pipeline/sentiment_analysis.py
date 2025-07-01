import logging
import torch

from config import get_finbert_models, vader_analyzer

finbert_tokenizer, finbert_model = get_finbert_models()

def analyze_vader_sentiment(text: str) -> dict:
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

def analyze_finbert_sentiment(texts: list[str], tokenizer, model) -> list[dict]:
    if not model or not tokenizer or not texts:
        logging.warning("FinBERT model or tokenizer not loaded, or texts are empty. Skipping FinBERT analysis.")
        return [{"positive": 0.0, "negative": 0.0, "neutral": 0.0, "classification": "neutral"}] * len(texts)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    labels = ["positive", "negative", "neutral"]
    results = []
    for pred in predictions:
        sentiment_scores = {label: p.item() for label, p in zip(labels, pred)}
        classification = labels[torch.argmax(pred)]
        results.append({
            "positive": sentiment_scores.get("positive", 0.0),
            "negative": sentiment_scores.get("negative", 0.0),
            "neutral": sentiment_scores.get("neutral", 0.0),
            "classification": classification
        })
    return results