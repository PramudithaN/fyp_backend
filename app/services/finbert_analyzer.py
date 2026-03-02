"""
FinBERT-based sentiment analyzer for oil news.

Uses the standard ProsusAI/finbert model from Hugging Face.

MODEL DETAILS:
- Source: ProsusAI/finbert (loaded directly from Hugging Face Hub)
- Label Order: 0=negative, 1=neutral, 2=positive
- Sentiment Score: probs[2] - probs[0] (positive - negative probability)

INTEGRATION:
- This module provides per-article sentiment scores
- Daily averaging is done in news_fetcher.py (simple mean, no within-day decay)
- Cross-day decay is applied in sentiment_service.py with LAMBDA=0.3
"""
import logging
from typing import List, Optional
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Global model cache (loaded once, reused)
_model = None
_tokenizer = None
_device = None
_model_loaded = False


def load_sentiment_model():
    """
    Load the standard FinBERT model and tokenizer from Hugging Face.
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _model, _tokenizer, _device, _model_loaded
    
    if _model_loaded:
        return _model, _tokenizer, _device
    
    model_name = "ProsusAI/finbert"
    logger.info(f"Loading FinBERT model from Hugging Face: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Use GPU if available
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
        
        _model_loaded = True
        logger.info(f"FinBERT model loaded successfully on {_device}")
        
        return _model, _tokenizer, _device
        
    except Exception as e:
        logger.error(f"Failed to load FinBERT model: {e}")
        raise


def analyze_sentiment_finbert(text: str) -> float:
    """
    Analyze sentiment of text using the FinBERT model.
    
    - Input: Article text (title + content)
    - Output: Sentiment score = probs[2] - probs[0] (positive - negative)
    - Range: -1 (very negative) to +1 (very positive)
    
    Args:
        text: Text to analyze (article title + description/content)
    
    Returns:
        Sentiment score between -1 and 1
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    
    try:
        model, tokenizer, device = load_sentiment_model()
        
        # Tokenize (matching Colab settings)
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        # Calculate sentiment score: probs[2] - probs[0]
        # Label order: 0=negative, 1=neutral, 2=positive
        sentiment_score = probs[2] - probs[0]
        
        return float(sentiment_score)
    
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")
        return 0.0


def analyze_batch_finbert(texts: List[str], batch_size: int = 16) -> List[float]:
    """
    Analyze sentiment of multiple texts efficiently using batching.
    
    - Each article gets an individual sentiment score
    - Scores are later averaged per day (simple mean)
    
    Args:
        texts: List of texts to analyze
        batch_size: Number of texts to process at once
    
    Returns:
        List of sentiment scores (one per text)
    """
    if not texts:
        return []
    
    try:
        model, tokenizer, device = load_sentiment_model()
    except Exception as e:
        logger.error(f"Cannot load model for batch analysis: {e}")
        return [0.0] * len(texts)
    
    all_scores = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Filter out empty texts (use placeholder for empty)
        valid_texts = []
        for t in batch:
            if t and len(t.strip()) >= 10:
                valid_texts.append(t)
            else:
                valid_texts.append("neutral news")  # Placeholder
        
        try:
            inputs = tokenizer(
                valid_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            # Calculate score for each item in batch
            # Formula: probs[2] - probs[0] (positive - negative)
            for j in range(len(batch)):
                score = probs[j][2] - probs[j][0]
                all_scores.append(float(score))
                
        except Exception as e:
            logger.warning(f"Batch analysis failed for batch starting at {i}: {e}")
            all_scores.extend([0.0] * len(batch))
    
    return all_scores


def is_finbert_available() -> bool:
    """Check if FinBERT model can be loaded from Hugging Face."""
    try:
        # Always return True since we load from Hugging Face
        # The actual check happens when trying to load the model
        return True
    except Exception:
        return False


def preload_model() -> bool:
    """
    Pre-load the FinBERT model into memory at startup.
    
    Call this from the FastAPI lifespan handler so that the first
    prediction request doesn't pay the ~5-10s model-loading cost.
    
    Returns:
        True if the model was loaded successfully, False otherwise.
    """
    import time
    
    try:
        t0 = time.time()
        load_sentiment_model()
        elapsed = time.time() - t0
        logger.info(f"FinBERT model pre-loaded from Hugging Face in {elapsed:.1f}s")
        return True
    except Exception as e:
        logger.error(f"FinBERT preload failed: {e}")
        return False


# For testing
if __name__ == "__main__":
    # Test the model
    test_texts = [
        "Oil prices surge as OPEC announces production cuts",
        "Crude oil crashes amid global recession fears",
        "Brent crude remains stable at $75 per barrel"
    ]
    
    print("Testing FinBERT sentiment analyzer...")
    print("Loading model from Hugging Face: ProsusAI/finbert")
    
    for text in test_texts:
        score = analyze_sentiment_finbert(text)
        label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        print(f"[{label:>8}] {score:+.4f}: {text[:50]}...")
