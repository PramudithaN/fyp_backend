"""
Custom FinBERT-based sentiment analyzer for oil news.

Uses ProsusAI/finbert model loaded from local directory.
This matches the exact model and methodology used in Colab training.

MODEL DETAILS:
- Source: ProsusAI/finbert (downloaded to model_artifacts/sentiment_model/finbert_sentiment_model/)
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


def get_model_path() -> Path:
    """Get the path to the local sentiment model."""
    from app.config import MODEL_ARTIFACTS_DIR
    return MODEL_ARTIFACTS_DIR / "sentiment_model" / "finbert_sentiment_model"


def load_sentiment_model():
    """
    Load the custom FinBERT model and tokenizer from local directory.
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _model, _tokenizer, _device, _model_loaded
    
    if _model_loaded:
        return _model, _tokenizer, _device
    
    model_path = get_model_path()
    
    if not model_path.exists():
        logger.error(f"Sentiment model not found at: {model_path}")
        raise FileNotFoundError(
            f"Sentiment model not found at {model_path}. "
            "Please ensure the model files are in model_artifacts/sentiment_model/finbert_sentiment_model/"
        )
    
    logger.info(f"Loading sentiment model from: {model_path}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        _model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        
        # Use GPU if available
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
        
        _model_loaded = True
        logger.info(f"Sentiment model loaded successfully on {_device}")
        
        return _model, _tokenizer, _device
        
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {e}")
        raise


def analyze_sentiment_finbert(text: str) -> float:
    """
    Analyze sentiment of text using the custom FinBERT model.
    
    This matches the Colab training code exactly:
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
        
        # Match Colab training code exactly:
        # return probs[2] - probs[0]
        # Even though config shows 0=positive, 1=negative, 2=neutral,
        # we use the same formula as training for consistency
        sentiment_score = probs[2] - probs[0]
        
        return float(sentiment_score)
    
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")
        return 0.0


def analyze_batch_finbert(texts: List[str], batch_size: int = 16) -> List[float]:
    """
    Analyze sentiment of multiple texts efficiently using batching.
    
    This matches the Colab methodology:
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
            # Match Colab formula: probs[2] - probs[0]
            for j in range(len(batch)):
                score = probs[j][2] - probs[j][0]
                all_scores.append(float(score))
                
        except Exception as e:
            logger.warning(f"Batch analysis failed for batch starting at {i}: {e}")
            all_scores.extend([0.0] * len(batch))
    
    return all_scores


def is_finbert_available() -> bool:
    """Check if the custom FinBERT model is available."""
    try:
        model_path = get_model_path()
        if not model_path.exists():
            return False
        
        # Check for required files
        required_files = ["config.json", "tokenizer.json"]
        for f in required_files:
            if not (model_path / f).exists():
                return False
        
        # Also need either model.safetensors or pytorch_model.bin
        has_model = (model_path / "model.safetensors").exists() or \
                    (model_path / "pytorch_model.bin").exists()
        
        return has_model
        
    except Exception:
        return False


# For testing
if __name__ == "__main__":
    # Test the model
    test_texts = [
        "Oil prices surge as OPEC announces production cuts",
        "Crude oil crashes amid global recession fears",
        "Brent crude remains stable at $75 per barrel"
    ]
    
    print("Testing custom FinBERT sentiment analyzer...")
    print(f"Model available: {is_finbert_available()}")
    
    if is_finbert_available():
        for text in test_texts:
            score = analyze_sentiment_finbert(text)
            label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
            print(f"[{label:>8}] {score:+.4f}: {text[:50]}...")
