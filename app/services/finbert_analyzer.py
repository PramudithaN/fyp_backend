"""
FinBERT-based sentiment analyzer for financial news.

Uses the ProsusAI/finbert model for accurate financial sentiment analysis.
"""
import logging
from typing import List, Tuple, Optional
import torch

logger = logging.getLogger(__name__)

# Global model cache (loaded once, reused)
_finbert_model = None
_finbert_tokenizer = None
_device = None


def get_finbert_model():
    """
    Load FinBERT model and tokenizer (cached after first load).
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _finbert_model, _finbert_tokenizer, _device
    
    if _finbert_model is None:
        logger.info("Loading FinBERT model (first time, may take a moment)...")
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_name = "ProsusAI/finbert"
        
        _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Use GPU if available
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _finbert_model.to(_device)
        _finbert_model.eval()
        
        logger.info(f"FinBERT loaded on {_device}")
    
    return _finbert_model, _finbert_tokenizer, _device


def analyze_sentiment_finbert(text: str) -> float:
    """
    Analyze sentiment of text using FinBERT.
    
    Args:
        text: Text to analyze (title + description)
    
    Returns:
        Sentiment score between -1 (negative) and 1 (positive)
        
    FinBERT outputs 3 classes:
        - 0: positive
        - 1: negative  
        - 2: neutral
    
    We convert to a score: positive - negative (range: -1 to 1)
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    
    try:
        model, tokenizer, device = get_finbert_model()
        
        # Tokenize (truncate to 512 tokens max)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
        
        # FinBERT labels: positive=0, negative=1, neutral=2
        positive_prob = probs[0].item()
        negative_prob = probs[1].item()
        # neutral_prob = probs[2].item()
        
        # Convert to single score: positive - negative
        sentiment_score = positive_prob - negative_prob
        
        return sentiment_score
    
    except Exception as e:
        logger.warning(f"FinBERT analysis failed: {e}")
        return 0.0


def analyze_batch_finbert(texts: List[str], batch_size: int = 16) -> List[float]:
    """
    Analyze sentiment of multiple texts efficiently using batching.
    
    Args:
        texts: List of texts to analyze
        batch_size: Number of texts to process at once
    
    Returns:
        List of sentiment scores
    """
    if not texts:
        return []
    
    model, tokenizer, device = get_finbert_model()
    
    all_scores = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Filter out empty texts
        valid_texts = [t if t and len(t.strip()) >= 10 else "neutral" for t in batch]
        
        try:
            inputs = tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
            
            # Calculate scores for batch
            for j in range(len(batch)):
                positive_prob = probs[j][0].item()
                negative_prob = probs[j][1].item()
                score = positive_prob - negative_prob
                all_scores.append(score)
                
        except Exception as e:
            logger.warning(f"Batch analysis failed: {e}")
            all_scores.extend([0.0] * len(batch))
    
    return all_scores


def is_finbert_available() -> bool:
    """Check if FinBERT can be loaded."""
    try:
        from transformers import AutoModelForSequenceClassification
        return True
    except ImportError:
        return False
