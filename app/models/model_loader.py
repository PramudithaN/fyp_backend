"""
Model artifact loader - loads all trained models at startup.
"""
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from app.config import MODEL_ARTIFACTS_DIR, HORIZON
from app.models.gru_models import MidFreqGRU, SentimentGRU

logger = logging.getLogger(__name__)


class ModelArtifacts:
    """
    Container for all loaded model artifacts.
    Loaded once at application startup.
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.scaler_mid = None
        self.scaler_price = None
        self.scaler_sent = None
        self.meta_models: Dict[int, Any] = {}
        self.meta_scalers: Dict[int, Any] = {}
        self.xgb_hf_models: Dict[int, Any] = {}
        self.mid_gru: Optional[MidFreqGRU] = None
        self.sent_gru: Optional[SentimentGRU] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
    
    def load_all(self) -> None:
        """Load all model artifacts from disk."""
        if self._loaded:
            logger.info("Models already loaded, skipping...")
            return
        
        logger.info(f"Loading model artifacts from {MODEL_ARTIFACTS_DIR}")
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self._load_config()
        
        # Load scalers
        self._load_scalers()
        
        # Load meta-ensemble models
        self._load_meta_models()
        
        # Load XGBoost HF models
        self._load_xgb_models()
        
        # Load PyTorch GRU models
        self._load_gru_models()
        
        self._loaded = True
        logger.info("All model artifacts loaded successfully!")
    
    def _load_config(self) -> None:
        """Load configuration from config.pkl"""
        config_path = MODEL_ARTIFACTS_DIR / "config.pkl"
        self.config = joblib.load(config_path)
        logger.info(f"Config loaded: LOOKBACK={self.config.get('LOOKBACK')}, "
                   f"HORIZON={self.config.get('HORIZON')}, "
                   f"ARIMA_ORDER={self.config.get('ARIMA_ORDER')}")
    
    def _load_scalers(self) -> None:
        """Load all sklearn scalers."""
        self.scaler_mid = joblib.load(MODEL_ARTIFACTS_DIR / "scaler_mid.pkl")
        self.scaler_price = joblib.load(MODEL_ARTIFACTS_DIR / "scaler_price.pkl")
        self.scaler_sent = joblib.load(MODEL_ARTIFACTS_DIR / "scaler_sent.pkl")
        
        # Check for scaler_mid mismatch (known issue from training notebook)
        # The mid-GRU expects 14 features, but scaler_mid may have been saved with fewer
        logger.info(f"Scalers loaded - mid: {self.scaler_mid.n_features_in_}, "
                   f"price: {self.scaler_price.n_features_in_}, "
                   f"sent: {self.scaler_sent.n_features_in_}")
    
    def _load_meta_models(self) -> None:
        """Load Ridge meta-ensemble models and their scalers."""
        self.meta_models = joblib.load(MODEL_ARTIFACTS_DIR / "meta_models.pkl")
        self.meta_scalers = joblib.load(MODEL_ARTIFACTS_DIR / "meta_scalers.pkl")
        logger.info(f"Meta models loaded for horizons: {list(self.meta_models.keys())}")
    
    def _load_xgb_models(self) -> None:
        """Load XGBoost high-frequency models."""
        self.xgb_hf_models = joblib.load(MODEL_ARTIFACTS_DIR / "xgb_hf_models.pkl")
        logger.info(f"XGBoost HF models loaded for horizons: {list(self.xgb_hf_models.keys())}")
    
    def _load_gru_models(self) -> None:
        """Load PyTorch GRU models."""
        from sklearn.preprocessing import StandardScaler
        
        horizon = self.config.get('HORIZON', HORIZON)
        
        # Get feature dimensions from scalers
        n_price_features = self.scaler_price.n_features_in_
        n_sent_features = self.scaler_sent.n_features_in_
        
        # For mid-GRU, check the actual model weight dimensions
        mid_gru_path = MODEL_ARTIFACTS_DIR / "mid_gru.pt"
        mid_state = torch.load(mid_gru_path, map_location='cpu')
        n_mid_features = mid_state['gru.weight_ih_l0'].shape[1]
        
        logger.info(f"Feature dimensions - mid: {n_mid_features}, "
                   f"price: {n_price_features}, sent: {n_sent_features}")
        
        # Handle scaler_mid mismatch
        if self.scaler_mid.n_features_in_ != n_mid_features:
            logger.warning(f"scaler_mid dimension mismatch: saved={self.scaler_mid.n_features_in_}, "
                          f"model expects={n_mid_features}. Creating new StandardScaler.")
            # Create a new scaler that will use identity transform
            # (mean=0, std=1 for all features - no scaling)
            self.scaler_mid = StandardScaler()
            # Fit with dummy data to set the right dimensions
            dummy_data = np.zeros((1, n_mid_features))
            self.scaler_mid.fit(dummy_data)
            # Set to identity transform (no actual scaling)
            self.scaler_mid.mean_ = np.zeros(n_mid_features)
            self.scaler_mid.scale_ = np.ones(n_mid_features)
            self.scaler_mid.var_ = np.ones(n_mid_features)
            logger.info("Created identity StandardScaler for mid features")
        
        # Load Mid-frequency GRU
        self.mid_gru = MidFreqGRU(
            n_features=n_mid_features,
            hidden_size=64,
            dropout=0.3,
            horizon=horizon
        ).to(self.device)
        
        self.mid_gru.load_state_dict(mid_state)
        self.mid_gru.eval()
        logger.info("Mid-frequency GRU loaded")
        
        # Load Sentiment GRU
        self.sent_gru = SentimentGRU(
            n_price=n_price_features,
            n_sent=n_sent_features,
            hidden=64,
            dropout=0.3,
            horizon=horizon
        ).to(self.device)
        
        sent_gru_path = MODEL_ARTIFACTS_DIR / "sent_gru.pt"
        self.sent_gru.load_state_dict(
            torch.load(sent_gru_path, map_location=self.device)
        )
        self.sent_gru.eval()
        logger.info("Sentiment GRU loaded")
    
    @property
    def lookback(self) -> int:
        return self.config.get('LOOKBACK', 30)
    
    @property
    def horizon(self) -> int:
        return self.config.get('HORIZON', 14)
    
    @property
    def arima_order(self) -> tuple:
        return self.config.get('ARIMA_ORDER', (1, 0, 1))
    
    @property
    def price_features(self) -> list:
        return self.config.get('price_features', [])
    
    @property
    def sentiment_features(self) -> list:
        return self.config.get('sentiment_features', [])


# Global singleton instance
model_artifacts = ModelArtifacts()
