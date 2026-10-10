import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.features.section_tfidf import FEATURE_NAMES as TFIDF_FEATURE_NAMES
from src.features.dense_embeddings import DENSE_FEATURE_NAMES

HYBRID_FEATURE_NAMES = TFIDF_FEATURE_NAMES + DENSE_FEATURE_NAMES

class HybridSectionFusionModel:
    """
    Hybrid Section Fusion Model combining Lexical TF-IDF similarities 
    and Dense Transformer Embedding similarities (16 features total).
    """

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0, random_state: int = 42):
        self.model_type = model_type
        self.alpha = alpha
        self.random_state = random_state

        if model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 4.0)

    def get_feature_importances(self) -> pd.DataFrame:
        """Extract feature weights/importances across 16 hybrid features."""
        if hasattr(self.model, "coef_"):
            importances = self.model.coef_
        elif hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = np.zeros(len(HYBRID_FEATURE_NAMES))

        df_imp = pd.DataFrame({
            "feature": HYBRID_FEATURE_NAMES,
            "channel": ["Lexical (TF-IDF)" if "dense" not in f else "Semantic (Dense)" for f in HYBRID_FEATURE_NAMES],
            "importance": importances
        }).sort_values("importance", ascending=False)
        return df_imp

    def save(self, filepath: str):
        joblib.dump({
            "model_type": self.model_type,
            "alpha": self.alpha,
            "model": self.model
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        instance = cls(model_type=data["model_type"], alpha=data.get("alpha", 1.0))
        instance.model = data["model"]
        return instance
