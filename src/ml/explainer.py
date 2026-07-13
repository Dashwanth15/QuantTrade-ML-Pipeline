"""
QuantTrade ML Pipeline — SHAP Explainability Module
TreeExplainer for XGBoost with global and local explanation support.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — explainability features disabled")


class SHAPExplainer:
    """
    SHAP-based model explainability for XGBoost.
    
    Provides:
    - Global feature importance (mean |SHAP|)
    - SHAP values matrix for full dataset
    - Local explanation for single prediction
    - Feature interaction values (top-N pairs)
    """

    def __init__(self, model=None) -> None:
        self.model = model
        self._explainer = None
        self._shap_values: np.ndarray | None = None
        self._feature_names: list[str] | None = None

    def fit(self, model, X_train: pd.DataFrame) -> None:
        """Initialize the SHAP TreeExplainer."""
        if not SHAP_AVAILABLE:
            return
        self.model = model
        self._feature_names = list(X_train.columns)
        logger.debug("Initializing SHAP TreeExplainer")
        self._explainer = shap.TreeExplainer(model)

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray | None:
        """Compute SHAP values for a dataset."""
        if not SHAP_AVAILABLE or self._explainer is None:
            return None
        logger.debug("Computing SHAP values for {} samples", len(X))
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        shap_values = self._explainer.shap_values(X_clean)
        self._shap_values = shap_values
        return shap_values

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return global feature importance as mean |SHAP|.
        
        Returns:
            DataFrame with columns: feature, mean_abs_shap, rank
        """
        if not SHAP_AVAILABLE:
            return self._fallback_importance(X)

        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return self._fallback_importance(X)

        feature_names = list(X.columns)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        })
        importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)
        return importance_df.reset_index(drop=True)

    def local_explanation(
        self, X: pd.DataFrame, sample_idx: int = 0
    ) -> pd.DataFrame:
        """
        SHAP waterfall values for a single prediction.
        
        Returns:
            DataFrame with columns: feature, value, shap_value, impact_direction
        """
        if not SHAP_AVAILABLE or self._explainer is None:
            return pd.DataFrame()

        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        shap_values = self._explainer.shap_values(X_clean.iloc[[sample_idx]])
        feature_names = list(X.columns)

        explanation_df = pd.DataFrame({
            "feature": feature_names,
            "feature_value": X_clean.iloc[sample_idx].values,
            "shap_value": shap_values[0],
        })
        explanation_df["impact_direction"] = explanation_df["shap_value"].apply(
            lambda x: "positive" if x > 0 else "negative"
        )
        explanation_df = explanation_df.sort_values("shap_value", key=abs, ascending=False)
        return explanation_df.reset_index(drop=True)

    def get_shap_dataframe(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """Return SHAP values as a DataFrame for plotting."""
        if not SHAP_AVAILABLE:
            return None
        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return None
        return pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    @staticmethod
    def _fallback_importance(X: pd.DataFrame) -> pd.DataFrame:
        """Return empty importance when SHAP is unavailable."""
        return pd.DataFrame({
            "feature": list(X.columns),
            "mean_abs_shap": [0.0] * len(X.columns),
            "rank": range(1, len(X.columns) + 1),
        })
