"""
QuantTrade ML Pipeline — Prediction & Inference Engine
Loads persisted model and generates predictions with confidence intervals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.simulation.engine import STRATEGY_REGISTRY


class PredictionEngine:
    """
    Loads a saved model and generates predictions for new data.
    
    Features:
    - Load model from joblib file
    - Predict expected PnL per strategy
    - Bootstrap confidence intervals
    - Strategy ranking by predicted PnL
    - Real-time feature computation for current bar
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or self._find_latest_model())
        self._artifact: dict | None = None
        self._pipeline = None
        self._feature_names: list[str] | None = None

    def load(self) -> None:
        """Load the model artifact from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        logger.info("Loading model from {}", self.model_path)
        self._artifact = joblib.load(self.model_path)
        self._pipeline = self._artifact["pipeline"]
        self._feature_names = self._artifact["feature_names"]
        logger.success(
            "Model loaded | run_id={} | features={}",
            self._artifact.get("run_id"), len(self._feature_names),
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict PnL for each row in X."""
        if self._pipeline is None:
            self.load()
        X_aligned = self._align_features(X)
        predictions = self._pipeline.predict(X_aligned)
        return pd.Series(predictions, index=X.index, name="predicted_pnl")

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        n_bootstrap: int = 50,
        alpha: float = 0.10,
    ) -> pd.DataFrame:
        """
        Predict PnL with bootstrap confidence intervals.

        Returns:
            DataFrame with columns: predicted_pnl, ci_lower, ci_upper
        """
        if self._pipeline is None:
            self.load()

        X_aligned = self._align_features(X)
        base_pred = self._pipeline.predict(X_aligned)

        # Bootstrap via feature perturbation
        bootstrap_preds = np.zeros((n_bootstrap, len(X)))
        for i in range(n_bootstrap):
            # Perturb features with small noise
            noise = np.random.normal(0, 0.01, X_aligned.shape)
            X_perturbed = X_aligned + noise
            bootstrap_preds[i] = self._pipeline.predict(X_perturbed)

        ci_lower = np.percentile(bootstrap_preds, (alpha / 2) * 100, axis=0)
        ci_upper = np.percentile(bootstrap_preds, (1 - alpha / 2) * 100, axis=0)

        return pd.DataFrame({
            "predicted_pnl": base_pred,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }, index=X.index)

    def recommend_strategies(
        self, X: pd.DataFrame, feature_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate strategy recommendations for the given market state.

        Returns:
            DataFrame ranked by predicted_pnl descending
        """
        if self._pipeline is None:
            self.load()

        # Use the most recent features
        recent_features = feature_df.tail(1)
        X_aligned = self._align_features(recent_features)

        # For each strategy, the features would be the same market context
        # but the trade-specific features differ. Use the base prediction
        # and adjust by historical strategy performance metrics.
        base_pred = float(self._pipeline.predict(X_aligned)[0])

        # Load strategy-specific adjustments from training results
        strategy_metrics = self._artifact.get("strategy_metrics", {})

        records = []
        for strategy_id in STRATEGY_REGISTRY.keys():
            adjustment = strategy_metrics.get(strategy_id, {}).get("avg_pnl_adjustment", 1.0)
            pred_pnl = base_pred * adjustment
            records.append({
                "strategy_id": strategy_id,
                "predicted_pnl": pred_pnl,
                "confidence": "high" if abs(pred_pnl) > 10 else "medium",
                "recommended": pred_pnl > 0,
            })

        df = pd.DataFrame(records).sort_values("predicted_pnl", ascending=False)
        df["rank"] = range(1, len(df) + 1)
        return df

    def get_model_info(self) -> dict:
        """Return model metadata."""
        if self._artifact is None:
            self.load()
        return {
            "run_id": self._artifact.get("run_id"),
            "target_col": self._artifact.get("target_col"),
            "n_features": len(self._feature_names or []),
            "trained_at": self._artifact.get("trained_at"),
            "best_params": self._artifact.get("best_params"),
            "overall_metrics": self._artifact.get("overall_metrics"),
        }

    # ------------------------------------------------------------------ #
    # Private Methods
    # ------------------------------------------------------------------ #

    def _align_features(self, X: pd.DataFrame) -> np.ndarray:
        """Align X to expected feature names, filling missing with 0."""
        if self._feature_names is None:
            return X.fillna(0).replace([np.inf, -np.inf], 0).values

        aligned = pd.DataFrame(index=X.index)
        for col in self._feature_names:
            aligned[col] = X.get(col, 0.0)
        return aligned.fillna(0).replace([np.inf, -np.inf], 0).values

    @staticmethod
    def _find_latest_model() -> Path:
        """Find the most recently created model file."""
        model_dir = Path(settings.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        models = sorted(model_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not models:
            raise FileNotFoundError(f"No model found in {model_dir}")
        return models[0]
