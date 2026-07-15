"""
QuantTrade ML Pipeline — Training Orchestrator
Full training pipeline: feature prep → walk-forward → tune → fit → evaluate → persist.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.ml.evaluator import ModelEvaluator
from src.ml.explainer import SHAPExplainer
from src.ml.pipeline import build_xgboost_pipeline, prepare_ml_dataset
from src.ml.tuner import XGBoostTuner
from src.ml.walk_forward import WalkForwardValidator


class ModelTrainer:
    """
    End-to-end training orchestrator.

    Workflow:
    1. Prepare ML dataset (align features + trades)
    2. Generate walk-forward folds
    3. For each fold: optionally tune → fit → evaluate
    4. Fit final model on all data with best params
    5. Compute SHAP values
    6. Persist model artifacts
    7. Return comprehensive results dict
    """

    def __init__(
        self,
        run_id: str | None = None,
        tune_hyperparams: bool = True,
        target_col: str = "pnl_usd",
    ) -> None:
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
        self.tune_hyperparams = tune_hyperparams
        self.target_col = target_col
        self.model_path = Path(settings.model_path) / f"{self.run_id}.joblib"
        self._results: dict = {}

    def train(
        self,
        feature_df: pd.DataFrame,
        trade_log: pd.DataFrame,
    ) -> dict:
        """
        Run the full training pipeline.

        Returns:
            Results dict with metrics, fold details, model path, feature importance
        """
        logger.info("Starting training run: {}", self.run_id)

        # ---- 1. Prepare ML Dataset ----
        logger.info("Preparing ML dataset")
        X, y, timestamps = prepare_ml_dataset(feature_df, trade_log, self.target_col)
        logger.info("ML dataset: {} samples × {} features", len(X), len(X.columns))
        self._results["n_samples"] = len(X)
        self._results["n_features"] = len(X.columns)
        self._results["feature_names"] = list(X.columns)
        self._results["target_col"] = self.target_col

        # ---- 2. Walk-Forward Folds ----
        wf = WalkForwardValidator()
        # Need a DF with timestamps as index
        indexed = X.copy()
        indexed.index = timestamps.values
        folds = wf.split(indexed)

        if not wf.verify_no_leakage(folds):
            raise RuntimeError("Data leakage detected! Aborting training.")

        if len(folds) == 0:
            raise ValueError("No walk-forward folds generated — dataset too small")

        logger.info("Running walk-forward validation: {} folds", len(folds))

        # ---- 3. Walk-Forward Training ----
        fold_results = []
        all_predictions = []
        all_actuals = []
        best_params = None
        evaluator = ModelEvaluator()

        for fold in folds:
            X_train = X[fold.train_mask]
            y_train = y[fold.train_mask]
            X_test = X[fold.test_mask]
            y_test = y[fold.test_mask]

            if len(X_train) < 50 or len(X_test) < 10:
                logger.warning("Fold {} too small, skipping", fold.fold_index)
                continue

            # Tune hyperparameters on first fold (expensive — reuse for subsequent)
            if self.tune_hyperparams and best_params is None:
                logger.info("Tuning hyperparameters on fold {}", fold.fold_index)
                tuner = XGBoostTuner(n_trials=settings.n_optuna_trials)
                best_params = tuner.tune(X_train, y_train)
            elif best_params is None:
                best_params = XGBoostTuner._default_params()

            # Build and train pipeline (imputer + scaler fit on train only)
            pipeline = build_xgboost_pipeline(best_params)
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred = pipeline.predict(X_test)
            metrics = evaluator.evaluate(y_test, y_pred)

            fold_result = {
                **fold.to_dict(),
                **metrics,
                "best_params": best_params,
            }
            fold_results.append(fold_result)

            all_predictions.extend(y_pred.tolist())
            all_actuals.extend(y_test.tolist())

            logger.info(
                "Fold {} | MAE={:.4f} | Sharpe={:.2f} | WinRate={:.1%}",
                fold.fold_index, metrics["mae"], metrics["sharpe"], metrics["win_rate"],
            )

        # ---- 4. Aggregate Fold Metrics ----
        fold_metrics_df = pd.DataFrame(fold_results)
        overall_metrics = evaluator.evaluate(
            np.array(all_actuals), np.array(all_predictions)
        )
        logger.info(evaluator.format_report(overall_metrics))

        # ---- 5. Final Model on All Data ----
        logger.info("Training final model on full dataset")
        final_pipeline = build_xgboost_pipeline(best_params)
        final_pipeline.fit(X, y)

        # ---- 6. SHAP Explainability ----
        # Extract the XGBoost model from pipeline for SHAP
        xgb_model = final_pipeline.named_steps["model"]
        scaler = final_pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X.fillna(0).replace([np.inf, -np.inf], 0))
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        explainer = SHAPExplainer()
        explainer.fit(xgb_model, X_scaled_df)
        shap_importance = explainer.global_importance(X_scaled_df.head(min(5000, len(X_scaled_df))))

        # XGBoost native importance
        xgb_importance = pd.DataFrame({
            "feature": list(xgb_model.get_booster().get_score(importance_type="gain").keys()),
            "importance_gain": list(xgb_model.get_booster().get_score(importance_type="gain").values()),
        }).sort_values("importance_gain", ascending=False)

        # ---- 7. Persist Artifacts ----
        logger.info("Saving model to {}", self.model_path)
        Path(settings.model_path).mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": final_pipeline,
                "feature_names": list(X.columns),
                "run_id": self.run_id,
                "target_col": self.target_col,
                "best_params": best_params,
                "overall_metrics": overall_metrics,
                "fold_results": fold_results,
                "all_predictions": all_predictions,
                "all_actuals": all_actuals,
                "shap_importance": shap_importance.to_dict("records") if not shap_importance.empty else [],
                "xgb_importance": xgb_importance.to_dict("records") if not xgb_importance.empty else [],
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            self.model_path,
            compress=3,
        )
        logger.success("Model saved: {}", self.model_path)

        # ---- 8. Compile Results ----
        self._results.update({
            "run_id": self.run_id,
            "model_path": str(self.model_path),
            "best_params": best_params,
            "overall_metrics": overall_metrics,
            "fold_results": fold_results,
            "fold_metrics_df": fold_metrics_df,
            "shap_importance": shap_importance,
            "xgb_importance": xgb_importance,
            "n_folds": len(fold_results),
            "all_predictions": all_predictions,
            "all_actuals": all_actuals,
            "feature_names": list(X.columns),
        })

        logger.success(
            "Training complete! Run: {} | Sharpe: {:.2f} | MAE: {:.4f}",
            self.run_id, overall_metrics["sharpe"], overall_metrics["mae"],
        )
        return self._results

    @property
    def results(self) -> dict:
        return self._results
