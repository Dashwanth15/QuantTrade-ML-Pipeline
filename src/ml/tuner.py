"""
QuantTrade ML Pipeline — XGBoost Hyperparameter Tuner
Uses Optuna with TPESampler for Bayesian hyperparameter optimization.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from loguru import logger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings

# XGBoost search space for EUR/USD PnL prediction
SEARCH_SPACE = {
    "n_estimators": (100, 1000),
    "max_depth": (3, 9),
    "learning_rate": (0.005, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.3, 1.0),
    "min_child_weight": (1, 20),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.5, 5.0),
    "gamma": (0.0, 2.0),
}


class XGBoostTuner:
    """
    Optuna-based hyperparameter tuner for XGBoost.
    Uses TimeSeriesSplit within the training fold to select params.
    """

    def __init__(
        self,
        n_trials: int | None = None,
        random_seed: int | None = None,
        n_cv_folds: int = 3,
    ) -> None:
        self.n_trials = n_trials or settings.n_optuna_trials
        self.random_seed = random_seed or settings.random_seed
        self.n_cv_folds = n_cv_folds
        self.best_params_: dict | None = None
        self.study_: optuna.Study | None = None

    def tune(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> dict:
        """
        Run Optuna optimization on the given training data.

        Args:
            X_train: Feature DataFrame
            y_train: Target Series

        Returns:
            Best hyperparameters dict
        """
        logger.info(
            "Starting Optuna tuning | n_trials={} | n_cv={}",
            self.n_trials, self.n_cv_folds,
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", *SEARCH_SPACE["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *SEARCH_SPACE["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"], log=True),
                "subsample": trial.suggest_float("subsample", *SEARCH_SPACE["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *SEARCH_SPACE["colsample_bytree"]),
                "min_child_weight": trial.suggest_int("min_child_weight", *SEARCH_SPACE["min_child_weight"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *SEARCH_SPACE["reg_alpha"]),
                "reg_lambda": trial.suggest_float("reg_lambda", *SEARCH_SPACE["reg_lambda"]),
                "gamma": trial.suggest_float("gamma", *SEARCH_SPACE["gamma"]),
                "tree_method": "hist",
                "random_state": self.random_seed,
                "verbosity": 0,
                "n_jobs": -1,
                "objective": "reg:squarederror",
            }

            cv = TimeSeriesSplit(n_splits=self.n_cv_folds)
            scores = []
            for train_idx, val_idx in cv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                preds = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                scores.append(rmse)

            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        self.study_ = optuna.create_study(direction="minimize", sampler=sampler)
        self.study_.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params_ = self.study_.best_params
        self.best_params_["tree_method"] = "hist"
        self.best_params_["random_state"] = self.random_seed
        self.best_params_["verbosity"] = 0
        self.best_params_["n_jobs"] = -1
        self.best_params_["objective"] = "reg:squarederror"

        logger.info(
            "Tuning complete | best_rmse={:.6f} | best_params={}",
            self.study_.best_value,
            self.best_params_,
        )
        return self.best_params_

    @property
    def best_params(self) -> dict:
        if self.best_params_ is None:
            return self._default_params()
        return self.best_params_

    @staticmethod
    def _default_params() -> dict:
        """Sensible default parameters when tuning is skipped."""
        return {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "gamma": 0.1,
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
            "objective": "reg:squarederror",
        }
