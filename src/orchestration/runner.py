import argparse
import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.database.repository import get_repos, get_db
from src.database.models import ForexCandle, MacroEvent, Trade, ModelRun, Prediction
from src.ingestion.forex_loader import load_forex_data
from src.ingestion.macro_scraper import scrape_macro_events
from src.preprocessing.cleaner import clean_forex_data
from src.preprocessing.validator import validate_forex_data
from src.features.feature_store import FeatureStore
from src.simulation.engine import run_simulation
from src.ml.trainer import ModelTrainer
from src.ml.predictor import PredictionEngine

# Import visualization utilities for pre-rendering
from src.visualization.charts import (
    candlestick_chart, session_heatmap, return_distribution_chart,
    macro_event_timeline, correlation_heatmap, equity_curve_chart,
    strategy_radar_chart, rolling_metrics_chart, drawdown_chart,
    feature_importance_chart, shap_summary_chart, walk_forward_chart,
    prediction_vs_actual_chart
)

class PipelineRunner:
    """
    Orchestrates the entire QuantTrade pipeline.
    Checks SQLite and local filesystem cache to skip already completed steps.
    """

    def __init__(self, force: bool = False) -> None:
        self.force = force
        self.db = get_db()
        self.forex_repo, self.macro_repo, self.trade_repo, self.model_repo = get_repos()
        self.feature_cache_path = Path(settings.data_processed_path) / "features.parquet"
        
        # Setup outputs directories
        self.outputs_dir = Path("data/outputs")
        self.models_dir = self.outputs_dir / "models"
        self.datasets_dir = self.outputs_dir / "datasets"
        self.reports_dir = self.outputs_dir / "reports"
        self.charts_dir = self.outputs_dir / "charts"
        
        for d in [self.models_dir, self.datasets_dir, self.reports_dir, self.charts_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _save_chart(self, fig, name: str) -> None:
        try:
            fig.write_json(str(self.charts_dir / f"{name}.json"))
            fig.write_html(str(self.charts_dir / f"{name}.html"))
            fig.write_image(str(self.charts_dir / f"{name}.png"), engine="kaleido")
            logger.info("Saved pre-rendered chart: {}", name)
        except Exception as e:
            logger.error("Failed to pre-render chart '{}': {}", name, e)

    def run(self) -> None:
        logger.info("=== Starting QuantTrade ML Pipeline ===")
        
        # 1. Ingestion, Cleaning & Validation
        forex_df = self._step_forex_data()
        
        # 2. Macro Events (dynamic range alignment!)
        macro_df = self._step_macro_events(forex_df)
        
        # 3. Feature Engineering
        feature_df = self._step_feature_engineering(forex_df, macro_df)
        
        # 4. Trade Simulation
        trade_log = self._step_trade_simulation(feature_df)
        
        # 5. Model Training
        model_results = self._step_model_training(feature_df, trade_log)
        
        # 6. Predictions
        self._step_predictions(feature_df, model_results)
        
        logger.success("=== Pipeline Orchestration Complete! ===")

    def _step_forex_data(self) -> pd.DataFrame:
        candles_count = self.forex_repo.count()
        csv_path = self.datasets_dir / "cleaned_forex.csv"
        parquet_path = self.datasets_dir / "cleaned_forex.parquet"
        
        if candles_count > 0 and csv_path.exists() and parquet_path.exists() and not self.force:
            logger.info("Step 1: Forex Data already loaded in database ({} rows) and saved on disk. Skipping Ingestion.", candles_count)
            return self.forex_repo.load_all()
        
        logger.info("Step 1: Ingesting, cleaning, and validating Forex Data...")
        raw_df = load_forex_data()
        cleaned_df, report = clean_forex_data(raw_df)
        validation = validate_forex_data(cleaned_df)
        if not validation.passed:
            raise RuntimeError("Forex data validation failed.")
        
        self.forex_repo.bulk_insert(cleaned_df)
        
        # Save to disk
        cleaned_df.to_parquet(parquet_path)
        cleaned_df.to_csv(csv_path)
        logger.info("Saved cleaned forex dataset to disk.")
        
        # Pre-render charts
        self._save_chart(candlestick_chart(cleaned_df, indicators=["ema_21", "ema_200"], n_bars=1000), "market_candlestick")
        returns = cleaned_df["mid_close"].pct_change().dropna() * 10000
        self._save_chart(return_distribution_chart(returns, "Hourly Return Distribution (pips)"), "market_returns_dist")
        self._save_chart(session_heatmap(cleaned_df), "market_session_heatmap")
        
        return cleaned_df

    def _step_macro_events(self, forex_df: pd.DataFrame) -> pd.DataFrame:
        with self.db.session() as sess:
            macro_count = sess.query(MacroEvent).count()
        csv_path = self.datasets_dir / "macro_events.csv"
            
        if macro_count > 0 and csv_path.exists() and not self.force:
            logger.info("Step 2: Macro Events already loaded in database ({} rows) and saved on disk. Skipping Ingestion.", macro_count)
            return self.macro_repo.load_all()

        logger.info("Step 2: Scraping Macro Events dynamically matching Forex date range...")
        start_date = forex_df.index.min().to_pydatetime()
        end_date = forex_df.index.max().to_pydatetime()
        
        try:
            macro_df = scrape_macro_events(
                api_key=settings.apify_api_key,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            logger.warning(f"Apify scrape failed, using synthetic fallback: {e}")
            macro_df = pd.DataFrame()
            
        if not macro_df.empty:
            self.macro_repo.bulk_insert(macro_df)
        else:
            # Re-generate synthetic fallback if empty/failed
            scraper = scrape_macro_events(start_date=start_date, end_date=end_date)
            self.macro_repo.bulk_insert(scraper)
            macro_df = self.macro_repo.load_all()
            
        # Save to disk
        macro_df.to_csv(csv_path, index=False)
        logger.info("Saved macro events dataset to disk.")
        
        # Pre-render charts
        self._save_chart(macro_event_timeline(forex_df, macro_df, n_bars=1000), "macro_timeline")
        return macro_df

    def _step_feature_engineering(self, forex_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        csv_path = self.datasets_dir / "engineered_features.csv"
        parquet_path = self.datasets_dir / "engineered_features.parquet"
        
        if self.feature_cache_path.exists() and csv_path.exists() and parquet_path.exists() and not self.force:
            logger.info("Step 3: Features cache found on disk. Skipping Feature Engineering.")
            return pd.read_parquet(self.feature_cache_path)

        logger.info("Step 3: Engineering features...")
        store = FeatureStore()
        feature_df = store.build_features(forex_df, macro_df=macro_df, use_cache=False)
        self.feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(self.feature_cache_path)
        feature_df.to_parquet(parquet_path)
        feature_df.to_csv(csv_path)
        logger.success("Features engineered and saved to {}", self.feature_cache_path)
        
        # Pre-render correlation heatmap
        self._save_chart(correlation_heatmap(feature_df, n_features=30), "feature_correlation")
        return feature_df

    def _step_trade_simulation(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        with self.db.session() as sess:
            trades_count = sess.query(Trade).count()
        trade_log_csv = self.datasets_dir / "trade_log.csv"
        recommendations_csv = self.reports_dir / "strategy_recommendations.csv"
            
        if trades_count > 0 and trade_log_csv.exists() and recommendations_csv.exists() and not self.force:
            logger.info("Step 4: Trade logs and recommendations already saved on disk. Skipping Trade Simulation.")
            return self.trade_repo.load_all()

        logger.info("Step 4: Running Trade Simulation...")
        trade_log, strategy_summary = run_simulation(feature_df)
        self.trade_repo.bulk_insert(trade_log)
        
        # Save to disk
        trade_log.to_csv(trade_log_csv, index=False)
        strategy_summary.to_csv(recommendations_csv, index=False)
        logger.info("Saved trade log and strategy summary recommendations to disk.")
        
        # Pre-render charts
        self._save_chart(equity_curve_chart(trade_log), "strategy_equity_curves")
        self._save_chart(strategy_radar_chart(strategy_summary), "strategy_radar")
        self._save_chart(rolling_metrics_chart(trade_log, window=50), "rolling_performance")
        
        pnl_series = trade_log.sort_values("exit_time")["pnl_usd"].reset_index(drop=True)
        self._save_chart(drawdown_chart(pnl_series, "Portfolio Drawdown"), "portfolio_drawdown")
        
        return trade_log

    def _step_model_training(self, feature_df: pd.DataFrame, trade_log: pd.DataFrame) -> dict:
        latest_run = self.model_repo.load_latest_run()
        eval_report_json = self.reports_dir / "evaluation_report.json"
        wf_results_csv = self.reports_dir / "walk_forward_results.csv"
        feat_imp_csv = self.reports_dir / "feature_importance.csv"
        shap_rep_csv = self.reports_dir / "shap_report.csv"
        model_copy_path = self.models_dir / "model.joblib"
        
        if latest_run and latest_run.get("status") == "success" and eval_report_json.exists() and wf_results_csv.exists() and not self.force:
            model_path = Path(latest_run["model_path"])
            if model_path.exists():
                logger.info("Step 5: Trained model run '{}' exists. Skipping Model Training.", latest_run["run_id"])
                return latest_run

        logger.info("Step 5: Training XGBoost Model...")
        trainer = ModelTrainer(tune_hyperparams=False)
        results = trainer.train(feature_df, trade_log)
        
        # Save model copy
        if Path(results["model_path"]).exists():
            shutil.copy2(results["model_path"], model_copy_path)
            logger.info("Saved trained model copy to {}", model_copy_path)
            
        # Save reports
        metrics = results.get("overall_metrics", {})
        with open(eval_report_json, "w") as f:
            json.dump(metrics, f, indent=4)
            
        fold_results = results.get("fold_results", [])
        if fold_results:
            pd.DataFrame(fold_results).to_csv(wf_results_csv, index=False)
            
        xgb_imp = results.get("xgb_importance")
        if xgb_imp is not None:
            xgb_imp.to_csv(feat_imp_csv, index=False)
            
        shap_imp = results.get("shap_importance")
        if shap_imp is not None:
            shap_imp.to_csv(shap_rep_csv, index=False)
            
        logger.info("Saved model evaluation metrics and training reports to disk.")
        
        # Pre-render charts
        if xgb_imp is not None:
            self._save_chart(feature_importance_chart(xgb_imp, top_n=25), "model_feature_importance")
        if shap_imp is not None:
            self._save_chart(shap_summary_chart(shap_imp, top_n=20), "model_shap_summary")
        if fold_results:
            self._save_chart(walk_forward_chart(fold_results, "sharpe", "Sharpe by Fold"), "wf_sharpe")
            self._save_chart(walk_forward_chart(fold_results, "mae", "MAE by Fold"), "wf_mae")
            self._save_chart(walk_forward_chart(fold_results, "win_rate", "Win Rate by Fold"), "wf_win_rate")
            
        return results

    def _step_predictions(self, feature_df: pd.DataFrame, model_results: dict) -> None:
        with self.db.session() as sess:
            pred_count = sess.query(Prediction).count()
        predictions_csv = self.datasets_dir / "predictions.csv"
            
        if pred_count > 0 and predictions_csv.exists() and not self.force:
            logger.info("Step 6: Predictions already present in database and saved on disk. Skipping predictions.")
            return

        logger.info("Step 6: Generating predictions...")
        predictor = PredictionEngine(model_path=model_results["model_path"])
        predictor.load()
        
        recent_df = feature_df.tail(1000)
        preds = predictor.predict(recent_df)
        
        # Save predictions to DB
        pred_records = []
        for ts, val in preds.items():
            pred_records.append(Prediction(
                run_id=model_results["run_id"],
                timestamp=ts.to_pydatetime().replace(tzinfo=None),
                predicted_pnl=float(val),
            ))
            
        with self.db.session() as sess:
            sess.query(Prediction).delete()
            sess.bulk_save_objects(pred_records)
            
        # Save to disk
        pred_df = pd.DataFrame([{
            "timestamp": ts, "predicted_pnl": float(val)
        } for ts, val in preds.items()])
        pred_df.to_csv(predictions_csv, index=False)
        
        logger.success("Saved {} predictions to database and disk", len(pred_records))
        
        # Pre-render charts
        import numpy as np
        all_actuals = model_results.get("all_actuals", [])
        all_preds = model_results.get("all_predictions", [])
        if all_preds and all_actuals:
            self._save_chart(prediction_vs_actual_chart(all_actuals, all_preds), "prediction_vs_actual")
            
            # Residual chart
            import plotly.graph_objects as go
            from src.visualization.themes import COLORS
            residuals = np.array(all_actuals) - np.array(all_preds)
            fig_resid = go.Figure(go.Scatter(
                x=np.array(all_actuals),
                y=residuals,
                mode="markers",
                marker=dict(color=COLORS["electric_blue"], size=3, opacity=0.4),
                name="Residuals",
            ))
            fig_resid.add_hline(y=0, line_dash="dash", line_color=COLORS["coral"])
            fig_resid.update_layout(
                template="quanttrade", height=380,
                title="Residuals vs Actual",
                xaxis_title="Actual PnL", yaxis_title="Residual",
            )
            self._save_chart(fig_resid, "residuals_vs_actual")
            
        self._save_chart(return_distribution_chart(pred_df["predicted_pnl"], "Predicted PnL Distribution"), "predicted_returns_dist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantTrade Pipeline Runner")
    parser.add_argument("--force", action="store_true", help="Force rebuild of all pipeline stages")
    args = parser.parse_args()
    
    runner = PipelineRunner(force=args.force)
    runner.run()
