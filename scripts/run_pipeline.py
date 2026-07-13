"""
QuantTrade ML Pipeline — Pipeline Runner Script
Executes the full end-to-end pipeline from CLI (no Streamlit required).
Useful for scheduled runs or CI/CD testing.

Usage:
    python scripts/run_pipeline.py --help
    python scripts/run_pipeline.py --no-tune --strategies momentum ma_crossover
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging
from config.settings import settings
from src.ingestion.forex_loader import load_forex_data
from src.preprocessing.cleaner import clean_forex_data
from src.preprocessing.validator import validate_forex_data
from src.features.feature_store import FeatureStore
from src.simulation.engine import run_simulation, STRATEGY_REGISTRY
from src.ml.trainer import ModelTrainer
from src.database.repository import get_repos
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantTrade ML Pipeline Runner")
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna tuning")
    parser.add_argument("--no-macro", action="store_true", help="Skip macro scraping")
    parser.add_argument("--use-cache", action="store_true", help="Use cached feature files")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGY_REGISTRY.keys()),
                        choices=list(STRATEGY_REGISTRY.keys()), help="Strategies to run")
    parser.add_argument("--train-days", type=int, default=settings.wf_train_days)
    parser.add_argument("--test-days", type=int, default=settings.wf_test_days)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(settings.log_level, settings.log_format, settings.log_path)

    logger.info("=" * 60)
    logger.info("QuantTrade ML Pipeline — Full Run")
    logger.info("=" * 60)

    # ---- Step 1: Load & Validate Market Data ----
    logger.info("Step 1: Loading EUR/USD market data")
    raw_df = load_forex_data()
    cleaned_df, cleaning_report = clean_forex_data(raw_df)
    validation = validate_forex_data(cleaned_df)

    if not validation.passed:
        logger.error("Data validation failed! Review errors before proceeding.")
        for check in validation.checks:
            if not check.passed and check.severity == "critical":
                logger.error("  CRITICAL: {} — {}", check.name, check.message)
        sys.exit(1)

    logger.success("Data validated: {} bars ({} → {})",
                   len(cleaned_df), cleaned_df.index[0].date(), cleaned_df.index[-1].date())

    # ---- Step 2: Macro Events ----
    macro_df = None
    if not args.no_macro:
        logger.info("Step 2: Scraping macro events")
        try:
            from src.ingestion.macro_scraper import scrape_macro_events
            macro_df = scrape_macro_events(api_key=settings.apify_api_key)
            logger.success("Scraped {} macro events", len(macro_df))
        except Exception as e:
            logger.warning("Macro scraping failed: {} — continuing without macro features", e)

    # ---- Step 3: Feature Engineering ----
    logger.info("Step 3: Building feature matrix")
    store = FeatureStore()
    feature_df = store.build_features(cleaned_df, macro_df, use_cache=args.use_cache)
    logger.success("Features: {} samples × {} columns", len(feature_df), len(feature_df.columns))

    # ---- Step 4: Trade Simulation ----
    logger.info("Step 4: Running trade simulation | strategies={}", args.strategies)
    trade_log, strategy_summary = run_simulation(feature_df, strategies=args.strategies)
    logger.success("Simulation: {} trades | win_rate={:.1%}",
                   len(trade_log), trade_log["win"].mean() if len(trade_log) > 0 else 0)

    print("\n" + strategy_summary.to_string())

    # ---- Step 5: ML Training ----
    logger.info("Step 5: Training XGBoost model")
    trainer = ModelTrainer(tune_hyperparams=not args.no_tune)
    results = trainer.train(feature_df, trade_log)

    metrics = results["overall_metrics"]
    logger.success("Training complete!")
    logger.success("  Sharpe: {:.3f} | MAE: {:.4f} | Win Rate: {:.1%}",
                   metrics["sharpe"], metrics["mae"], metrics["win_rate"])
    logger.success("  Model saved: {}", results["model_path"])

    # ---- Step 6: Persist to Database ----
    logger.info("Step 6: Saving to database")
    forex_repo, macro_repo, trade_repo, model_repo = get_repos()
    forex_repo.bulk_insert(cleaned_df)
    if macro_df is not None:
        macro_repo.bulk_insert(macro_df)
    trade_repo.bulk_insert(trade_log)
    model_repo.save_run({
        "run_id": results["run_id"],
        "target_variable": "pnl_usd",
        "n_features": results["n_features"],
        "n_train_samples": results["n_samples"],
        "n_folds": results["n_folds"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "sharpe": metrics["sharpe"],
        "win_rate": metrics["win_rate"],
        "best_params": results["best_params"],
        "model_path": results["model_path"],
        "status": "completed",
    })

    logger.success("=" * 60)
    logger.success("Pipeline complete! Run ID: {}", results["run_id"])
    logger.success("Launch dashboard: streamlit run app/main.py")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
