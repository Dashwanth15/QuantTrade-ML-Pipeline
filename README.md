# QuantTrade ML Pipeline

Production-grade quantitative intelligence and ML-driven trading pipeline.

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![Framework](https://img.shields.io/badge/dashboard-Streamlit-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📷 Dashboard

> 📷 Screenshot coming soon

---

## Quick Highlights

- **Scale:** 93,000+ hourly EUR/USD candles (2005–2020) with live macroeconomic data scraping.
- **Robustness:** Walk-forward validation with embargo to prevent lookahead bias.
- **Explainability:** Full SHAP integration for model transparency.
- **Optimization:** Automated hyperparameter tuning using Optuna TPE.
- **Analytics:** 12+ interactive Plotly dashboards decoupled from execution logic.

---

## Features

### Data Engineering
- Automated Apify scraping for macroeconomic events
- Robust SQLite persistence layer with SQLAlchemy ORM
- Strict data validation and anomaly detection

### Feature Engineering
- **Time Features:** Session encoding, cyclical hours/days
- **Price Features:** Volatility windows, returns, spreads
- **Technical Indicators:** Bollinger Bands, EMA, SMA
- **Macro Features:** Impact scoring, event proximity

### Machine Learning
- XGBoost Regressor for PnL prediction
- 90/30 day walk-forward splits
- Optuna hyperparameter optimization
- SHAP feature importance tracking

### Trading Simulation
- 7 distinct quantitative trading strategies
- Dynamic risk management and position sizing
- Detailed trade logging and PnL calculation

### Dashboard
- Fully interactive Streamlit interface
- Pre-rendered Plotly visualizations
- Graceful missing-data handling
- Strategy radar and rolling metrics

---

## Architecture Diagram

> 📷 Architecture Diagram coming soon

---

## Tech Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.14 |
| **Machine Learning** | XGBoost, Scikit-learn, Optuna, SHAP |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLite, SQLAlchemy |
| **Visualization** | Streamlit, Plotly Express |
| **Ingestion** | Apify Client |

---

## Project Workflow

1. **Ingest:** Load historical forex data and scrape macro events.
2. **Preprocess:** Validate, clean, and align datasets.
3. **Feature Build:** Generate 60+ engineered features.
4. **Simulate:** Run 7 professional strategies to generate trade logs.
5. **Train:** Optimize and train XGBoost models using walk-forward validation.
6. **Evaluate:** Generate SHAP explanations and strategy recommendations.
7. **Visualize:** Explore insights via the Streamlit dashboard.

---

## Installation

```bash
git clone https://github.com/Dashwanth15/QuantTrade-ML-Pipeline.git
cd QuantTrade-ML-Pipeline

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

**Execute Backend Pipeline:**
```bash
python scripts/run_pipeline.py
```

**Launch Analytics Dashboard:**
```bash
streamlit run app/main.py
```

---

## Project Structure

- `app/` — Streamlit dashboard and pages.
- `config/` — Environment variables and logging setup.
- `data/` — SQLite databases, datasets, and generated artifacts.
- `scripts/` — CLI execution scripts.
- `src/` — Core backend modules (ingestion, ML, features, simulation).
- `tests/` — Pytest unit and integration suites.

---

## Future Improvements

- Add support for high-frequency tick data.
- Integrate deep learning models (LSTMs/Transformers).
- Deploy dashboard via Docker and AWS ECS.
- Expand macro scraping to more economic calendars.

---

## License

This project is licensed under the MIT License.
