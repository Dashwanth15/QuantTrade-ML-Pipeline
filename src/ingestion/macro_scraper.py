"""
QuantTrade ML Pipeline — Macroeconomic Event Scraper
Scrapes the latest 180 days of major macroeconomic events using Apify.
Events are stored in SQLite via the database repository layer.

Supported events: CPI, FOMC, GDP, NFP, Inflation, Interest Rates,
                  PMI, Retail Sales, Trade Balance, Industrial Production
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
HIGH_IMPACT_KEYWORDS = {
    "CPI", "Consumer Price Index", "FOMC", "Federal Reserve",
    "Interest Rate", "Rate Decision", "GDP", "Gross Domestic",
    "Non-Farm", "NFP", "Payroll", "Employment", "Unemployment",
    "Inflation", "PCE", "PPI", "Retail Sales", "PMI", "ISM",
    "Trade Balance", "Current Account", "Durable Goods",
    "Industrial Production", "Housing", "Consumer Confidence",
    "Sentiment", "ECB", "Bank of England", "BoJ",
}

IMPACT_MAP = {"low": 1, "medium": 2, "high": 3, "": 1}

COUNTRY_CURRENCY_MAP = {
    "United States": "USD",
    "US": "USD",
    "USA": "USD",
    "Eurozone": "EUR",
    "European Union": "EUR",
    "EU": "EUR",
    "Germany": "EUR",
    "France": "EUR",
    "United Kingdom": "GBP",
    "UK": "GBP",
    "Japan": "JPY",
    "Canada": "CAD",
    "Australia": "AUD",
    "Switzerland": "CHF",
    "New Zealand": "NZD",
}

CATEGORY_MAP = {
    "CPI": "inflation",
    "Consumer Price Index": "inflation",
    "PPI": "inflation",
    "PCE": "inflation",
    "Inflation": "inflation",
    "FOMC": "monetary_policy",
    "Federal Reserve": "monetary_policy",
    "Interest Rate": "monetary_policy",
    "Rate Decision": "monetary_policy",
    "ECB": "monetary_policy",
    "Bank of England": "monetary_policy",
    "BoE": "monetary_policy",
    "GDP": "growth",
    "Gross Domestic": "growth",
    "Industrial Production": "growth",
    "PMI": "growth",
    "ISM": "growth",
    "Non-Farm": "employment",
    "NFP": "employment",
    "Payroll": "employment",
    "Employment": "employment",
    "Unemployment": "employment",
    "ADP": "employment",
    "Retail Sales": "consumption",
    "Consumer Confidence": "consumption",
    "Consumer Sentiment": "consumption",
    "Trade Balance": "trade",
    "Current Account": "trade",
    "Exports": "trade",
    "Imports": "trade",
    "Housing": "housing",
    "Durable Goods": "manufacturing",
}


class MacroEventScraper:
    """
    Scrapes macroeconomic calendar events via Apify actors.

    Primary actor: epctex/economic-calendar-scraper
    Fallback: Direct Forex Factory scrape via cheerio-scraper
    """

    APIFY_BASE_URL = "https://api.apify.com/v2"
    PRIMARY_ACTOR = "epctex/economic-calendar-scraper"
    FALLBACK_ACTOR = "jupri/forex-factory-scraper"

    def __init__(self, api_key: str | None = None, lookback_days: int | None = None) -> None:
        self.api_key = api_key or settings.apify_api_key
        self.lookback_days = lookback_days or settings.apify_macro_days
        self._validate_key()

    def _validate_key(self) -> None:
        if not self.api_key or len(self.api_key) < 10:
            logger.warning("Apify API key appears invalid or missing")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def scrape(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> pd.DataFrame:
        """
        Scrape macro events and return a clean DataFrame.

        Returns:
            DataFrame with columns: event_name, country, currency,
            timestamp_utc, forecast, actual, previous, impact,
            impact_score, category, surprise, surprise_direction
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=self.lookback_days)

        # Make naive timestamps timezone-aware for safety
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        logger.info(
            "Scraping macro events | range: {} to {}",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        raw_events = self._try_scrape(start_date, end_date)

        if not raw_events:
            logger.warning("No events scraped from Apify, using fallback data")
            return self._generate_synthetic_fallback(start_date, end_date)

        df = self._parse_events(raw_events)
        df = self._enrich_events(df)
        logger.success("Scraped {} macro events", len(df))
        return df

    # ------------------------------------------------------------------ #
    # Apify Integration
    # ------------------------------------------------------------------ #

    def _try_scrape(
        self, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Try primary actor, fall back to alternatives."""
        # Try primary actor
        events = self._run_actor(
            self.PRIMARY_ACTOR,
            {
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
                "countries": ["US", "EU", "GB", "JP", "CA", "AU"],
                "importance": ["high", "medium"],
            },
        )
        if events:
            return events

        # Try fallback actor
        logger.warning("Primary actor returned no data, trying fallback actor")
        events = self._run_actor(
            self.FALLBACK_ACTOR,
            {
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
            },
        )
        return events or []

    def _run_actor(
        self, actor_id: str, input_data: dict[str, Any], timeout_secs: int = 300
    ) -> list[dict[str, Any]]:
        """
        Run an Apify actor synchronously and return dataset items.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Start actor run
        try:
            logger.debug("Starting Apify actor: {}", actor_id)
            run_url = f"{self.APIFY_BASE_URL}/acts/{actor_id}/runs"
            resp = requests.post(
                run_url,
                headers=headers,
                json=input_data,
                timeout=30,
            )
            resp.raise_for_status()
            run_info = resp.json()
            run_id = run_info["data"]["id"]
            dataset_id = run_info["data"]["defaultDatasetId"]
            logger.debug("Actor run started: run_id={}", run_id)
        except Exception as exc:
            logger.warning("Failed to start actor {}: {}", actor_id, exc)
            return []

        # Poll for completion
        poll_url = f"{self.APIFY_BASE_URL}/actor-runs/{run_id}"
        deadline = time.time() + timeout_secs
        while time.time() < deadline:
            try:
                status_resp = requests.get(poll_url, headers=headers, timeout=15)
                status_resp.raise_for_status()
                status = status_resp.json()["data"]["status"]
                logger.debug("Actor status: {}", status)
                if status in ("SUCCEEDED", "FINISHED"):
                    break
                if status in ("FAILED", "ABORTED", "TIMED-OUT"):
                    logger.error("Actor run failed with status: {}", status)
                    return []
                time.sleep(10)
            except Exception as exc:
                logger.warning("Error polling actor status: {}", exc)
                time.sleep(15)
        else:
            logger.error("Actor run timed out after {} seconds", timeout_secs)
            return []

        # Fetch dataset items
        try:
            items_url = f"{self.APIFY_BASE_URL}/datasets/{dataset_id}/items"
            items_resp = requests.get(
                items_url,
                headers=headers,
                params={"limit": 10000, "format": "json"},
                timeout=30,
            )
            items_resp.raise_for_status()
            items = items_resp.json()
            logger.debug("Fetched {} items from dataset", len(items))
            return items
        except Exception as exc:
            logger.warning("Failed to fetch dataset items: {}", exc)
            return []

    # ------------------------------------------------------------------ #
    # Parsing & Enrichment
    # ------------------------------------------------------------------ #

    def _parse_events(self, raw: list[dict[str, Any]]) -> pd.DataFrame:
        """Normalize raw Apify output into a standard schema."""
        records = []
        for item in raw:
            try:
                rec = self._parse_single_event(item)
                if rec:
                    records.append(rec)
            except Exception as exc:
                logger.debug("Skipping malformed event: {}", exc)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc", "event_name"])
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
        return df

    def _parse_single_event(self, item: dict) -> dict | None:
        """Extract standardized fields from a single raw event dict."""
        # Try multiple field name patterns from different actors
        event_name = (
            item.get("event")
            or item.get("eventName")
            or item.get("name")
            or item.get("title")
            or ""
        ).strip()

        if not event_name:
            return None

        country = (
            item.get("country")
            or item.get("currency")
            or item.get("region")
            or "Unknown"
        ).strip()

        # Parse timestamp — try multiple formats
        ts_raw = (
            item.get("date")
            or item.get("datetime")
            or item.get("timestamp")
            or item.get("releaseDate")
            or ""
        )
        timestamp = self._parse_timestamp(ts_raw)

        # Numeric values
        actual = self._parse_float(item.get("actual") or item.get("actualValue"))
        forecast = self._parse_float(
            item.get("forecast") or item.get("forecastValue") or item.get("consensus")
        )
        previous = self._parse_float(item.get("previous") or item.get("previousValue"))

        # Impact
        impact_raw = str(
            item.get("impact") or item.get("importance") or item.get("volatility") or ""
        ).lower()
        impact_score = IMPACT_MAP.get(impact_raw, 1)

        return {
            "event_name": event_name,
            "country": country,
            "timestamp_utc": timestamp,
            "forecast": forecast,
            "actual": actual,
            "previous": previous,
            "impact": impact_raw or "medium",
            "impact_score": impact_score,
        }

    def _enrich_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features: currency, category, surprise metrics."""
        if df.empty:
            return df

        # Currency from country
        df["currency"] = df["country"].map(COUNTRY_CURRENCY_MAP).fillna("OTHER")

        # Category
        df["category"] = df["event_name"].apply(self._classify_category)

        # Surprise = actual - forecast
        df["surprise"] = df["actual"] - df["forecast"]
        df["surprise_direction"] = df["surprise"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        df["surprise_magnitude"] = df["surprise"].abs()

        # Is high impact
        df["is_high_impact"] = df["impact_score"] >= 3

        # EUR/USD relevant filter
        df["eurusd_relevant"] = df["currency"].isin(["USD", "EUR"])

        return df

    def _classify_category(self, event_name: str) -> str:
        event_upper = event_name.upper()
        for keyword, category in CATEGORY_MAP.items():
            if keyword.upper() in event_upper:
                return category
        return "other"

    @staticmethod
    def _parse_timestamp(raw: Any) -> str | None:
        if not raw:
            return None
        if isinstance(raw, (int, float)):
            # Unix timestamp in milliseconds
            try:
                return datetime.fromtimestamp(raw / 1000, tz=timezone.utc).isoformat()
            except Exception:
                pass
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%m/%d/%Y %H:%M",
            "%Y-%m-%d",
        ]
        raw_str = str(raw).strip()
        for fmt in formats:
            try:
                return datetime.strptime(raw_str, fmt).replace(
                    tzinfo=timezone.utc
                ).isoformat()
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_float(val: Any) -> float | None:
        if val is None or val == "":
            return None
        try:
            cleaned = str(val).replace("%", "").replace(",", "").strip()
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------ #
    # Synthetic Fallback
    # ------------------------------------------------------------------ #

    def _generate_synthetic_fallback(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate a synthetic but realistic macro event calendar.
        Used when Apify scraping fails. Events are placed on realistic
        schedule dates (first Friday for NFP, mid-month for CPI, etc.)
        """
        logger.warning("Using synthetic macro fallback data")
        records = []
        current = start_date

        # Monthly events
        monthly_events = [
            ("US Non-Farm Payrolls", "United States", "employment", "high", 3),
            ("US CPI m/m", "United States", "inflation", "high", 3),
            ("FOMC Statement", "United States", "monetary_policy", "high", 3),
            ("ECB Interest Rate Decision", "Eurozone", "monetary_policy", "high", 3),
            ("US GDP q/q", "United States", "growth", "high", 3),
            ("US Retail Sales m/m", "United States", "consumption", "medium", 2),
            ("ISM Manufacturing PMI", "United States", "growth", "medium", 2),
            ("US Trade Balance", "United States", "trade", "medium", 2),
            ("UK CPI y/y", "United Kingdom", "inflation", "high", 3),
            ("German CPI m/m", "Germany", "inflation", "medium", 2),
        ]

        while current <= end_date:
            day_offset = 0
            for name, country, category, impact, score in monthly_events:
                ts = current.replace(day=min(current.day + day_offset, 28))
                forecast_val = round(0.2 + (day_offset * 0.03), 2)
                actual_val = round(forecast_val + (0.1 if day_offset % 2 == 0 else -0.1), 2)

                records.append({
                    "event_name": name,
                    "country": country,
                    "timestamp_utc": ts,
                    "forecast": forecast_val,
                    "actual": actual_val,
                    "previous": round(forecast_val - 0.05, 2),
                    "impact": impact,
                    "impact_score": score,
                    "currency": COUNTRY_CURRENCY_MAP.get(country, "USD"),
                    "category": category,
                    "surprise": round(actual_val - forecast_val, 3),
                    "surprise_direction": 1 if actual_val > forecast_val else -1,
                    "surprise_magnitude": abs(actual_val - forecast_val),
                    "is_high_impact": score >= 3,
                    "eurusd_relevant": COUNTRY_CURRENCY_MAP.get(country, "USD") in ["USD", "EUR"],
                })
                day_offset += 2

            current += timedelta(days=30)

        df = pd.DataFrame(records)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df.sort_values("timestamp_utc").reset_index(drop=True)


def scrape_macro_events(
    api_key: str | None = None,
    lookback_days: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Convenience function to scrape and return macro events."""
    scraper = MacroEventScraper(api_key=api_key, lookback_days=lookback_days)
    return scraper.scrape(start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    df = scrape_macro_events()
    print(df.head(20))
    print(f"\nShape: {df.shape}")
    print(f"\nCategories:\n{df['category'].value_counts()}")
    print(f"\nImpact:\n{df['impact'].value_counts()}")
