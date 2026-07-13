"""
QuantTrade ML Pipeline — Data Validator
Runs comprehensive data integrity checks on the cleaned DataFrame
and generates a detailed validation report.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ValidationResult:
    """Container for a single validation check result."""
    name: str
    passed: bool
    message: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Full validation report for a dataset."""
    results: list[ValidationResult] = field(default_factory=list)
    dataset_shape: tuple[int, int] = (0, 0)
    date_range: tuple[str, str] = ("", "")

    @property
    def passed(self) -> bool:
        return all(r.passed or r.severity != "ERROR" for r in self.results)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "WARNING")

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Shape     : {self.dataset_shape[0]:,} rows × {self.dataset_shape[1]} cols",
            f"Date range: {self.date_range[0]} → {self.date_range[1]}",
            f"Status    : {'✅ PASSED' if self.passed else '❌ FAILED'}",
            f"Errors    : {self.error_count}",
            f"Warnings  : {self.warning_count}",
            "-" * 60,
        ]
        for r in self.results:
            icon = "✅" if r.passed else ("⚠️" if r.severity == "WARNING" else "❌")
            lines.append(f"{icon} {r.name}: {r.message}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "dataset_shape": self.dataset_shape,
            "date_range": self.date_range,
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


class DataValidator:
    """
    Runs a comprehensive suite of validation checks on the EUR/USD DataFrame.
    All checks are non-destructive — they report findings without modifying data.
    """

    REQUIRED_COLUMNS = [
        "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
        "mid_open", "mid_high", "mid_low", "mid_close",
        "spread", "spread_pips",
    ]

    MIN_ROWS = 5_000
    MAX_NULL_RATE = 0.01  # 1% max null rate
    MIN_DATE = pd.Timestamp("2000-01-01", tz="UTC")
    MAX_DATE = pd.Timestamp("2030-12-31", tz="UTC")
    EXPECTED_FREQUENCY = "H"  # Hourly

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run all validation checks and return a report."""
        report = ValidationReport()
        report.dataset_shape = df.shape
        report.date_range = (
            str(df.index[0])[:19] if len(df) > 0 else "",
            str(df.index[-1])[:19] if len(df) > 0 else "",
        )

        checks = [
            self._check_minimum_rows,
            self._check_required_columns,
            self._check_index_type,
            self._check_monotonic_index,
            self._check_date_range,
            self._check_null_rate,
            self._check_ohlc_integrity,
            self._check_spread_positivity,
            self._check_price_bounds,
            self._check_return_stationarity,
            self._check_duplicate_timestamps,
            self._check_data_continuity,
        ]

        for check_fn in checks:
            try:
                result = check_fn(df)
                report.results.append(result)
                if result.passed:
                    logger.debug("✅ {}", result.name)
                elif result.severity == "WARNING":
                    logger.warning("⚠️  {} | {}", result.name, result.message)
                else:
                    logger.error("❌ {} | {}", result.name, result.message)
            except Exception as exc:
                report.results.append(ValidationResult(
                    name=str(check_fn.__name__),
                    passed=False,
                    message=f"Check raised exception: {exc}",
                    severity="ERROR",
                ))

        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------ #
    # Individual Checks
    # ------------------------------------------------------------------ #

    def _check_minimum_rows(self, df: pd.DataFrame) -> ValidationResult:
        passed = len(df) >= self.MIN_ROWS
        return ValidationResult(
            name="Minimum Row Count",
            passed=passed,
            message=f"{len(df):,} rows (min: {self.MIN_ROWS:,})",
            severity="ERROR",
            details={"row_count": len(df)},
        )

    def _check_required_columns(self, df: pd.DataFrame) -> ValidationResult:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        return ValidationResult(
            name="Required Columns",
            passed=len(missing) == 0,
            message=f"Missing: {missing}" if missing else "All required columns present",
            severity="ERROR",
            details={"missing_columns": missing},
        )

    def _check_index_type(self, df: pd.DataFrame) -> ValidationResult:
        is_dt = isinstance(df.index, pd.DatetimeIndex)
        is_utc = hasattr(df.index, "tzinfo") and str(df.index.tzinfo) in ("UTC", "pytz.UTC")
        passed = is_dt and is_utc
        return ValidationResult(
            name="UTC DatetimeIndex",
            passed=passed,
            message=f"Index type: {type(df.index).__name__}, tz={getattr(df.index, 'tzinfo', 'None')}",
            severity="ERROR",
        )

    def _check_monotonic_index(self, df: pd.DataFrame) -> ValidationResult:
        passed = df.index.is_monotonic_increasing
        return ValidationResult(
            name="Monotonic Timestamps",
            passed=passed,
            message="Timestamps are strictly increasing" if passed else "Non-monotonic timestamps detected",
            severity="ERROR",
        )

    def _check_date_range(self, df: pd.DataFrame) -> ValidationResult:
        if len(df) == 0:
            return ValidationResult("Date Range", False, "Empty DataFrame", "ERROR")
        start = df.index[0]
        end = df.index[-1]
        passed = self.MIN_DATE <= start and end <= self.MAX_DATE
        span_years = (end - start).days / 365.25
        return ValidationResult(
            name="Date Range",
            passed=passed,
            message=f"Range: {start.date()} to {end.date()} ({span_years:.1f} years)",
            severity="WARNING",
            details={"start": str(start), "end": str(end), "span_years": span_years},
        )

    def _check_null_rate(self, df: pd.DataFrame) -> ValidationResult:
        price_cols = [c for c in self.REQUIRED_COLUMNS if c in df.columns]
        null_rates = df[price_cols].isnull().mean()
        max_null_rate = null_rates.max()
        worst_col = null_rates.idxmax() if len(null_rates) > 0 else "N/A"
        passed = max_null_rate <= self.MAX_NULL_RATE
        return ValidationResult(
            name="Null Rate",
            passed=passed,
            message=f"Max null rate: {max_null_rate:.4%} in column '{worst_col}'",
            severity="ERROR" if max_null_rate > 0.05 else "WARNING",
            details={"null_rates": null_rates.to_dict()},
        )

    def _check_ohlc_integrity(self, df: pd.DataFrame) -> ValidationResult:
        violations = 0
        for pfx in ["bid", "ask", "mid"]:
            cols = [f"{pfx}_{x}" for x in ["open", "high", "low", "close"]]
            if not all(c in df.columns for c in cols):
                continue
            o, h, l, c = [df[col] for col in cols]
            violations += ((h < o) | (h < c)).sum()
            violations += ((l > o) | (l > c)).sum()

        passed = violations == 0
        return ValidationResult(
            name="OHLC Integrity",
            passed=passed,
            message=f"{violations} OHLC constraint violations" if not passed else "All OHLC constraints satisfied",
            severity="WARNING",
            details={"violation_count": int(violations)},
        )

    def _check_spread_positivity(self, df: pd.DataFrame) -> ValidationResult:
        if "spread" not in df.columns:
            return ValidationResult("Spread Positivity", False, "spread column missing", "WARNING")
        negative_spreads = (df["spread"] < 0).sum()
        passed = negative_spreads == 0
        return ValidationResult(
            name="Spread Positivity",
            passed=passed,
            message=f"{negative_spreads} negative spreads detected",
            severity="ERROR",
            details={"negative_spread_count": int(negative_spreads)},
        )

    def _check_price_bounds(self, df: pd.DataFrame) -> ValidationResult:
        if "mid_close" not in df.columns:
            return ValidationResult("Price Bounds", True, "N/A (mid_close missing)", "INFO")
        prices = df["mid_close"]
        out_of_bounds = ((prices < 0.8) | (prices > 2.0)).sum()
        passed = out_of_bounds == 0
        return ValidationResult(
            name="Price Bounds (EUR/USD)",
            passed=passed,
            message=f"{out_of_bounds} prices outside historical bounds [0.8, 2.0]",
            severity="WARNING",
            details={"min": float(prices.min()), "max": float(prices.max()), "out_of_bounds": int(out_of_bounds)},
        )

    def _check_return_stationarity(self, df: pd.DataFrame) -> ValidationResult:
        """Check that hourly returns look stationary (mean near 0, finite std)."""
        if "mid_close" not in df.columns:
            return ValidationResult("Return Stationarity", True, "N/A", "INFO")
        returns = df["mid_close"].pct_change().dropna()
        mean_r = returns.mean()
        std_r = returns.std()
        skew = float(returns.skew())
        passed = abs(mean_r) < 0.001 and std_r < 0.05 and np.isfinite(std_r)
        return ValidationResult(
            name="Return Stationarity",
            passed=passed,
            message=f"Returns: mean={mean_r:.6f}, std={std_r:.6f}, skew={skew:.2f}",
            severity="WARNING",
            details={"mean": float(mean_r), "std": float(std_r), "skew": skew},
        )

    def _check_duplicate_timestamps(self, df: pd.DataFrame) -> ValidationResult:
        n_dups = df.index.duplicated().sum()
        passed = n_dups == 0
        return ValidationResult(
            name="Duplicate Timestamps",
            passed=passed,
            message=f"{n_dups} duplicate timestamps" if not passed else "No duplicate timestamps",
            severity="ERROR",
            details={"duplicate_count": int(n_dups)},
        )

    def _check_data_continuity(self, df: pd.DataFrame) -> ValidationResult:
        """Check for unexpectedly large gaps (excluding weekends)."""
        time_diffs = df.index.to_series().diff().dt.total_seconds() / 3600
        # Gaps > 2h that are NOT weekend gaps (> 30h) are suspicious
        weekday_gaps = time_diffs[(time_diffs > 2) & (time_diffs < 30)]
        gap_count = len(weekday_gaps)
        max_gap = float(weekday_gaps.max()) if gap_count > 0 else 0.0
        passed = gap_count < 100  # Allow some gaps for holidays etc.
        return ValidationResult(
            name="Data Continuity",
            passed=passed,
            message=f"{gap_count} weekday gaps >2h detected (max: {max_gap:.1f}h)",
            severity="WARNING",
            details={"gap_count": gap_count, "max_gap_hours": max_gap},
        )


def validate_forex_data(df: pd.DataFrame) -> ValidationReport:
    """Validate a forex DataFrame and return a report."""
    return DataValidator().validate(df)
