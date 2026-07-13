"""
QuantTrade ML Pipeline — Structured Logging Configuration
Uses loguru for production-grade logging with rotation, JSON support,
and colored console output for development.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "colored",
    log_path: Path | None = None,
) -> None:
    """
    Configure loguru for the application.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: "colored" for dev, "json" for production
        log_path: Directory to write rotating log files (optional)
    """
    # Remove default loguru handler
    logger.remove()

    # ------------------------------------------------------------------ #
    # Console Handler
    # ------------------------------------------------------------------ #
    if log_format == "json":
        console_fmt = (
            '{{"time":"{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level":"{level}", '
            '"name":"{name}", '
            '"function":"{function}", '
            '"line":{line}, '
            '"message":"{message}"}}'
        )
        colorize = False
    else:
        console_fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        colorize = True

    logger.add(
        sys.stderr,
        format=console_fmt,
        level=log_level,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
    )

    # ------------------------------------------------------------------ #
    # File Handler (rotating)
    # ------------------------------------------------------------------ #
    if log_path is not None:
        log_path = Path(log_path)
        log_path.mkdir(parents=True, exist_ok=True)

        # Application log — rotates daily, kept 30 days
        logger.add(
            str(log_path / "quanttrade_{time:YYYY-MM-DD}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation="00:00",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=False,
            enqueue=True,  # Thread-safe async logging
        )

        # Error log — kept 90 days
        logger.add(
            str(log_path / "errors_{time:YYYY-MM-DD}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
            level="ERROR",
            rotation="00:00",
            retention="90 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )

    logger.info(
        "Logging initialized | level={level} | format={fmt}",
        level=log_level,
        fmt=log_format,
    )


def get_logger(name: str):
    """Return a bound logger for a module."""
    return logger.bind(name=name)
