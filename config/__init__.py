"""QuantTrade ML Pipeline — Config package."""
from .logging_config import setup_logging
from .settings import settings

__all__ = ["settings", "setup_logging"]
