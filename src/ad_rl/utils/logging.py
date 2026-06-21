"""Project-wide logging with a Rich handler when available."""

from __future__ import annotations

import logging

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        fmt = "%(message)s"
    except ImportError:  # pragma: no cover - rich is a core dep but stay robust
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[handler])
    _CONFIGURED = True


def get_logger(name: str = "ad_rl") -> logging.Logger:
    """Return a configured logger.

    Idempotent: the root configuration is only installed once per process.
    """
    _configure_root()
    return logging.getLogger(name)


__all__ = ["get_logger"]
