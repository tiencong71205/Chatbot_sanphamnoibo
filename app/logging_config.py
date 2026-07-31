"""Structured logging configuration."""
from __future__ import annotations
import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(log_level)
    logger_name = name or "vhomenex"
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
