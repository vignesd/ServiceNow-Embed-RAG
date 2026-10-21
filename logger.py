import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import sys

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure centralized logging with optional console and file outputs driven by .env.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging if called multiple times
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Define standard log format
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (Optional based on .env flag ENABLE_CONSOLE_LOGGING)
    enable_console_logging = os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() in {
        "true", "1", "yes", "on"
    }

    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)

    # 2. File Handler (Optional based on .env flag ENABLE_FILE_LOGGING)
    enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", "false").lower() in {
        "true", "1", "yes", "on"
    }

    if enable_file_logging:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
        # Note: Avoid calling root_logger.info here before handlers are fully set or if console is disabled,
        # but since handlers are configured above, file handler will catch this if enabled.


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
