import logging
import os
from pathlib import Path


def setup_logging(
    log_file: str | None = None, level: int = logging.DEBUG
) -> logging.Logger:
    """Configure root logger to write to a file only (no console handlers).

    Returns the root logger for convenience. Default log file is
    logs/create_dataset.log (relative to package folder) or
    overridden with LOG_FILE env var.
    """
    if log_file is None:
        log_file = os.getenv(
            "LOG_FILE",
            "logs/create_dataset.log",
        )
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers to avoid duplicate writes
    logger = logging.getLogger()
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    logger.addHandler(fh)
    logger.setLevel(level)

    # Do not add StreamHandler; console output will still show due to prints
    return logger
