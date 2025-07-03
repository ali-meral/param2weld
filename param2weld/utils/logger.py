import logging
import sys


def get_logger(name: str = "param2weld") -> logging.Logger:
    """
    Create and configure a logger instance.

    Parameters
    ----------
    name : str
        Name for the logger (e.g., module or class name).

    Returns
    -------
    logging.Logger
        Configured logger with stream handler.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
