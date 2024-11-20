"""
This module provides utils functions to logging data
"""

import logging


def get_logger_handle(name: str, level=logging.INFO) -> logging.Logger:
    """
    Construct a logger handle based on name and logging level and return the handle.
    """
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
