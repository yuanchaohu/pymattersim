"""
This module provides utils functions to logging data
"""

import logging
import pandas as pd

from tabulate import tabulate


def get_logger_handle(name: str, level=logging.INFO) -> logging.Logger:
    """
    Construct a logger handle based on name and logging level and return the handle.
    """
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def log_table(
    pd_df: pd.DataFrame, logger: logging.Logger, round_digits: int = 3
) -> None:
    """
    Get pretty table format logging of a dataframe
    """
    pretty_df = tabulate(
        pd_df.round(round_digits),
        headers="keys",
        tablefmt="psql")
    logger.info(pretty_df)
