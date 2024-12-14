# coding = utf-8
"""see documentation @ ../../docs/reader.md"""

import numpy as np
import pandas as pd

from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)


def read_lammpslog(filename) -> [pd.DataFrame]:
    """
    Extract the thermodynamic quantities from lammp log file

    Input:
        filename (str): lammps log file name

    Return:
        list of pandas dataframe for each logging section
    """

    with open(filename, "r", encoding="utf-8") as f:
        data = f.readlines()

    # ----get how many sections are there----
    start = [i for i, val in enumerate(data) if val.lstrip().startswith("Step ")]
    end = [i for i, val in enumerate(data) if val.lstrip().startswith("Loop time of ")]

    if data[-1] != "\n":
        if data[-1].split()[0].isnumeric():  # incomplete log file
            end.append(len(data) - 2)

    start = np.array(start)
    end = np.array(end)
    linenum = end - start - 1
    logger.info(f"Section Number: {len(linenum)} \t Line Numbers: {str(linenum)}")
    del data

    final = []
    for i in range(linenum.shape[0]):
        data = pd.read_csv(filename, sep=r"\s+", skiprows=start[i], nrows=linenum[i])
        final.append(data)
        del data
    return final
