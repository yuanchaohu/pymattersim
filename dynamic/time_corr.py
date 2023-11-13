# coding = utf-8

"""This module calc"""

import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

def time_correlation(
    snapshots: Snapshots,
    condition: np.ndarray,
    dt: float=0.002,
    outputfile: str=None
) -> np.ndarray:
    """calculate the time correlation of the condition"""

    logger.info("Calculate time correlation of input conditional property")
    results = np.zeros((snapshots.nsnapshots-1, 2))
    counts = np.zeros(snapshots.nsnapshots-1)
    timesteps = np.array([snapshot.timestep for snapshot in snapshots.snapshots])
    results[:, 0] = (timesteps[1:] - timesteps[0]) * dt

    # linear output -- moving average
    if len(set(np.diff(timesteps)))==1:
        for n in range(snapshots.nsnapshots-1):
            for nn in range(n+1):
                results[nn] += (condition[n] * np.conj(condition[n-nn])).sum().real
                counts[nn] += 1
        results[:, 1] /= counts

    # log output -- single calculation
    else:
        for n in range(1, snapshots.nsnapshots):
            results[n-1] = (condition[n]*np.conj(condition[0])).sum().real

    results = pd.DataFrame(results, columns="t Ct".split())
    if outputfile:
        results.to_csv(outputfile, float_format="%.6f", index=False)
    logger.info("Finish calculating time correlation of input conditional property")
    return results