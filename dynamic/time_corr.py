# coding = utf-8

"""see xxx"""

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
    results = np.zeros(snapshots.nsnapshots)
    counts = np.zeros(snapshots.nsnapshots)
    timesteps = np.array([snapshot.timestep for snapshot in snapshots.snapshots])

    # linear output -- moving average
    if len(set(np.diff(timesteps)))==1:
        for n in range(snapshots.nsnapshots):
            for nn in range(n+1):
                results[nn] += (condition[n] * np.conj(condition[n-nn])).sum().real
                counts[nn] += 1
        results /= counts

    # log output -- single calculation
    else:
        for n in range(snapshots.nsnapshots):
            results[n] = (condition[n] * np.conj(condition[0])).sum().real

    results = np.column_stack(((timesteps - timesteps[0])*dt, results))
    results = pd.DataFrame(results, columns="t time_corr".split())
    if outputfile:
        results.to_csv(outputfile, float_format="%.6f", index=False)
    logger.info("Finish calculating time correlation of input conditional property")
    return results