# coding = utf-8

"""see documentation @ ../../docs/dynamics.md"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..reader.reader_utils import Snapshots
from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


def time_correlation(
    snapshots: Snapshots,
    condition: npt.NDArray,
    dt: float = 0.002,
    outputfile: str = "",
) -> pd.DataFrame:
    """
    Calculate the time correlation of the input property given by condition
    There are three cases considered, given by the shape of condition:
    1. condition is float scalar type, for example, density
    2. condition is float vector type, for example, velocity
    3. condition is float tensor type, for example, nematic order

    Input:
        1. snapshots (read.reader_utils.snapshots): multiple trajectories dumped linearly or in logscale
        2. condition (npt.NDArray): particle-level condition / property, type should be float
                                   shape: [num_of_snapshots, num_of_particles, xxx]
        3. dt (float): time step of input snapshots, default 0.002
        4. outputfile (str): output file name, default "" (None)

    Return:
        calculated time-correlation information in pandas dataframe
    """

    timesteps = np.array([snapshot.timestep for snapshot in snapshots.snapshots])
    if len(set(np.diff(timesteps))) == 1:
        # input configuration dumped in linear interval
        # ensemble average available
        cal_type = "linear"
    else:
        # input configuration dumped in log scale
        # ensemble average OFF
        cal_type = "log"
    logger.info(f"Calculate time correlation of input property in {cal_type} style")

    results = np.zeros(snapshots.nsnapshots)
    counts = np.zeros_like(results)

    if len(condition.shape) == 2:
        # input condition is float or complex-number scalar
        # bool should be converted externally to 0 or 1
        if cal_type == "linear":
            for n in range(snapshots.nsnapshots):
                for nn in range(n + 1):
                    # sum over particles
                    results[nn] += (condition[n] * np.conj(condition[n - nn])).sum().real
                    counts[nn] += 1
            results /= counts
        else:
            results = (np.conj(condition[0][np.newaxis, :]) * condition).sum(axis=1).real
    elif len(condition.shape) == 3:
        # input condition is float or complex-number vector
        if cal_type == "linear":
            for n in range(snapshots.nsnapshots):
                for nn in range(n + 1):
                    # sum over particles and dimensionality, i.e. [x, y, z]
                    results[nn] += (condition[n] * np.conj(condition[n - nn])).sum().real
                    counts[nn] += 1
            results /= counts
        else:
            for n in range(snapshots.nsnapshots):
                results[n] = (condition[n] * np.conj(condition[0])).sum().real
    elif len(condition.shape) == 4:
        # input condition is float or complex-number tensor
        if cal_type == "linear":
            for n in range(snapshots.nsnapshots):
                for nn in range(n + 1):
                    for i in range(snapshots.snapshots[0].nparticle):
                        results[nn] += np.trace(np.matmul(condition[n, i], np.conj(condition[n - nn, i])))
                        counts[nn] += 1
            results /= counts
        else:
            for n in range(snapshots.nsnapshots):
                for i in range(snapshots.snapshots[0].nparticle):
                    results[n] += np.trace(np.matmul(condition[n, i], np.conj(condition[0, i])))
    else:
        raise ValueError("WRONG input condition")

    results /= results[0]
    results = np.column_stack(((timesteps - timesteps[0]) * dt, results))
    results = pd.DataFrame(results, columns="t time_corr".split())
    if outputfile:
        results.to_csv(outputfile, float_format="%.8f", index=False)
    logger.info("Finish calculating time correlation of input conditional property")
    return results
