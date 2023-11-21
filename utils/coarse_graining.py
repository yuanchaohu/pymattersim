# coding = utf-8

"""
see documentation @ ../docs/utils.md
"""

from typing import Tuple
import numpy as np
from reader.reader_utils import Snapshots
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

def time_average(
        snapshots: Snapshots,
        input_property: np.ndarray,
        time_period: float=0.0,
        dt: float=0.002
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate time average of the input property

    Input:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. input_property (np.ndarray): the input particle-level property,
                          in np.ndarray with shape [nsnapshots, nparticle]
        3. time_period (float): time used to average, default 0.0
        4. dt (float): timestep used in user simulations, default 0.002

    Return:
        1. Calculated time averaged input results (np.ndarray)
           shape [nsnapshots_updated, nparticles]
        2. Corresponding snapshot id of the middle snapshot of each time period
           shape [nsnapshots_updated]
    """
    logger.info(f"Time average of the input variables for time_period={time_period}")
    time_interval = snapshots.snapshots[1].timestep - snapshots.snapshots[0].timestep
    time_interval *= dt
    time_nsnapshot = int(time_period/time_interval)
    # save the time averaged results
    results = np.zeros((snapshots.nsnapshots-time_nsnapshot, snapshots.snapshots[0].nparticle))
    # save the middle snapshot id for each time average period
    results_middle_snapshots = []

    for n in range(results.shape[0]):
        results[n, :] = input_property[n: n+time_nsnapshot].mean(axis=0)
        results_middle_snapshots.append(round(n+time_nsnapshot/2))
    return results, np.array(results_middle_snapshots)

def spatial_average():
    pass

def gaussian_blurring():
    pass

def atomic_position_average():
    pass