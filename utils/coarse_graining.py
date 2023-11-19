# coding = utf-8

"""
see documentation @ ../docs/utils.md
"""

import numpy as np
from reader.reader_utils import Snapshots

def time_average(
        raw_props: np.ndarray,
        snapshots: Snapshots,
        avet: float=0.0,
        dt: float=0.002
) -> np.ndarray:
    """
    Calculate the time averaged input results

    Input:
        1. raw_props (np.ndarray): the raw property of particle,
                                   in np.ndarray with shape [nsnapshots, nparticle]
        3. avet (float): time used to average, default 0.0
        4. dt (float): timestep used in user simulations, default 0.002
    Return:
        Calculated time averaged input results (np.ndarray)
    """

    timestep_interval = np.diff([snapshot.timestep for snapshot in snapshots.snapshots])[0]
    assert len(set(np.diff(
        [snapshot.timestep for snapshot in snapshots.snapshots]
    ))) == 1, "Warning: Dump interval changes during simulation"

    nparticle = snapshots.snapshots[0].nparticle
    assert len(
        {snapshot.nparticle for snapshot in snapshots.snapshots}
    ) == 1, "Paticle number changes during simulation"

    avet = int(avet/dt/timestep_interval)
    averesults = np.zeros((snapshots.nsnapshots-avet, nparticle))

    for n in range(snapshots.nsnapshots-avet):
        averesults[n] = raw_props[n:n+avet].mean(axis=0)
    return averesults

def spatial_average():
    pass

def gaussian_blurring():
    pass

def atomic_position_average():
    pass