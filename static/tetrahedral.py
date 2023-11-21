# coding = utf-8

"""see documentation @ ../docs/orderings.md"""

import numpy as np
from reader.reader_utils import Snapshots
from utils.logging import get_logger_handle
from utils.pbc import remove_pbc

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name

def q8(
    snapshots: Snapshots,
    ppp: np.ndarray=np.array([1,1,1]),
    outputfile: str=None
) -> np.ndarray:
    """
    Calculate local tetrahedral order of the simulation system,
    such as for water-type and silicon-type systems in three dimensions

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. ppp (np.ndarray): the periodic boundary conditions,
                        setting 1 for yes and 0 for no, default np.ndarray=np.array([1,1,1])
        3. outputfile (str): the name of file to save the calculated local tetrahedral order

    Return:
        calculated local tetrahedral order in np.ndarray with shape [nsnapshots, nparticle]
    """
    logger.info("Start calculating local tetrahedral order q8")

    assert len(
        {snapshot.nparticle for snapshot in snapshots.snapshots}
        )==1, "Paticle number changes during simulation"
    assert len(
        {tuple(snapshot.boxlength) for snapshot in snapshots.snapshots}
        )==1, "Simulation box length changes during simulation"
    assert ppp.shape[0]==3, "Simulation box is in three dimensions"

    # only consider the nearest four neighbors
    num_nearest = 4
    resutls = np.zeros((snapshots.nsnapshots, snapshots.snapshots[0].nparticle))
    for n, snapshot in enumerate(snapshots.snapshots):
        for i in range(snapshot.nparticle):
            RIJ = snapshot.positions - snapshot.positions[i]
            RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
            distance = np.linalg.norm(RIJ, axis=1)
            nearests = np.argpartition(distance, num_nearest+1)[:num_nearest+1]
            nearests = [j for j in nearests if j != i]
            for j in range(num_nearest-1):
                for k in range(j+1, num_nearest):
                    medium1 = np.dot(RIJ[nearests[j]], RIJ[nearests[k]])
                    medium2 = distance[nearests[j]] * distance[nearests[k]]
                    resutls[n, i] += (medium1 / medium2 + 1.0/3)**2
    resutls = 1.0 - 3.0/8*resutls/num_nearest

    if outputfile:
        np.savetxt(outputfile, resutls, fmt="%.6f", header="", comments="")
    logger.info("Finish calculating local tetrahedral order of the input system")
    return resutls
