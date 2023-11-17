# coding = utf-8

""" see xxx @ orderings.md """

import numpy as np
from reader.reader_utils import Snapshots
from utils.logging import get_logger_handle
from utils.pbc import remove_pbc

logger = get_logger_handle(__name__)

def q8(
    snapshots: Snapshots,
    ppp: np.ndarray=np.array([1,1,1]),
    outputfile: str=None
):
    """
    Calculate local tetrahedral order of the simulation system,
    such as for water-type and silicon-type systems

    Inputs:
        snapshots:
        ppp:
        outputfile:
    """
    logger.info("Start calculating local tetrahedral order of the input system")

    assert len(
        {snapshot.nparticle for snapshot in snapshots.snapshots}
        )==1, "Paticle number changes during simulation"
    assert len(
        {tuple(snapshot.boxlength) for snapshot in snapshots.snapshots}
        )==1, "Simulation box length changes during simulation"

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

    if outputfile:
        np.savetxt(outputfile, resutls, fmt="%.6f", header="", comments="")
    logger.info("Finish calculating local tetrahedral order of the input system")
    return resutls
