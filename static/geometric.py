#coding = utf-8

"""see documentation @ ../docs/orderings.md"""

from itertools import combinations
import numpy as np
from reader.reader_utils import Snapshots
from neighbors.read_neighbors import read_neighbors
from utils.geometry import triangle_angle
from utils.pbc import remove_pbc
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

def packing_capability_2d(
        snapshots: Snapshots,
        sigmas: np.ndarray,
        neighborfile: str,
        ppp: np.ndarray=np.array([1,1]),
        outputfile: str=None
) -> np.ndarray:
    """
    Calculate packing capability of a 2D system based on geometry

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. sigmas (np.ndarray): particle sizes for each pair of particle type (ascending order) 
                                in numpy array, shape [particle_type, particle_type]
        3. neighborfile (str): file name of particle neighbors (see module neighbors)
        4. ppp (np.ndarray): the periodic boundary conditions, setting 1 for yes and 0 for no, 
                             default np.ndarray=np.array([1,1])
        5. outputfile (str): file name to save the calculated packing capability, default None
                             supporting both binary npy file with extension "npy" and text file with extension "dat" or "txt"

    Return:
        Calculated packing capability of a 2D system
        shape [nsnapshots, nparticles]
    """
    assert ppp.shape[0]==2, "This calculation is for a two-dimensional system"
    logger.info("Start calculating packing capability in two dimensions")

    # calcualte reference angles, save to a 3d numpy array
    particle_type = sigmas.shape[0]
    assert particle_type==snapshots.snapshots[0].particle_type.max(),\
          'Error shape of input sigmas (particle diameters)'

    reference_angles = np.zeros((particle_type, particle_type, particle_type))
    for o in range(particle_type):
        for i in range(particle_type):
            for j in range(particle_type):
                reference_angles[o, i, j]=triangle_angle(
                    sigmas[o, i], sigmas[o, j], sigmas[i, j])

    # calculate real angles in trajectory
    results = np.zeros((snapshots.nsnapshots, snapshots.snapshots[0].nparticle))
    fneighbor = open(neighborfile, 'r', encoding="utf-8")
    for n, snapshot in enumerate(snapshots.snapshots):
        neighborlist = read_neighbors(fneighbor, snapshot.nparticle, 20)
        for o in range(snapshot.nparticle):
            cnlist = neighborlist[o, 1:neighborlist[o, 0]+1]
            theta_o = 0
            for i, j in combinations(cnlist, 2):
                i_cnlist = neighborlist[i, 1:neighborlist[i, 0]+1]
                j_cnlist = neighborlist[j, 1:neighborlist[j, 0]+1]
                if (j in i_cnlist) & (i in j_cnlist):
                    vectors_oij = snapshot.positions[[i, j]] - snapshot.positions[o][np.newaxis, :]
                    vectors_oij = remove_pbc(vectors_oij, snapshot.hmatrix, ppp)
                    distance = np.linalg.norm(vectors_oij, axis=1)
                    vectors_oij /= distance[:, np.newaxis]
                    theta = np.dot(vectors_oij[0], vectors_oij[1])
                    theta = np.arccos(theta)
                    otype, itype, jtype = snapshot.particle_type[[o, i, j]]-1
                    theta_o += abs(theta - reference_angles[otype, itype, jtype])
            results[n, o] = theta_o / neighborlist[o, 0]
    fneighbor.close()

    if outputfile:
        if outputfile[-3:] == 'npy':
            np.save(outputfile, results)
        elif outputfile[-3:] == ('dat' or 'txt'):
            np.savetxt(outputfile, results, fmt='%.6f', header='', comments='')
        else:
            logger.info('The format of outputfile supports binary npy file with extension "npy" and text file with extension "dat" or "txt"')
    logger.info("Finish calculating packing capability of a 2D system")
    return results
