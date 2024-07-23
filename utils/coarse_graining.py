# coding = utf-8

"""
see documentation @ ../docs/utils.md
"""

from typing import Tuple, List
import numpy as np
from reader.reader_utils import Snapshots
from neighbors.read_neighbors import read_neighbors
from utils.logging import get_logger_handle
from utils.funcs import grid_gaussian
from utils.pbc import remove_pbc

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
    results = np.zeros((
        snapshots.nsnapshots-time_nsnapshot,
        snapshots.snapshots[0].nparticle),
    dtype=np.complex128)
    # save the middle snapshot id for each time average period
    results_middle_snapshots = []

    for n in range(results.shape[0]):
        results[n, :] = input_property[n:n+time_nsnapshot].mean(axis=0)
        results_middle_snapshots.append(round(n+time_nsnapshot/2))
    return results, np.array(results_middle_snapshots)

def spatial_average(
    input_property: np.ndarray,
    neighborfile: str,
    Nmax: int=30,
    outputfile: str="",
) -> np.ndarray:
    """
    coarse-graining the input variable over certain length scale
    given by the pre-defined neighbor list

    Inputs:
        1. input_property (np.ndarray): input property to be coarse-grained,
            should be in the shape [num_of_snapshots, num_of_particles]
        2. neighborfile (str): file name of pre-defined neighbor list
        3. Namx (int): maximum number of particle neighbors
        4. outputfile (str): file name of coarse-grained variable
    
    Return:
        coarse-grained input property in numpy ndarray
    """
    cg_input_property = np.zeros_like(input_property)
    fneighbor = open(neighborfile, "-r", encoding="utf-8")
    for n in range(input_property.shape[0]):
        cnlist = read_neighbors(fneighbor, input_property.shape[1], Nmax)
        for i in range(input_property.shape[1]):
            indices = cnlist[i, 1:1+cnlist[i,0]].tolist()
            indices.append(i)
            cg_input_property[n, i] = input_property[n, indices].mean()
    fneighbor.close()
    if outputfile:
        np.save(outputfile, cg_input_property)
    return cg_input_property

def gaussian_blurring(
    snapshots: Snapshots,
    condition: np.ndarray,
    ngrids: List[int],
    sigma: float=2.0,
    ppp: np.ndarray=np.array([1,1,1]),
    gaussian_cut: float=6.0,
    outputfile: str="",
):
    ndim = len(ngrids)
    ppp = ppp[:ndim]
    ngrids = np.array(ngrids)
    if len(condition)==2:
        # scalar
        ncol_add = 1
    elif len(condition)==3:
        # vector
        ncol_add = 2
    elif len(condition)==4:
        # tensor
        ncol_add = 3
    else:
        raise ValueError("Wrong input condition variable")

    logger.info(f"Performing Gaussian Blurring for a {ndim}-dimensional system")
    grid_property = np.zeros((snapshots.nsnapshots, np.prod(ngrids), ndim+ncol_add))
    for n, snapshot in enumerate(snapshots.snapshots):
        bxobounds = snapshot.boxbounds
        X = np.linspace(bxobounds[0,0], bxobounds[0,1], ngrids[0])
        Y = np.linspace(bxobounds[1,0], bxobounds[1,1], ngrids[1])
        if ndim==2:
            for i in range(ngrids[0]):
                for j in range(ngrids[1]):
                    indice = i*ngrids[0]+j
                    grid_property[n, indice, :ndim] = [X[i], Y[j]]
        else:
            Z = np.linspace(bxobounds[2,0], bxobounds[2,1], ngrids[2])
            for i in range(ngrids[0]):
                for j in range(ngrids[1]):
                    for k in range(ngrids[2]):
                        indice = i*ngrids[0]+j*ngrids[1]+k
                        grid_property[n, indice, :ndim] = [X[i], Y[j], Z[k]]

        for i in range(snapshot.nparticle):
            RIJ = grid_property[n, :, :ndim] - snapshot.positions[i]
            RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp=ppp)
            RIJ = np.linalg.norm(RIJ, axis=1)
            grid_property[n, i, ndim:] += np.where(
                RIJ < gaussian_cut,
                grid_gaussian(RIJ, sigma) * condition[n, i],
                0
            )
    if outputfile:
        np.save(outputfile, grid_property)
    return grid_property

def atomic_position_average():
    pass