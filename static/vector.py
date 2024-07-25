"""see documentation @ ../docs/vectors.md"""

from typing import List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from math import pi
from reader.reader_utils import Snapshots
from neighbors.read_neighbors import read_neighbors
from utils.pbc import remove_pbc
from utils.wavevector import choosewavevector, continuousvector
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

# TODO @Yibang please benchmark with
# https://github.com/yuanchaohu/MyCodes/blob/master/EigenvectorAnalysis.py#L21
def participation_ratio(vector: np.ndarray) -> float:
    """
    Calculate the participation ratio for an vector field

    Inputs:
        vector (np.ndarray): input vector field, shape as [num_of_particles, ndim]
    
    Return:
        participation ratio of the vector field
    """

    num_of_particles = vector.shape[0]
    value_PR = 1.0/(np.sum(np.square((vector*vector).sum(axis=1)))*num_of_particles)
    value_PR *= np.square((vector*vector).sum())

    return value_PR

# TODO @Yibang please benchmark with
# https://github.com/yuanchaohu/MyCodes/blob/master/EigenvectorAnalysis.py#L34
def local_vector_alignment(vector: np.ndarray, neighborfile:str)->np.ndarray:
    """
    Calculate the local orientational order of a vector field
    Maximum 200 neighbors are considered
    
    Inputs:
        1. vector (np.ndarray): input vector field, shape as [num_of_particles, ndim]
        2. neighborfile (str): file name of particle neighbors (see module neighbors)
    
    Return:
        orientation order of the vector field, shape as [num_of_particles]
    """

    num_of_particles = vector.shape[0]
    with open(neighborfile, "r", encoding="utf-8") as f:
        cnlist = read_neighbors(f, num_of_particles)

    results = np.zeros(num_of_particles)
    for i in range(num_of_particles):
        medium = (vector[i]*vector[cnlist[i, 1:1+cnlist[i, 0]]]).sum(axis=1)
        results[i] = medium.mean()
    return results

def phase_quotient(vector:np.ndarray, neighborfile:str) -> float:
    """
    Calculate the phase quotient of a vector field
    Maximum 200 neighbors are considered
    
    Inputs:
        1. vector (np.ndarray): input vector field, shape as [num_of_particles, ndim]
        2. neighborfile (str): file name of particle neighbors (see module neighbors)
    
    Return:
        phase quotient measured as a float
    """

    num_of_particles = vector.shape[0]
    with open(neighborfile, "r", encoding="utf-8") as f:
        cnlist = read_neighbors(f, num_of_particles)

    sum_0, sum_1 = 0, 0
    for i in range(num_of_particles):
        medium = (vector[i]*vector[cnlist[i, 1:1+cnlist[i, 0]]]).sum(axis=1)
        sum_0 += medium.sum()
        sum_1 += np.abs(medium).sum()
    return sum_0/sum_1

# TODO @Yibang please benchmark with
# https://github.com/yuanchaohu/MyCodes/blob/master/EigenvectorAnalysis.py#L250
def divergence_curl(
    snapshot: Snapshots,
    vector: np.ndarray,
    ppp: np.ndarray,
    neighborfile: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calculate the divergence and curl of a vector field
    Divergence is scalar over all dimensions
    Curl only exists in 3D as a vector
    Maximum 200 neighbors are considered

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory 
                     (returned by reader.dump_reader.DumpReader)
        2. vector (np.ndarray): vector field shape as [num_of_partices, ndim]
        3. ppp (np.ndarray): the periodic boundary conditions,
                       setting 1 for yes and 0 for no, default np.array([1,1,1]),
                       set np.array([1,1]) for two-dimensional systems
        4. neighborfile (str): file name of particle neighbors (see module neighbors)
    
    Return:
        divergence and curl (only 3D) in numpy array of the input vector
    """

    num_of_particles, ndim = vector.shape
    with open(neighborfile, "r", encoding="utf-8") as f:
        cnlist = read_neighbors(f, num_of_particles)

    divergence = np.zeros(num_of_particles)
    if ndim==3:
        curl=np.zeros((num_of_particles, ndim))

    for i in range(num_of_particles):
        i_cnlist = cnlist[i, 1:cnlist[i, 0]+1]
        #positional vectors
        RIJ = snapshot.snapshots[0].positions[i_cnlist] - snapshot.snapshots[0].positions[i]
        RIJ = remove_pbc(RIJ, snapshot.snapshots[0].hmatrix, ppp)

        #divergence
        UIJ = vector[i_cnlist] - vector[i]
        divergence[i] = (RIJ * UIJ).sum(axis=1).mean()

        #curl
        if ndim==3:
            for j in range(num_of_particles):
                curl[i] += np.cross(RIJ[j], UIJ[j])
            curl[i] /= cnlist[i, 0]

    if ndim != 3:
        return divergence
    return divergence, curl

def kspace_decomposition():
    pass

# TODO @Yibang please benchmark with
# https://github.com/yuanchaohu/MyCodes/blob/master/EigenvectorAnalysis.py#L360
def vibrability(
        eigenfrequencies: np.ndarray,
        eigenvectors: np.ndarray,
        num_of_partices: int,
        outputfile: str="",
    ) -> np.ndarray:
    """
    Calculate particle-level vibrability from the eigenmodes

    Inputs:
        1. eigenfrequencies (np.ndarray): eigen frequencies generally from Hessian,
                    shape as [num_of_modes,]
        2. eigenvectors (np.ndarray): eigen vectors associated with eigenfrequencies,
                    each column represents an eigen mode as from np.linalg.eig method
    
    Return:
        particle-level vibrability in a numpy array
    """
    results = np.zeros(num_of_partices)
    eigenvalues = np.square(eigenfrequencies)
    for i in range(eigenvectors.shape[1]):
        medium = eigenvectors[:, i].reshape(num_of_partices, -1)
        results += np.square(medium).sum(axis=1)/eigenvalues[i]
    if outputfile:
        np.save(outputfile, results)
    return results
