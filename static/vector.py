"""see documentation @ ../docs/vectors.md"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from reader.reader_utils import Snapshots, SingleSnapshot
from neighbors.read_neighbors import read_neighbors
from dynamic.time_corr import time_correlation
from utils.pbc import remove_pbc
from sq import conditional_sq
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

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


def vector_decomposition_sq(
        snapshot: SingleSnapshot,
        qvector: np.ndarray,
        vector: np.ndarray,
        outputfile: str="",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decompose the input vector into transverse and longitudinal component
    and calculate the associated magitude

    Input:
        1. snapshot (reader.reader_utils.SingleSnapshot): single snapshot object of input trajectory
        2. qvector (np.ndarray of int): designed wavevectors in two-dimensional np.array 
                                        (see utils.wavevector)
        3. vector (np.ndarray): particle-level vector, shape as [num_of_particles, ndim],
                                for example, eigenvector field and velocity field
        4. outputfile (str): the name of csv file to save the calculated S(q), default None
    
    Return:
        calculated transverse and longitudinal S(q) for each input wavevector (pd.DataFrame)
        and the ensemble averaged S(q) over the same wavenumber (pd.DataFrame)
        (FFT in complex number is also returned for reference)
    """
    ndim = qvector.shape[1]
    vector_fft = conditional_sq(snapshot, qvector, vector)[0]
    unitq = vector_fft[[f"q{i}" for i in range(ndim)]].values/vector_fft["q"].values
    fft_columns = vector_fft[[f"FFT{i}" for i in range(ndim)]].values
    vector_L = np.zeros_like(fft_columns)
    for n in range(qvector.shape[0]):
        medium = np.dot(unitq[n], fft_columns[n])
        vector_L[n] = unitq[n] * medium
    vector_T = fft_columns - vector_L

    medium = pd.DataFrame(vector_T, columns=[f"T_FFT{i}" for i in range(ndim)])
    vector_fft.join(medium)
    vector_fft["Sq_T"] = (vector_T * np.conj(vector_T)).sum(axis=1).real

    medium = pd.DataFrame(vector_L, columns=[f"L_FFT{i}" for i in range(ndim)])
    vector_fft.join(medium)
    vector_fft["Sq_L"] = (vector_L * np.conj(vector_L)).sum(axis=1).real

    vector_fft = vector_fft.round(8)
    ave_sqresults = vector_fft[["Sq", "Sq_T", "Sq_L"]].groupby(vector_fft["q"]).mean().reset_index()
    if outputfile:
        ave_sqresults.to_csv(outputfile, float_format="%.8f", index=False)
    return vector_fft, ave_sqresults


def vector_fft_corr(
        snapshots: Snapshots,
        qvector: np.ndarray,
        vectors: np.ndarray,
        dt: float=0.002,
        outputfile: str="",
    ) -> dict[str, pd.DataFrame]:
    """
    Calculate spectra and auto-correlation of a vector field by FFT

    Inputs:
        1. snapshots (read.reader_utils.snapshots): multiple trajectories dumped linearly or in logscale
        2. qvector (np.ndarray of int): designed wavevectors in two-dimensional np.array 
                                        (see utils.wavevector)
        3. vectors (np.ndarray): particle-level vector, shape as [num_of_snapshots, num_of_particles, ndim],
                                for example, eigenvector field and velocity field
        4. dt (float): time step of input snapshots, default 0.002
        5. outputfile (str): the name of csv file to save the calculated S(q), default None
    
    Return:
        time correlation of FFT of vectors in full, transverse, and longitudinal mode
        Dict as {"FFT": pd.DataFrame, "T_FFT": pd.DataFrame, "L_FFT": pd.DataFrame}
    """

    ndim = qvector.shape[1]
    spectra = 0
    vectors_fft = []
    for n, snapshot in enumerate(snapshots.snapshots):
        vector_fft, ave_sqresults = vector_decomposition_sq(
            snapshot=snapshot,
            qvector=qvector,
            vector=vectors[n]
        )
        spectra += ave_sqresults
        vectors_fft.append(vector_fft)
    spectra /= snapshots.nsnapshots
    spectra.to_csv(outputfile+".spectra.csv", float_format="%.8f", index=False)

    alldata = {}
    for header in ["FFT", "T_FFT", "L_FFT"]:
        logger.info(f"Calculate autocorrelation for {header} vector")
        cal_data = pd.DataFrame(
            0, 
            columns=np.arange(qvector.shape[0]),
            index=np.arange(snapshots.nsnapshots)
        )
        column_name = [f"{header}{i}" for i in range(ndim)]
        for n in range(qvector.shape[0]):
            condition = [item[column_name].values[n] for item in vectors_fft]
            medium = time_correlation(
                snapshots=snapshots,
                condition=np.array(condition),
                dt=dt,
            )
            cal_data[n] = medium["time_corr"].values
        cal_data.index = medium["t"].values
        # columns: [q0, q1, q2, q, t1, t2, t3....]
        final_data = pd.concat([
            vectors_fft[0][[f"q{i}" for i in range(ndim)]+["q"]],
            cal_data.T
        ], axis=1).round(8)
        np.save(outputfile+header+".npy", final_data.values)
        alldata[header] = final_data
    return alldata