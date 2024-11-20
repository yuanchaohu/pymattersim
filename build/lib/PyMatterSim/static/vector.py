"""see documentation @ ../../docs/vectors.md"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..dynamic.time_corr import time_correlation
from ..neighbors.read_neighbors import read_neighbors
from ..reader.reader_utils import SingleSnapshot, Snapshots
from ..static.sq import conditional_sq
from ..utils.logging import get_logger_handle
from ..utils.pbc import remove_pbc

logger = get_logger_handle(__name__)
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


def participation_ratio(vector: npt.NDArray) -> float:
    """
    Calculate the participation ratio for an vector field

    Inputs:
        vector (npt.NDArray): input vector field, shape as [num_of_particles, ndim]

    Return:
        participation ratio of the vector field (float)
    """

    num_of_particles = vector.shape[0]
    value_PR = 1.0 / (np.sum(np.square((vector * vector).sum(axis=1))) * num_of_particles)
    value_PR *= np.square((vector * vector).sum())

    return value_PR


def local_vector_alignment(vector: npt.NDArray, neighborfile: str) -> npt.NDArray:
    """
    Calculate the local orientational order of a vector field
    Maximum 200 neighbors are considered

    Inputs:
        1. vector (npt.NDArray): input vector field, shape as [num_of_particles, ndim]
        2. neighborfile (str): file name of particle neighbors (see module neighbors)

    Return:
        orientation order of the vector field, shape as [num_of_particles]
    """

    num_of_particles = vector.shape[0]
    with open(neighborfile, "r", encoding="utf-8") as f:
        cnlist = read_neighbors(f, num_of_particles)

    results = np.zeros(num_of_particles)
    for i in range(num_of_particles):
        medium = (vector[i] * vector[cnlist[i, 1 : 1 + cnlist[i, 0]]]).sum(axis=1)
        results[i] = medium.mean()
    return results


def phase_quotient(vector: npt.NDArray, neighborfile: str) -> float:
    """
    Calculate the phase quotient of a vector field
    Maximum 200 neighbors are considered

    Inputs:
        1. vector (npt.NDArray): input vector field, shape as [num_of_particles, ndim]
        2. neighborfile (str): file name of particle neighbors (see module neighbors)

    Return:
        phase quotient measured as a float
    """

    num_of_particles = vector.shape[0]
    with open(neighborfile, "r", encoding="utf-8") as f:
        cnlist = read_neighbors(f, num_of_particles)

    sum_0, sum_1 = 0, 0
    for i in range(num_of_particles):
        medium = (vector[i] * vector[cnlist[i, 1 : 1 + cnlist[i, 0]]]).sum(axis=1)
        sum_0 += medium.sum()
        sum_1 += np.abs(medium).sum()
    return sum_0 / sum_1


def divergence_curl(snapshot: SingleSnapshot, vector: npt.NDArray, ppp: npt.NDArray, neighborfile: str) -> Tuple[npt.NDArray, Optional[npt.NDArray]]:
    """
    Calculate the divergence and curl of a vector field at 2D and 3D
    Divergence is scalar over all dimensions
    Curl only exists in 3D as a vector
    Maximum 200 neighbors are considered

    Inputs:
        1. snapshots (reader.reader_utils.SingleSnapshot): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. vector (npt.NDArray): vector field shape as [num_of_partices, ndim], it determines the
                    dimensionality of the calculation.
        3. ppp (npt.NDArray): the periodic boundary conditions,
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
    if ndim == 3:
        curl = np.zeros((num_of_particles, ndim))

    for i in range(num_of_particles):
        i_cnlist = cnlist[i, 1 : cnlist[i, 0] + 1]
        RIJ = snapshot.positions[i_cnlist] - snapshot.positions[i]
        RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)

        UIJ = vector[i_cnlist] - vector[i]
        divergence[i] = (RIJ * UIJ).sum(axis=1).mean()

        if ndim == 3:
            for j in range(cnlist[i, 0]):
                curl[i] += np.cross(RIJ[j], UIJ[j])
            curl[i] /= cnlist[i, 0]

    if ndim == 2:
        return divergence
    return divergence, curl


def kspace_decomposition():
    pass


def vibrability(
    eigenfrequencies: npt.NDArray,
    eigenvectors: npt.NDArray,
    num_of_partices: int,
    outputfile: str = "",
) -> npt.NDArray:
    """
    Calculate particle-level vibrability from the eigenmodes

    Inputs:
        1. eigenfrequencies (npt.NDArray): eigen frequencies generally from
                    Hessian diagonalization, shape as [num_of_modes,]
        2. eigenvectors (npt.NDArray): eigen vectors associated with eigenfrequencies,
                    each column represents an eigen mode as from np.linalg.eig method

    Return:
        particle-level vibrability in a numpy array
    """
    results = np.zeros(num_of_partices)
    eigenvalues = np.square(eigenfrequencies)
    for i in range(eigenvectors.shape[1]):
        medium = eigenvectors[:, i].reshape(num_of_partices, -1)
        results += np.square(medium).sum(axis=1) / eigenvalues[i]
    if outputfile:
        np.save(outputfile, results)
    return results


def vector_decomposition_sq(
    snapshot: SingleSnapshot,
    qvector: npt.NDArray,
    vector: npt.NDArray,
    outputfile: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decompose the input vector into transverse and longitudinal component by FFT
    and calculate the associated magnitude (or the spectra)
    for a SINGLE snapshot

    Input:
        1. snapshot (reader.reader_utils.SingleSnapshot): single snapshot object of input trajectory
        2. qvector (npt.NDArray of int): designed wavevectors in two-dimensional np.array
                                        (see utils.wavevector)
        3. vector (npt.NDArray): particle-level vector, shape as [num_of_particles, ndim],
                                for example, eigenvector field and velocity field
        4. outputfile (str): filename.csv to save the calculated S(q), default None

    Return:
        1. vector_fft: calculated transverse and longitudinal S(q) for each input wavevector (pd.DataFrame), FFT in complex number is also returned for reference
        2. ave_sqresults: the ensemble averaged S(q) over the same wavenumber (pd.DataFrame)
    """
    ndim = qvector.shape[1]
    vector_fft = conditional_sq(snapshot, qvector, vector)[0]
    unitq = vector_fft[[f"q{i}" for i in range(ndim)]].values
    unitq /= (vector_fft["q"].values)[:, np.newaxis]
    fft_columns = vector_fft[[f"FFT{i}" for i in range(ndim)]].values
    vector_L = np.zeros_like(fft_columns)
    for n in range(qvector.shape[0]):
        medium = np.dot(unitq[n], fft_columns[n])
        vector_L[n] = unitq[n] * medium
    vector_T = fft_columns - vector_L

    medium = pd.DataFrame(vector_T, columns=[f"T_FFT{i}" for i in range(ndim)])
    vector_fft = vector_fft.join(medium)
    vector_fft["Sq_T"] = (vector_T * np.conj(vector_T)).sum(axis=1).real

    medium = pd.DataFrame(vector_L, columns=[f"L_FFT{i}" for i in range(ndim)])
    vector_fft = vector_fft.join(medium)
    vector_fft["Sq_L"] = (vector_L * np.conj(vector_L)).sum(axis=1).real

    vector_fft = vector_fft.round(8)
    ave_sqresults = vector_fft[["Sq", "Sq_T", "Sq_L"]].groupby(vector_fft["q"]).mean().reset_index()
    if outputfile:
        if not outputfile.endswith(".csv"):
            outputfile += ".csv"
        ave_sqresults.to_csv(outputfile, float_format="%.8f", index=False)
    return vector_fft, ave_sqresults


def vector_fft_corr(
    snapshots: Snapshots,
    qvector: npt.NDArray,
    vectors: npt.NDArray,
    dt: float = 0.002,
    outputfile: str = "",
) -> dict[str, pd.DataFrame]:
    """
    Calculate spectra and time correlation of the longitudinal and tranverse components of a vector field by FFT

    Inputs:
        1. snapshots (read.reader_utils.snapshots): multiple trajectories dumped linearly or in logscale
        2. qvector (npt.NDArray of int): designed wavevectors in two-dimensional np.array
                                        (see utils.wavevector)
        3. vectors (npt.NDArray): particle-level vector, shape as [num_of_snapshots, num_of_particles, ndim],
                                for example, eigenvector field and velocity field
        4. dt (float): time step of input snapshots, default 0.002
        5. outputfile (str): filename.csv to save the calculated S(q), default None

    Return:
        1. the averaged spectra of full, transverse, and longitudinal mode,
            saved into a csv dataset
        2. time correlation of FFT of vectors in full, transverse, and longitudinal mode
            Dict as {"FFT": pd.DataFrame, "T_FFT": pd.DataFrame, "L_FFT": pd.DataFrame}
    """
    ndim = qvector.shape[1]
    logger.info(f"Calculate spectra of decomposed modes at d={ndim}")
    spectra = 0
    vectors_fft = []
    for n, snapshot in enumerate(snapshots.snapshots):
        vector_fft, ave_sqresults = vector_decomposition_sq(snapshot=snapshot, qvector=qvector, vector=vectors[n])
        spectra += ave_sqresults
        vectors_fft.append(vector_fft)
    spectra /= snapshots.nsnapshots
    spectra.to_csv(outputfile + ".spectra.csv", float_format="%.8f", index=False)

    logger.info(f"Calculate time correlations of decomposed modes at d={ndim}")
    alldata = {}
    for header in ["FFT", "T_FFT", "L_FFT"]:
        logger.info(f"Calculate autocorrelation for {header} vector")
        cal_data = pd.DataFrame(
            0,
            columns=np.arange(qvector.shape[0]),
            index=np.arange(snapshots.nsnapshots),
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
        final_data = pd.concat([vectors_fft[0][[f"q{i}" for i in range(ndim)] + ["q"]], cal_data.T], axis=1).round(8)
        np.save(outputfile + "." + header + ".npy", final_data.values)
        alldata[header] = final_data
    logger.info(f"Calculate time correlations of decomposed modes at d={ndim} Done")
    return alldata
