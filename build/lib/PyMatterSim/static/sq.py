# coding = utf-8

"""see documentation @ ../../docs/sq.md"""

from math import sqrt
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..reader.reader_utils import SingleSnapshot, Snapshots
from ..utils.logging import get_logger_handle
from ..utils.wavevector import choosewavevector

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


def conditional_sq(
    snapshot: SingleSnapshot,
    qvector: npt.NDArray,
    condition: npt.NDArray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the structure factor of a single configuration for selected particles,
    Also useful to calculate the Fourier-Transform of a physical quantity "A".
    There are three cases considered:
    1. condition is bool type, so calculate S(q) & FFT for selected particles
    2. condition is float vector type, so calculate spectral & FFT of vector quantity
    3. condition is float scalar type, so calculate spectral & FFT of scalar quantity

    Input:
        1. snapshot (reader.reader_utils.SingleSnapshot): single snapshot object of input trajectory
        2. qvector (npt.NDArray of int): designed wavevectors in two-dimensional np.array (see utils.wavevector)
        3. condition (npt.NDArray): particle-level condition / property

    Return:
        calculated conditional S(q) for each input wavevector (pd.DataFrame)
        and the ensemble averaged S(q) over the same wavenumber (pd.DataFrame)
        (FFT in complex number is also returned for reference)
    """
    ndim = snapshot.positions.shape[1]
    sqresults = pd.DataFrame(
        0,
        index=range(
            qvector.shape[0]),
        columns="q Sq".split())
    twopidl = 2 * np.pi / snapshot.boxlength
    qvector = qvector.astype(np.float64) * twopidl[np.newaxis, :]
    df_qvector = pd.DataFrame(qvector, columns=[f"q{i}" for i in range(ndim)])
    sqresults["q"] = np.linalg.norm(qvector, axis=1)

    if condition.dtype == "bool":
        Natom = condition.sum()
        logger.info(f"Calculate S(q) for {Natom} selected atoms")
        exp_thetas = 0
        positions = snapshot.positions[condition]
        for i in range(Natom):
            thetas = (qvector * positions[i][np.newaxis, :]).sum(axis=1)
            exp_thetas += np.exp(-1j * thetas)
        exp_thetas /= sqrt(Natom)
        sqresults["Sq"] = (exp_thetas * np.conj(exp_thetas)).real
        sqresults["FFT"] = exp_thetas
    elif len(condition.shape) > 1:
        logger.info(
            "Calculate Fourier-Transform of float-Vector physical quantity A")
        exp_thetas = 0
        for i in range(snapshot.nparticle):
            thetas = (qvector *
                      snapshot.positions[i][np.newaxis, :]).sum(axis=1)
            exp_thetas += np.exp(-1j * thetas)[:,
                                               np.newaxis] * condition[i][np.newaxis, :]
        exp_thetas /= sqrt(snapshot.nparticle)
        sqresults["Sq"] = (exp_thetas * np.conj(exp_thetas)).sum(axis=1).real
        dim_fft = pd.DataFrame(
            exp_thetas, columns=[
                f"FFT{i}" for i in range(ndim)])
        sqresults = sqresults.join(dim_fft)
    else:
        logger.info(
            "Calculate Fourier-Transform of float-Scalar physical quantity A")
        exp_thetas = 0
        for i in range(snapshot.nparticle):
            thetas = (qvector *
                      snapshot.positions[i][np.newaxis, :]).sum(axis=1)
            exp_thetas += np.exp(-1j * thetas) * condition[i]
        exp_thetas /= sqrt(snapshot.nparticle)
        sqresults["Sq"] = (exp_thetas * np.conj(exp_thetas)).real
        sqresults["FFT"] = exp_thetas

    # TODO @Yibang please test the new float df_qvector in sqresults
    sqresults = df_qvector.join(sqresults)
    # ensemble average over same q but different directions
    sqresults = sqresults.round(8)
    ave_sqresults = sqresults["Sq"].groupby(
        sqresults["q"]).mean().reset_index()

    return sqresults, ave_sqresults


class sq:
    """
    This module is used to calculate static structure factors
    covering unary to quinary systems.
    """

    def __init__(
            self,
            snapshots: Snapshots,
            qrange: float = 10.0,
            onlypositive: bool = False,
            qvector: npt.NDArray = None,
            saveqvectors: bool = False,
            outputfile: str = None
    ) -> None:
        """
        Initializing S(q) class

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                         (returned by reader.dump_reader.DumpReader)
            2. qrange (float): the wave number range to be calculated, default 10
            3. onlypositive (bool): whether only consider positive wave vectors, default False
            4. qvector (npt.NDArray of int): input wave vectors in integers as two-dimensional np.array,
                                            if None (default) use qrange & onlypositive
            5. saveqvectors (bool): whether to save S(q) for specific wavevectors, default False
            6. outputfile (str): the name of csv file to save the calculated S(q), default None

        Return:
            None
        """
        self.snapshots = snapshots
        self.outputfile = outputfile
        self.saveqvectors = saveqvectors

        self.nsnapshots = snapshots.nsnapshots
        self.nparticle = snapshots.snapshots[0].nparticle
        assert len({snapshot.nparticle for snapshot in self.snapshots.snapshots}
                   ) == 1, "Paticle Number Changes during simulation"
        assert len({tuple(snapshot.boxlength) for snapshot in self.snapshots.snapshots}
                   ) == 1, "Simulation Box Length Changes during simulation"

        self.typenumber, self.typecount = np.unique(
            self.snapshots.snapshots[0].particle_type, return_counts=True)
        assert np.sum(self.typecount) == self.nparticle, \
            "Sum of Indivdual types is Not the Total Amount"
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')

        ndim = snapshots.snapshots[0].positions.shape[1]
        twopidl = 2 * np.pi / self.snapshots.snapshots[0].boxlength
        if qvector is not None:
            self.qvector = qvector
        else:
            numofq = int(qrange * 2.0 / twopidl.min())
            self.qvector = choosewavevector(ndim, numofq, onlypositive)
        self.df_qvector = pd.DataFrame(
            self.qvector, columns=[
                f"q{i}" for i in range(ndim)])
        self.qvector = self.qvector.astype(np.float64) * twopidl[np.newaxis, :]
        self.qvalue = np.linalg.norm(self.qvector, axis=1)

    def getresults(self) -> Optional[Callable]:
        """
        Calculating S(q) for system with different particle type numbers

        Return:
            Optional[Callable]
        """
        if len(self.typenumber) == 1:
            return self.unary()
        if len(self.typenumber) == 2:
            return self.binary()
        if len(self.typenumber) == 3:
            return self.ternary()
        if len(self.typenumber) == 4:
            return self.quarternary()
        if len(self.typenumber) == 5:
            return self.quinary()
        if len(self.typenumber) > 5:
            logger.info(
                f"This is a {len(self.typenumber)} system, only overall S(q) calculated")
            return self.unary()

    def unary(self) -> pd.DataFrame:
        """
        Calculating S(q) for unary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start calculating S(q) of a unary system')

        sqresults = pd.DataFrame(
            0,
            index=self.df_qvector.index,
            columns="q Sq".split())
        sqresults["q"] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = 0
            for i in range(snapshot.nparticle):
                thetas = (self.qvector *
                          snapshot.positions[i][np.newaxis, :]).sum(axis=1)
                exp_thetas += np.exp(-1j * thetas)
            sqresults["Sq"] += (exp_thetas * np.conj(exp_thetas)).real
        sqresults["Sq"] /= (self.nsnapshots * self.nparticle)
        if self.saveqvectors:
            self.df_qvector.join(sqresults).to_csv(
                self.outputfile[:-4] + "_qvectors.csv",
                float_format="%.6f",
                index=False
            )
        # ensemble average over same q but different directions
        sqresults = sqresults.round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish calculating S(q) of a unary system')
        return results

    def binary(self) -> pd.DataFrame:
        """
        Calculating S(q) for binary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start calculating S(q) for a binary system')
        sqresults = pd.DataFrame(
            0,
            index=self.df_qvector.index,
            columns="q Sq Sq11 Sq22 Sq12".split()
        )
        sqresults["q"] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = {
                "all": 0,
                "11": 0,
                "22": 0
            }
            for i in range(snapshot.nparticle):
                thetas = (self.qvector *
                          snapshot.positions[i][np.newaxis, :]).sum(axis=1)
                medium = np.exp(-1j * thetas)
                exp_thetas["all"] += medium
                if snapshot.particle_type[i] == 1:
                    exp_thetas["11"] += medium
                else:
                    exp_thetas["22"] += medium
            sqresults["Sq"] += (exp_thetas["all"] *
                                np.conj(exp_thetas["all"])).real
            sqresults["Sq11"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["11"])).real
            sqresults["Sq22"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["22"])).real
            sqresults["Sq12"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["22"])).real

        sqresults["Sq"] /= (self.nsnapshots * self.nparticle)
        sqresults["Sq11"] /= (self.nsnapshots * self.typecount[0])
        sqresults["Sq22"] /= (self.nsnapshots * self.typecount[1])
        sqresults["Sq12"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[1]))
        if self.saveqvectors:
            self.df_qvector.join(sqresults).to_csv(
                self.outputfile[:-4] + "_qvectors.csv",
                float_format="%.6f",
                index=False
            )
        # ensemble average over same q but different directions
        sqresults = sqresults.round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish calculating S(q) of a binary system')
        return results

    def ternary(self) -> pd.DataFrame:
        """
        Calculating S(q) for ternary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start calculating S(q) for a ternary system')
        sqresults = pd.DataFrame(
            0,
            index=self.df_qvector.index,
            columns="q Sq Sq11 Sq22 Sq33 Sq12 Sq13 Sq23".split()
        )
        sqresults["q"] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = {
                "all": 0,
                "11": 0,
                "22": 0,
                "33": 0
            }
            for i in range(snapshot.nparticle):
                thetas = (self.qvector *
                          snapshot.positions[i][np.newaxis, :]).sum(axis=1)
                medium = np.exp(-1j * thetas)
                exp_thetas["all"] += medium
                if snapshot.particle_type[i] == 1:
                    exp_thetas["11"] += medium
                elif snapshot.particle_type[i] == 2:
                    exp_thetas["22"] += medium
                else:
                    exp_thetas["33"] += medium
            sqresults["Sq"] += (exp_thetas["all"] *
                                np.conj(exp_thetas["all"])).real
            sqresults["Sq11"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["11"])).real
            sqresults["Sq22"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["22"])).real
            sqresults["Sq33"] += (exp_thetas["33"] *
                                  np.conj(exp_thetas["33"])).real
            sqresults["Sq12"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["22"])).real
            sqresults["Sq13"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["33"])).real
            sqresults["Sq23"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["33"])).real

        sqresults["Sq"] /= (self.nsnapshots * self.nparticle)
        sqresults["Sq11"] /= (self.nsnapshots * self.typecount[0])
        sqresults["Sq22"] /= (self.nsnapshots * self.typecount[1])
        sqresults["Sq33"] /= (self.nsnapshots * self.typecount[2])
        sqresults["Sq12"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[1]))
        sqresults["Sq13"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[2]))
        sqresults["Sq23"] /= (self.nsnapshots *
                              sqrt(self.typecount[1] * self.typecount[2]))
        if self.saveqvectors:
            self.df_qvector.join(sqresults).to_csv(
                self.outputfile[:-4] + "_qvectors.csv",
                float_format="%.6f",
                index=False
            )
        # ensemble average over same q but different directions
        sqresults = sqresults.round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish calculating S(q) of a ternary system')
        return results

    def quarternary(self) -> pd.DataFrame:
        """
        Calculating S(q) for quarternary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start calculating S(q) for a quarternary system')
        sqresults = pd.DataFrame(
            0,
            index=self.df_qvector.index,
            columns="q Sq Sq11 Sq22 Sq33 Sq44 Sq12 Sq13 Sq14 Sq23 Sq24 Sq34".split())
        sqresults["q"] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = {
                "all": 0,
                "11": 0,
                "22": 0,
                "33": 0,
                "44": 0
            }
            for i in range(snapshot.nparticle):
                thetas = (self.qvector *
                          snapshot.positions[i][np.newaxis, :]).sum(axis=1)
                medium = np.exp(-1j * thetas)
                exp_thetas["all"] += medium
                if snapshot.particle_type[i] == 1:
                    exp_thetas["11"] += medium
                elif snapshot.particle_type[i] == 2:
                    exp_thetas["22"] += medium
                elif snapshot.particle_type[i] == 3:
                    exp_thetas["33"] += medium
                else:
                    exp_thetas["44"] += medium
            sqresults["Sq"] += (exp_thetas["all"] *
                                np.conj(exp_thetas["all"])).real
            sqresults["Sq11"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["11"])).real
            sqresults["Sq22"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["22"])).real
            sqresults["Sq33"] += (exp_thetas["33"] *
                                  np.conj(exp_thetas["33"])).real
            sqresults["Sq44"] += (exp_thetas["44"] *
                                  np.conj(exp_thetas["44"])).real
            sqresults["Sq12"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["22"])).real
            sqresults["Sq13"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["33"])).real
            sqresults["Sq14"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["44"])).real
            sqresults["Sq23"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["33"])).real
            sqresults["Sq24"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["44"])).real
            sqresults["Sq34"] += (exp_thetas["33"] *
                                  np.conj(exp_thetas["44"])).real

        sqresults["Sq"] /= (self.nsnapshots * self.nparticle)
        sqresults["Sq11"] /= (self.nsnapshots * self.typecount[0])
        sqresults["Sq22"] /= (self.nsnapshots * self.typecount[1])
        sqresults["Sq33"] /= (self.nsnapshots * self.typecount[2])
        sqresults["Sq44"] /= (self.nsnapshots * self.typecount[3])
        sqresults["Sq12"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[1]))
        sqresults["Sq13"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[2]))
        sqresults["Sq14"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[3]))
        sqresults["Sq23"] /= (self.nsnapshots *
                              sqrt(self.typecount[1] * self.typecount[2]))
        sqresults["Sq24"] /= (self.nsnapshots *
                              sqrt(self.typecount[1] * self.typecount[3]))
        sqresults["Sq34"] /= (self.nsnapshots *
                              sqrt(self.typecount[2] * self.typecount[3]))
        if self.saveqvectors:
            self.df_qvector.join(sqresults).to_csv(
                self.outputfile[:-4] + "_qvectors.csv",
                float_format="%.6f",
                index=False
            )
        # ensemble average over same q but different directions
        sqresults = sqresults.round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish calculating S(q) of a quarternary system')
        return results

    def quinary(self) -> pd.DataFrame:
        """
        Calculating S(q) for quainary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start calculating S(q) for a quinary system')
        sqresults = pd.DataFrame(
            0,
            index=self.df_qvector.index,
            columns="q Sq Sq11 Sq22 Sq33 Sq44 Sq55 Sq12 Sq13 Sq14 Sq15 \
                    Sq23 Sq24 Sq25 Sq34 Sq35 Sq45".split()
        )
        sqresults["q"] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = {
                "all": 0,
                "11": 0,
                "22": 0,
                "33": 0,
                "44": 0,
                "55": 0,
            }
            for i in range(snapshot.nparticle):
                thetas = (self.qvector *
                          snapshot.positions[i][np.newaxis, :]).sum(axis=1)
                medium = np.exp(-1j * thetas)
                exp_thetas["all"] += medium
                if snapshot.particle_type[i] == 1:
                    exp_thetas["11"] += medium
                elif snapshot.particle_type[i] == 2:
                    exp_thetas["22"] += medium
                elif snapshot.particle_type[i] == 3:
                    exp_thetas["33"] += medium
                elif snapshot.particle_type[i] == 4:
                    exp_thetas["44"] += medium
                else:
                    exp_thetas["55"] += medium
            sqresults["Sq"] += (exp_thetas["all"] *
                                np.conj(exp_thetas["all"])).real
            sqresults["Sq11"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["11"])).real
            sqresults["Sq22"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["22"])).real
            sqresults["Sq33"] += (exp_thetas["33"] *
                                  np.conj(exp_thetas["33"])).real
            sqresults["Sq44"] += (exp_thetas["44"] *
                                  np.conj(exp_thetas["44"])).real
            sqresults["Sq55"] += (exp_thetas["55"] *
                                  np.conj(exp_thetas["55"])).real
            sqresults["Sq12"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["22"])).real
            sqresults["Sq13"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["33"])).real
            sqresults["Sq14"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["44"])).real
            sqresults["Sq15"] += (exp_thetas["11"] *
                                  np.conj(exp_thetas["55"])).real
            sqresults["Sq23"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["33"])).real
            sqresults["Sq24"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["44"])).real
            sqresults["Sq25"] += (exp_thetas["22"] *
                                  np.conj(exp_thetas["55"])).real
            sqresults["Sq34"] += (exp_thetas["33"] *
                                  np.conj(exp_thetas["44"])).real
            sqresults["Sq35"] += (exp_thetas["33"] *
                                  np.conj(exp_thetas["55"])).real
            sqresults["Sq45"] += (exp_thetas["44"] *
                                  np.conj(exp_thetas["55"])).real

        sqresults["Sq"] /= (self.nsnapshots * self.nparticle)
        sqresults["Sq11"] /= (self.nsnapshots * self.typecount[0])
        sqresults["Sq22"] /= (self.nsnapshots * self.typecount[1])
        sqresults["Sq33"] /= (self.nsnapshots * self.typecount[2])
        sqresults["Sq44"] /= (self.nsnapshots * self.typecount[3])
        sqresults["Sq55"] /= (self.nsnapshots * self.typecount[4])
        sqresults["Sq12"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[1]))
        sqresults["Sq13"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[2]))
        sqresults["Sq14"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[3]))
        sqresults["Sq15"] /= (self.nsnapshots *
                              sqrt(self.typecount[0] * self.typecount[4]))
        sqresults["Sq23"] /= (self.nsnapshots *
                              sqrt(self.typecount[1] * self.typecount[2]))
        sqresults["Sq24"] /= (self.nsnapshots *
                              sqrt(self.typecount[1] * self.typecount[3]))
        sqresults["Sq25"] /= (self.nsnapshots *
                              sqrt(self.typecount[1] * self.typecount[4]))
        sqresults["Sq34"] /= (self.nsnapshots *
                              sqrt(self.typecount[2] * self.typecount[3]))
        sqresults["Sq35"] /= (self.nsnapshots *
                              sqrt(self.typecount[2] * self.typecount[4]))
        sqresults["Sq45"] /= (self.nsnapshots *
                              sqrt(self.typecount[3] * self.typecount[4]))
        if self.saveqvectors:
            self.df_qvector.join(sqresults).to_csv(
                self.outputfile[:-4] + "_qvectors.csv",
                float_format="%.6f",
                index=False
            )
        # ensemble average over same q but different directions
        sqresults = sqresults.round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish calculating S(q) of a quinary system')
        return results
