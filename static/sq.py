# coding = utf-8

"""see documentation @ ../docs/static.md"""

from typing import Optional, Callable
import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from utils.wavevector import choosewavevector
from utils.logging_utils import get_logger_handle
from math import sqrt

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


class sq:
    """
    This module is used to calculate static structure factors
    covering unary to senary systems.
    """
    def __init__(
            self,
            snapshots: Snapshots,
            qrange: float=10.0,
            onlypositive: bool=False,
            outputfile: str=None,
            saveqvectors: bool=False,
            ) -> None:
        """
        Initializing S(q) class

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory 
                         (returned by reader.dump_reader.DumpReader)
            2. qrange (float): the wave number range to be calculated, default is 10
            3. onlypositive (bool): whether only consider positive wave vectors
            4. outputfile (str): the name of csv file to save the calculated S(q)
            5. saveqvectors (bool): whether to save S(q) for specific wavevectors

        Return:
            None
        """
        self.snapshots = snapshots
        self.qrange = qrange * 2.0
        self.onlypositive = onlypositive
        self.outputfile = outputfile
        self.saveqvectors = saveqvectors

        self.nsnapshots = snapshots.nsnapshots
        self.ndim = snapshots.snapshots[0].positions.shape[1]

        self.nparticle = snapshots.snapshots[0].nparticle
        assert len({snapshot.nparticle for snapshot in self.snapshots.snapshots}) == 1,\
            "Paticle Number Changes during simulation"
        assert len({tuple(snapshot.boxlength) for snapshot in self.snapshots.snapshots}) == 1,\
            "Simulation Box Length Changes during simulation"
        
        self.type, self.typenumber = np.unique(self.snapshots.snapshots[0].particle_type, return_counts=True)
        assert np.sum(self.typenumber) == self.nparticle,\
            "Sum of Indivdual types is Not the Total Amount"
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typenumber/self.nparticle, 3)])}')

        twopidl = 2 * np.pi / self.snapshots.snapshots[0].boxlength #[2PI/Lx, 2PI/Ly]
        Numofq = int(self.qrange*2.0/twopidl.min())
        self.qvector = choosewavevector(self.ndim, Numofq, self.onlypositive).astype(np.float64) * twopidl[np.newaxis, :]
        self.qvalue = np.linalg.norm(self.qvector, axis=1)
        self.df_qvector = pd.DataFrame(self.qvector, columns=[f"q{i}" for i in range(self.ndim)])
 
    def getresults(self) -> Optional[Callable]:
        """
        Calculating S(q) for system with different particle type numbers

        Return:
            Optional[Callable]
        """
        if len(self.type) == 1:
            return self.unary()
        if len(self.type) == 2: 
            return self.binary()
        if len(self.type) == 3: 
            return self.ternary()
        if len(self.type) == 4:
            return self.quarternary()
        if len(self.type) == 5: 
            return self.quinary()
        if len(self.type) > 6:
            logger.info(f"This is a {len(self.type)} system, only overall S(q) calculated")
            return self.unary()

    def unary(self) -> pd.DataFrame:
        """
        Calculating S(q) for unary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start Calculating S(q) of a Unary System')

        sqresults = pd.DataFrame(0, columns=["q Sq".split()])
        sqresults["q"] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = 0
            for i in range(snapshot.nparticle):
                thetas = (self.qvector*snapshot.positions[i][np.newaxis,:]).sum(axis=1)
                exp_thetas += np.exp(-1j*thetas)
            sqresults["Sq"] += (exp_thetas*np.conj(exp_thetas)).real
        sqresults["Sq"] /= (self.nsnapshots*self.nparticle)
        if self.saveqvectors:
            self.df_qvector.join(sqresults).to_csv(
                self.outputfile+"_qvectors.csv", 
                float_format="%.6f", 
                index=False
            )
        # ensemble average over same q but different directions
        sqresults = sqresults.round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)
            
        logger.info('Finish Calculating S(q) of a Unary System')
        return results

    def binary(self) -> pd.DataFrame:
        """
        Calculating S(q) for binary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start Calculating S(q) for a Binary System')
        sqresults = pd.DataFrame(0, columns=["q Sqall Sq11 Sq22 Sq12".split()])
        sqresults["q"] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = {
                "all": 0,
                "11": 0,
                "22": 0,
            }
            for i in range(snapshot.nparticle):
                thetas = (self.qvector*snapshot.positions[i][np.newaxis,:]).sum(axis=1)
                medium = np.exp(-1j*thetas)
                exp_thetas["all"] += medium
                if snapshot.particle_type[i]==1:
                    exp_thetas["11"] += medium
                else:
                    exp_thetas["22"] += medium
            sqresults["Sqall"] += (exp_thetas["all"]*np.conj(exp_thetas["all"])).real
            sqresults["Sq11"] += (exp_thetas["11"]*np.conj(exp_thetas["11"])).real
            sqresults["Sq22"] += (exp_thetas["22"]*np.conj(exp_thetas["22"])).real
            sqresults["Sq12"] += (exp_thetas["11"]*np.conj(exp_thetas["22"])).real

        sqresults["Sqall"] /= (self.nsnapshots*self.nparticle)
        sqresults["Sq11"] /= (self.nsnapshots*self.typenumber[0])
        sqresults["Sq22"] /= (self.nsnapshots*self.typenumber[1])
        sqresults["Sq12"] /= (self.nsnapshots*sqrt(self.typenumber[0]*self.typenumber[1]))
        if self.saveqvectors:
            self.df_qvector.join(sqresults).to_csv(
                self.outputfile+"_qvectors.csv", 
                float_format="%.6f", 
                index=False
            )
        # ensemble average over same q but different directions
        sqresults = sqresults.round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating S(q) of a Binary System')
        return results

    def ternary(self) -> pd.DataFrame:
        """
        Calculating S(q) for ternary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start Calculating S(q) of a Ternary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typenumber/self.nparticle, 3)])}')

        sqresults = np.zeros((self.qvector.shape[0], 5))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = 0
            exp_thetas_11 = 0
            exp_thetas_22 = 0
            exp_thetas_33 = 0
            for i in range(snapshot.nparticle):
                thetas = (self.qvector*snapshot.positions[i][np.newaxis,:]).sum(axis=1)
                medium = np.exp(-1j*thetas)
                exp_thetas += medium
                if snapshot.particle_type[i] == 1:
                    exp_thetas_11 += medium
                elif snapshot.particle_type[i] == 2:
                    exp_thetas_22 += medium
                else:
                    exp_thetas_33 += medium
            sqresults[:, 1] += (exp_thetas*np.conj(exp_thetas)).real
            sqresults[:, 2] += (exp_thetas_11*np.conj(exp_thetas_11)).real
            sqresults[:, 3] += (exp_thetas_22*np.conj(exp_thetas_22)).real
            sqresults[:, 4] += (exp_thetas_33*np.conj(exp_thetas_33)).real
        sqresults[:, 1] /= (self.nsnapshots*self.nparticle)
        sqresults[:, 2] /= (self.nsnapshots*self.typenumber[0])
        sqresults[:, 3] /= (self.nsnapshots*self.typenumber[1])
        sqresults[:, 4] /= (self.nsnapshots*self.typenumber[2])
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)'
        sqresults = pd.DataFrame(sqresults, columns=names.split()).round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating S(q) of a Ternary System')
        return results

    def quarternary(self) -> pd.DataFrame:
        """
        Calculating S(q) for quarternary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start Calculating S(q) of a Quarternary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typenumber/self.nparticle, 3)])}')

        sqresults = np.zeros((self.qvector.shape[0], 6))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = 0
            exp_thetas_11 = 0
            exp_thetas_22 = 0
            exp_thetas_33 = 0
            exp_thetas_44 = 0
            for i in range(snapshot.nparticle):
                thetas = (self.qvector*snapshot.positions[i][np.newaxis,:]).sum(axis=1)
                medium = np.exp(-1j*thetas)
                exp_thetas += medium
                if snapshot.particle_type[i] == 1:
                    exp_thetas_11 += medium
                elif snapshot.particle_type[i] == 2:
                    exp_thetas_22 += medium
                elif snapshot.particle_type[i] == 3:
                    exp_thetas_33 += medium
                else:
                    exp_thetas_44 += medium
            sqresults[:, 1] += (exp_thetas*np.conj(exp_thetas)).real
            sqresults[:, 2] += (exp_thetas_11*np.conj(exp_thetas_11)).real
            sqresults[:, 3] += (exp_thetas_22*np.conj(exp_thetas_22)).real
            sqresults[:, 4] += (exp_thetas_33*np.conj(exp_thetas_33)).real
            sqresults[:, 5] += (exp_thetas_44*np.conj(exp_thetas_44)).real
        sqresults[:, 1] /= (self.nsnapshots*self.nparticle)
        sqresults[:, 2] /= (self.nsnapshots*self.typenumber[0])
        sqresults[:, 3] /= (self.nsnapshots*self.typenumber[1])
        sqresults[:, 4] /= (self.nsnapshots*self.typenumber[2])
        sqresults[:, 5] /= (self.nsnapshots*self.typenumber[3])
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)'
        sqresults = pd.DataFrame(sqresults, columns=names.split()).round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating S(q) of a Quarternary System')
        return results

    def quinary(self) -> pd.DataFrame:
        """
        Calculating S(q) for quinary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start Calculating S(q) of a Quinary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typenumber/self.nparticle, 3)])}')

        sqresults = np.zeros((self.qvector.shape[0], 7))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            exp_thetas = 0
            exp_thetas_11 = 0
            exp_thetas_22 = 0
            exp_thetas_33 = 0
            exp_thetas_44 = 0
            exp_thetas_55 = 0
            for i in range(snapshot.nparticle):
                thetas = (self.qvector*snapshot.positions[i][np.newaxis,:]).sum(axis=1)
                medium = np.exp(-1j*thetas)
                exp_thetas += medium
                if snapshot.particle_type[i] == 1:
                    exp_thetas_11 += medium
                elif snapshot.particle_type[i] == 2:
                    exp_thetas_22 += medium
                elif snapshot.particle_type[i] == 3:
                    exp_thetas_33 += medium
                elif snapshot.particle_type[i] == 4:
                    exp_thetas_44 += medium
                else:
                    exp_thetas_55 += medium
            sqresults[:, 1] += (exp_thetas*np.conj(exp_thetas)).real
            sqresults[:, 2] += (exp_thetas_11*np.conj(exp_thetas_11)).real
            sqresults[:, 3] += (exp_thetas_22*np.conj(exp_thetas_22)).real
            sqresults[:, 4] += (exp_thetas_33*np.conj(exp_thetas_33)).real
            sqresults[:, 5] += (exp_thetas_44*np.conj(exp_thetas_44)).real
            sqresults[:, 6] += (exp_thetas_55*np.conj(exp_thetas_55)).real
        sqresults[:, 1] /= (self.nsnapshots*self.nparticle)
        sqresults[:, 2] /= (self.nsnapshots*self.typenumber[0])
        sqresults[:, 3] /= (self.nsnapshots*self.typenumber[1])
        sqresults[:, 4] /= (self.nsnapshots*self.typenumber[2])
        sqresults[:, 5] /= (self.nsnapshots*self.typenumber[3])
        sqresults[:, 6] /= (self.nsnapshots*self.typenumber[4])
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)  S55(q)'
        sqresults = pd.DataFrame(sqresults, columns=names.split()).round(6)
        results = sqresults.groupby(sqresults["q"]).mean().reset_index()
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating S(q) of a Quinary System')
        return results
