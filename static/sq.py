# coding = utf-8

"""see documentation @ ../docs/static.md"""

from typing import Optional, Callable
import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from utils.wavevector import choosewavevector
from utils.logging_utils import get_logger_handle

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
            outputfile: str=None
            ) -> None:
        """
        Initializing S(q) class

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory 
                         (returned by reader.dump_reader.DumpReader)
            2. qrange (float): the wave number range to be calculated, default is 10
            3. onlypositive (bool): whether only consider positive wave vectors
            4. outputfile (str): the name of csv file to save the calculated S(q)

        Return:
            None
        """
        self.snapshots = snapshots
        self.qrange = qrange * 2.0
        self.onlypositive = onlypositive
        self.outputfile = outputfile

        self.nsnapshots = snapshots.nsnapshots
        self.ndim = snapshots.snapshots[0].positions.shape[1]
        self.nparticle = snapshots.snapshots[0].nparticle
        assert snapshots.snapshots[0].nparticle == snapshots.snapshots[-1].nparticle,\
            "Paticle Number Changes during simulation"
        self.boxlength = snapshots.snapshots[0].boxlength
        assert (snapshots.snapshots[0].boxlength == snapshots.snapshots[-1].boxlength).all(),\
            "Paticle Number Changes during simulation"
        self.typecounts = np.unique(snapshots.snapshots[0].particle_type, return_counts=True)
        self.type = self.typecounts[0]
        self.typenumber = self.typecounts[1]
        assert np.sum(self.typenumber) == self.nparticle,\
            "Sum of Indivdual types is Not the Total Amount"
        
        self.twopidl = 2 * np.pi / self.boxlength #[2PI/Lx, 2PI/Ly]
        Numofq = int(self.qrange*2.0/self.twopidl.min())
        self.qvector = choosewavevector(self.ndim, Numofq, self.onlypositive).astype(np.float64)
        self.qvector *= self.twopidl[np.newaxis, :]
        self.qvalue = np.linalg.norm(self.qvector, axis=1)
        
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
        if len(self.type) == 6: 
            return self.senary()
        if len(self.type) > 6:
            logger.info('This is a system with more than 6 species, only overall S(q) is calculated')
            return self.unary()

    def unary(self) -> pd.DataFrame:
        """
        Calculating S(q) for unary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start Calculating S(q) of a Unary System')

        sqresults = np.zeros((self.qvector.shape[0], 2))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            sqtotal = np.zeros_like(sqresults) #[sin cos]
            for i in range(self.nparticle):
                thetas = (snapshot.positions[i]*self.qvector).sum(axis=1)
                sqtotal[:, 0] += np.sin(thetas)
                sqtotal[:, 1] += np.cos(thetas)    
            sqresults[:, 1] += np.square(sqtotal).sum(axis=1)
        sqresults[:, 1] = sqresults[:, 1] / self.nsnapshots / self.nparticle

        sqresults = pd.DataFrame(sqresults).round(6)
        names = 'q  S(q)'
        results = pd.DataFrame(sqresults.groupby(sqresults[0]).mean().reset_index().values, columns=names.split())
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
        logger.info('Start Calculating S(q) of a Binary System')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        sqresults = np.zeros((self.qvector.shape[0], 4))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            sqtotal = np.zeros((self.qvector.shape[0], 2))
            sq11 = np.zeros_like(sqtotal)
            sq22 = np.zeros_like(sqtotal)
            for i in range(self.nparticle):
                thetas = (snapshot.positions[i]*self.qvector).sum(axis=1)
                sin_parts = np.sin(thetas)
                cos_parts = np.cos(thetas)
                sqtotal[:, 0] += sin_parts
                sqtotal[:, 1] += cos_parts
                if snapshot.particle_type[i] == 1: 
                    sq11[:, 0] += sin_parts
                    sq11[:, 1] += cos_parts
                if snapshot.particle_type[i] == 2: 
                    sq22[:, 0] += sin_parts
                    sq22[:, 1] += cos_parts
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis=1) / self.nparticle
            sqresults[:, 2] += np.square(sq11).sum(axis=1) / self.typenumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis=1) / self.typenumber[1]
        sqresults[:, 1:] /= self.nsnapshots

        sqresults = pd.DataFrame(sqresults).round(6)
        names = 'q  S(q)  S11(q)  S22(q)'
        results = pd.DataFrame(sqresults.groupby(sqresults[0]).mean().reset_index().values, columns=names.split())
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
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        sqresults = np.zeros((self.qvector.shape[0], 5))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            sqtotal = np.zeros((self.qvector.shape[0], 2))
            sq11 = np.zeros_like(sqtotal)
            sq22 = np.zeros_like(sqtotal)
            sq33 = np.zeros_like(sqtotal)
            for i in range(self.nparticle):
                thetas = (snapshot.positions[i] * self.qvector).sum(axis=1)
                sin_parts = np.sin(thetas)
                cos_parts = np.cos(thetas)
                sqtotal[:, 0] += sin_parts
                sqtotal[:, 1] += cos_parts
                if snapshot.particle_type[i] == 1: 
                    sq11 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 2: 
                    sq22 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 3: 
                    sq33 += np.column_stack((sin_parts, cos_parts))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis=1) / self.nparticle
            sqresults[:, 2] += np.square(sq11).sum(axis=1) / self.typenumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis=1) / self.typenumber[1]
            sqresults[:, 4] += np.square(sq33).sum(axis=1) / self.typenumber[2]
        sqresults[:, 1:] /= self.nsnapshots

        sqresults = pd.DataFrame(sqresults).round(6)
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)'
        results = pd.DataFrame(sqresults.groupby(sqresults[0]).mean().reset_index().values, columns=names.split())
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
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        sqresults = np.zeros((self.qvector.shape[0], 6))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            sqtotal = np.zeros((self.qvector.shape[0], 2))
            sq11 = np.zeros_like(sqtotal)
            sq22 = np.zeros_like(sqtotal)
            sq33 = np.zeros_like(sqtotal)
            sq44 = np.zeros_like(sqtotal)
            for i in range(self.nparticle):
                thetas = (snapshot.positions[i] * self.qvector).sum(axis=1)
                sin_parts = np.sin(thetas)
                cos_parts = np.cos(thetas)
                sqtotal += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 1: 
                    sq11 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 2: 
                    sq22 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 3: 
                    sq33 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 4: 
                    sq44 += np.column_stack((sin_parts, cos_parts))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis=1) / self.nparticle
            sqresults[:, 2] += np.square(sq11).sum(axis=1) / self.typenumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis=1) / self.typenumber[1]
            sqresults[:, 4] += np.square(sq33).sum(axis=1) / self.typenumber[2]
            sqresults[:, 5] += np.square(sq44).sum(axis=1) / self.typenumber[3]
        sqresults[:, 1:] /= self.nsnapshots

        sqresults = pd.DataFrame(sqresults).round(6)
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)'
        results = pd.DataFrame(sqresults.groupby(sqresults[0]).mean().reset_index().values, columns=names.split())
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
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        sqresults = np.zeros((self.qvector.shape[0], 7))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            sqtotal = np.zeros((self.qvector.shape[0], 2))
            sq11 = np.zeros_like(sqtotal)
            sq22 = np.zeros_like(sqtotal)
            sq33 = np.zeros_like(sqtotal)
            sq44 = np.zeros_like(sqtotal)
            sq55 = np.zeros_like(sqtotal)
            for i in range(self.nparticle):
                thetas = (snapshot.positions[i] * self.qvector).sum(axis=1)
                sin_parts = np.sin(thetas)
                cos_parts = np.cos(thetas)
                sqtotal += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 1: 
                    sq11 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 2: 
                    sq22 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 3: 
                    sq33 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 4: 
                    sq44 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 5: 
                    sq55 += np.column_stack((sin_parts, cos_parts))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.nparticle
            sqresults[:, 2] += np.square(sq11).sum(axis = 1) / self.typenumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis = 1) / self.typenumber[1]
            sqresults[:, 4] += np.square(sq33).sum(axis = 1) / self.typenumber[2]
            sqresults[:, 5] += np.square(sq44).sum(axis = 1) / self.typenumber[3]
            sqresults[:, 6] += np.square(sq55).sum(axis = 1) / self.typenumber[4]
        sqresults[:, 1:] /= self.nsnapshots

        sqresults = pd.DataFrame(sqresults).round(6)
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)  S55(q)'
        results = pd.DataFrame(sqresults.groupby(sqresults[0]).mean().reset_index().values, columns=names.split())
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)
        logger.info('Finish Calculating S(q) of a Quinary System')

        return results

    def senary(self) -> pd.DataFrame:
        """
        Calculating S(q) for senary system

        Return:
            calculated S(q) (pd.DataFrame)
        """
        logger.info('Start Calculating S(q) of a Senary System')
        logger.info('Only calculate the overall S(q) at this stage')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        sqresults = np.zeros((self.qvector.shape[0], 8))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            sqtotal = np.zeros((self.qvector.shape[0], 2))
            sq11 = np.zeros_like(sqtotal)
            sq22 = np.zeros_like(sqtotal)
            sq33 = np.zeros_like(sqtotal)
            sq44 = np.zeros_like(sqtotal)
            sq55 = np.zeros_like(sqtotal)
            sq66 = np.zeros_like(sqtotal)
            for i in range(self.nparticle):
                thetas = (snapshot.positions[i] * self.qvector).sum(axis=1)
                sin_parts = np.sin(thetas)
                cos_parts = np.cos(thetas)
                sqtotal += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 1: 
                    sq11 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 2: 
                    sq22 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 3: 
                    sq33 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 4: 
                    sq44 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 5: 
                    sq55 += np.column_stack((sin_parts, cos_parts))
                if snapshot.particle_type[i] == 6: 
                    sq66 += np.column_stack((sin_parts, cos_parts))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.nparticle
            sqresults[:, 2] += np.square(sq11).sum(axis = 1) / self.typenumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis = 1) / self.typenumber[1]
            sqresults[:, 4] += np.square(sq33).sum(axis = 1) / self.typenumber[2]
            sqresults[:, 5] += np.square(sq44).sum(axis = 1) / self.typenumber[3]
            sqresults[:, 6] += np.square(sq55).sum(axis = 1) / self.typenumber[4]
            sqresults[:, 7] += np.square(sq66).sum(axis = 1) / self.typenumber[5]
        sqresults[:, 1:] /= self.nsnapshots

        sqresults = pd.DataFrame(sqresults).round(6)
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)  S55(q)  S66(q)'
        results = pd.DataFrame(sqresults.groupby(sqresults[0]).mean().reset_index().values, columns=names.split())
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)
        logger.info('Finish Calculating S(q) of a Senary System')
        logger.info('Only the overall S(q) is calculated')

        return results
