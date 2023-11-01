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
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


class sq:
    """
    This module is used to calculate static structure factors
    covering unary to senary systems.
    """
    def __init__(self, snapshots: Snapshots, qrange=10, onlypositive=False) -> None:
        """
        Initializing sq class

        Inputs:
            snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory 
                      (returned by reader.dump_reader.DumpReader)

        Return:
            None
        """
        self.snapshots = snapshots
        self.nsnapshots = snapshots.nsnapshots
        self.ndim = snapshots.snapshots[0].positions.shape[1]
        self.nparticle = snapshots.snapshots[0].nparticle
        assert snapshots.snapshots[0].nparticle == snapshots.snapshots[-1].nparticle,\
            "Paticle Number Changes during simulation"
        self.boxlength = snapshots.snapshots[0].boxlength
        assert (snapshots.snapshots[0].boxlength == snapshots.snapshots[-1].boxlength).all(),\
            "Paticle Number Changes during simulation"
        self.typecounts = np.unique(snapshots.snapshots[0].particle_type, return_counts = True)
        self.type = self.typecounts[0]
        self.typenumber = self.typecounts[1]
        assert np.sum(self.typenumber) == self.nparticle,\
            "Sum of Indivdual types is Not the Total Amount"
        
        self.twopidl = 2 * np.pi / self.boxlength #[2PI/Lx, 2PI/Ly]
        #set up wave vectors
        self.qrange = qrange*2.0
        self.onlypositive = onlypositive #only consider positive wave vectors
        Numofq = int(self.qrange*2.0 / self.twopidl.min())
        self.qvector = choosewavevector(self.ndim, Numofq, self.onlypositive).astype(np.float64)
        self.qvector *= self.twopidl[np.newaxis, :]
        self.qvalue = np.linalg.norm(self.qvector, axis=1)

    def getresults(self, outputfile: str=None) -> Optional[Callable]:
        """
        Calculating S(q) for system with different particle type numbers

        Inputs:
            outputfile (str): the file name to save the calculated S(q)

        Return: Optional[Callable]
        """
        if len(self.type) == 1:
            return self.unary(outputfile)
        if len(self.type) == 2: 
            return self.binary(outputfile)
        if len(self.type) == 3: 
            return self.ternary(outputfile)
        if len(self.type) == 4: 
            return self.quarternary(outputfile)
        if len(self.type) == 5: 
            return self.quinary(outputfile)
        if len(self.type) == 6: 
            return self.senary(outputfile)
        if len(self.type) > 6:
            logger.info('This is a system with more than 6 species, only overall Sq is calculated')
            return self.unary(outputfile)

    def unary(self, outputfile: str=None) -> None:
        """
        Calculating SF for unary system
        
        Inputs:
            outputfile (str): the file name to save the calculated S(q)

        Return:
            None [output calculated S(q) to a document]
        """
        logger.info('Start Calculating SF of a Unary System')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        sqresults = np.zeros((self.qvector.shape[0], 2))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            sqtotal = np.zeros_like(sqresults) #[sin cos]
            for i in range(self.nparticle):
                thetas = (snapshot.positions[i] * self.qvector).sum(axis=1)
                sqtotal[:, 0] += np.sin(thetas)
                sqtotal[:, 1] += np.cos(thetas)    
            sqresults[:, 1] += np.square(sqtotal).sum(axis=1)
        sqresults[:, 1]  = sqresults[:, 1] / self.nsnapshots / self.nparticle
        
        sqresults = pd.DataFrame(sqresults).round(6)
        results   = sqresults.groupby(sqresults[0]).mean().reset_index().values
        names = 'q  S(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header=names, comments='')
        logger.info('Finish Calculating SF of a Unary System')

    def binary(self, outputfile=None) -> None:
        """
        Calculating SF for binary system
        
        Inputs:
            outputfile (str): the file name to save the calculated S(q)

        Return:
            None [output calculated S(q) to a document]
        """
        logger.info('Start Calculating SF of a Binary System')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        sqresults = np.zeros((self.qvector.shape[0], 4))
        sqresults[:, 0] = self.qvalue
        for snapshot in self.snapshots.snapshots:
            sqtotal = np.zeros((self.qvector.shape[0], 2))
            sq11    = np.zeros_like(sqtotal)
            sq22    = np.zeros_like(sqtotal)
            for i in range(self.nparticle):
                thetas = (snapshot.positions[i] * self.qvector).sum(axis=1)
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
        results   = sqresults.groupby(sqresults[0]).mean().reset_index().values
        names = 'q  S(q)  S11(q)  S22(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        logger.info('Finish Calculating SF of a Binary System')

    def ternary(self, outputfile=None) -> None:
        """
        Calculating SF for ternary system
        
        Inputs:
            outputfile (str): the file name to save the calculated S(q)

        Return:
            None [output calculated S(q) to a document]
        """
        logger.info('Start Calculating SF of a Ternary System')
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
        results   = sqresults.groupby(sqresults[0]).mean().reset_index().values
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        logger.info('Finish Calculating SF of a Ternary System')

    def quarternary(self, outputfile=None) -> None:
        """
        Calculating SF for quarternary system
        
        Inputs:
            outputfile (str): the file name to save the calculated S(q)

        Return:
            None [output calculated S(q) to a document]
        """
        logger.info('Start Calculating SF of a Quarternary System')
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
        results   = sqresults.groupby(sqresults[0]).mean().reset_index().values
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        logger.info('Finish Calculating SF of a Quarternary System')

    def quinary(self, outputfile=None) -> None:
        """
        Calculating SF for quinary system
        
        Inputs:
            outputfile (str): the file name to save the calculated S(q)

        Return:
            None [output calculated S(q) to a document]
        """
        logger.info('Start Calculating SF of a Quinary System')
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
        results   = sqresults.groupby(sqresults[0]).mean().reset_index().values
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)  S55(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        logger.info('Finish Calculating SF of a Quinary System')

    def senary(self, outputfile=None) -> None:
        """
        Calculating SF for senary system
        
        Inputs:
            outputfile (str): the file name to save the calculated S(q)

        Return:
            None [output calculated S(q) to a document]
        """
        logger.info('Start Calculating SF of a Senary System')
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
        results   = sqresults.groupby(sqresults[0]).mean().reset_index().values
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)  S55(q)  S66(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        logger.info('Finish Calculating SF of a Senary System')
        logger.info('Only the overall g(r) is calculated')
