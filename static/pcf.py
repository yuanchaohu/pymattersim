# coding = utf-8

"""see documentation @ ../docs/static.md"""

from typing import Optional, Callable
import numpy as np
from reader.reader_utils import Snapshots
from utils.pbc import remove_pbc
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


class gr:
    """
    This module is used to calculate pair correlation functions
    covering unary to senary systems.
    """
    def __init__(self, snapshots: Snapshots) -> None:
        """
        Initializing gr class

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
        self.boxvolume = np.prod(self.boxlength)
        self.typecounts = np.unique(snapshots.snapshots[0].particle_type, return_counts = True)
        self.type = self.typecounts[0]
        self.typenumber = self.typecounts[1]
        assert np.sum(self.typenumber) == self.nparticle,\
            "Sum of Indivdual Types is Not the Total Amount"
        self.nidealfac = 4.0 / 3 if self.ndim == 3 else 1.0
        self.rhototal = self.nparticle / self.boxvolume
        self.rhotype = self.typenumber / self.boxvolume

    def getresults(
            self,
            ppp: list=[1,1,1],
            rdelta: float=0.01,
            outputfile: str=None) -> Optional[Callable]:
        """
        Calculating g(r) for system with different particle type numbers

        Inputs:
            1. ppp (list): the periodic boundary conditions,
                           setting 1 for yes and 0 for no, default [1,1,1],
                           that is, PBC is applied in all three dimensions for 3D box.
                           set [1, 1] for two-dimensional systems
            2. rdelta (float): bin size calculating g(r), the default value is 0.01

        Return: Optional[Callable]
        """
        if len(self.type) == 1:
            return self.unary(ppp, rdelta, outputfile)
        if len(self.type) == 2:
            return self.binary(ppp, rdelta, outputfile)
        if len(self.type) == 3:
            return self.ternary(ppp, rdelta, outputfile)
        if len(self.type) == 4:
            return self.quarternary(ppp, rdelta, outputfile)
        if len(self.type) == 5:
            return self.quinary(ppp, rdelta, outputfile)
        if len(self.type) == 6:
            return self.senary(ppp, rdelta, outputfile)
        if len(self.type) > 6:
            logger.info('This is a system with more than 6 species, only overall gr is calculated')
            return self.unary(ppp, rdelta, outputfile)

    def unary(self, ppp: list, rdelta: float=0.01, outputfile :str=None) -> None:
        """
        Calculating PCF for unary system
        
        Inputs:
            1. ppp (list): the periodic boundary conditions,
                           setting 1 for yes and 0 for no, default [1,1,1],
                           that is, PBC is applied in all three dimensions for 3D box.
                           set [1, 1] for two-dimensional systems
            2. rdelta (float): bin size calculating g(r), the default value is 0.01
            3. outputfile (str): the file name to save the calculated g(r)

        Return:
            None [output calculated g(r) to a document]
        """
        logger.info('Start Calculating PCF of a Unary System')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        maxbin = int(self.boxlength.min() / 2.0 / rdelta)
        grresults = np.zeros(maxbin)
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))
                countvalue, binedge = np.histogram(distance, bins=maxbin, range=(0,maxbin*rdelta))
                grresults += countvalue
        binleft = binedge[:-1]
        binright = binedge[1:]
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults = grresults * 2 / self.nparticle / self.nsnapshots / (nideal * self.rhototal)

        # middle of each bin
        binright = binright - 0.5 * rdelta
        results  = np.column_stack((binright, grresults))
        names = 'r  g(r)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        logger.info('Finish Calculating PCF of a Unary System')

    def binary(self, ppp: list, rdelta: float=0.01, outputfile :str=None) -> None:
        """
        Calculating PCF for binary system
        
        Inputs:
            1. ppp (list): the periodic boundary conditions,
                           setting 1 for yes and 0 for no, default [1,1,1],
                           that is, PBC is applied in all three dimensions for 3D box.
                           set [1, 1] for two-dimensional systems
            2. rdelta (float): bin size calculating g(r), the default value is 0.01
            3. outputfile (str): the file name to save the calculated g(r)

        Return:
            None [output calculated g(r) to a document]
        """
        logger.info('Start Calculating PCF of a Binary System')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        maxbin = int(self.boxlength.min() / 2.0 / rdelta)
        grresults = np.zeros((maxbin, 4))
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle - 1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:],
                            np.zeros_like(snapshot.particle_type[i+1:]) + snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))

                countvalue, binedge = np.histogram(distance, bins=maxbin, range=(0, maxbin*rdelta))
                grresults[:, 0] += countvalue

                countsum = TIJ.sum(axis = 1)
                countvalue, binedge = np.histogram(distance[countsum == 2],
                                                   bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 1] += countvalue
                countvalue, binedge = np.histogram(distance[countsum == 3],
                                                   bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 2] += countvalue
                countvalue, binedge = np.histogram(distance[countsum == 4],
                                                   bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 3] += countvalue

        binleft = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults[:, 0] = grresults[:, 0] * 2 / self.nsnapshots /self.nparticle / (nideal * self.rhototal)
        grresults[:, 1] = grresults[:, 1] * 2 / self.nsnapshots / self.typenumber[0] / (nideal * self.rhotype[0])
        grresults[:, 2] = grresults[:, 2] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[0] / self.typenumber[1] / 2.0
        grresults[:, 3] = grresults[:, 3] * 2 / self.nsnapshots / self.typenumber[1] / (nideal * self.rhotype[1])

        binright = binright - 0.5 * rdelta #middle of each bin
        results  = np.column_stack((binright, grresults))
        names    = 'r  g(r)  g11(r)  g12(r)  g22(r)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        logger.info('Finish Calculating PCF of a Binary System')


    def ternary(self, ppp: list, rdelta: float=0.01, outputfile :str=None) -> None:
        """
        Calculating PCF for ternary system
        
        Inputs:
            1. ppp (list): the periodic boundary conditions,
                           setting 1 for yes and 0 for no, default [1,1,1],
                           that is, PBC is applied in all three dimensions for 3D box.
                           set [1, 1] for two-dimensional systems
            2. rdelta (float): bin size calculating g(r), the default value is 0.01
            3. outputfile (str): the file name to save the calculated g(r)

        Return:
            None [output calculated g(r) to a document]
        """
        logger.info('Start Calculating PCF of a Ternary System')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        maxbin = int(self.boxlength.min() / 2.0 / rdelta)
        grresults   = np.zeros((maxbin, 7))
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle - 1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:], np.zeros_like(snapshot.particle_type[i+1:]) + snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))

                countvalue, binedge = np.histogram(distance, bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 0] += countvalue

                countsum   = TIJ.sum(axis = 1)
                countsub   = np.abs(TIJ[:, 0] - TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum  == 2], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 1] += countvalue # 11
                countvalue, binedge = np.histogram(distance[(countsum == 4) & (countsub == 0)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 2] += countvalue # 22
                countvalue, binedge = np.histogram(distance[countsum  == 6], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 3] += countvalue # 33
                countvalue, binedge = np.histogram(distance[countsum  == 3], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 4] += countvalue # 12
                countvalue, binedge = np.histogram(distance[countsum  == 5], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 5] += countvalue # 23
                countvalue, binedge = np.histogram(distance[(countsum == 4) & (countsub == 2)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 6] += countvalue # 13

        binleft  = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults[:, 0]  = grresults[:, 0] * 2 / self.nsnapshots / self.nparticle / (nideal * self.rhototal)
        grresults[:, 1]  = grresults[:, 1] * 2 / self.nsnapshots / self.typenumber[0] / (nideal * self.rhotype[0])
        grresults[:, 2]  = grresults[:, 2] * 2 / self.nsnapshots / self.typenumber[1] / (nideal * self.rhotype[1])
        grresults[:, 3]  = grresults[:, 3] * 2 / self.nsnapshots / self.typenumber[2] / (nideal * self.rhotype[2])
        grresults[:, 4]  = grresults[:, 4] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[0] / self.typenumber[1] / 2.0
        grresults[:, 5]  = grresults[:, 5] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[1] / self.typenumber[2] / 2.0
        grresults[:, 6]  = grresults[:, 6] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[0] / self.typenumber[2] / 2.0

        binright = binright - 0.5 * rdelta #middle of each bin
        results  = np.column_stack((binright, grresults))
        names    = 'r  g(r)  g11(r)  g22(r)  g33(r)  g12(r)  g23(r)  g13(r)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        logger.info('Finish Calculating PCF of a Ternary System')

    def quarternary(self, ppp: list, rdelta: float=0.01, outputfile :str=None) -> None:
        """
        Calculating PCF for quarternary system
        
        Inputs:
            1. ppp (list): the periodic boundary conditions,
                           setting 1 for yes and 0 for no, default [1,1,1],
                           that is, PBC is applied in all three dimensions for 3D box.
                           set [1, 1] for two-dimensional systems
            2. rdelta (float): bin size calculating g(r), the default value is 0.01
            3. outputfile (str): the file name to save the calculated g(r)

        Return:
            None [output calculated g(r) to a document]
        """
        logger.info('Start Calculating PCF of a Quarternary System')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        maxbin = int(self.boxlength.min() / 2.0 / rdelta)
        grresults = np.zeros((maxbin, 11))
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle - 1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:], np.zeros_like(snapshot.particle_type[i+1:]) + snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))

                countvalue, binedge = np.histogram(distance, bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 0] += countvalue

                countsum   = TIJ.sum(axis = 1)
                countsub   = np.abs(TIJ[:, 0] - TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum == 2], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 1] += countvalue #11
                countvalue, binedge = np.histogram(distance[(countsum  == 4) & (countsub == 0)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 2] += countvalue #22
                countvalue, binedge = np.histogram(distance[(countsum  == 6) & (countsub == 0)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 3] += countvalue #33
                countvalue, binedge = np.histogram(distance[countsum  == 8], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 4] += countvalue #44
                countvalue, binedge = np.histogram(distance[countsum  == 3], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 5] += countvalue #12
                countvalue, binedge = np.histogram(distance[(countsum  == 4) & (countsub == 2)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 6] += countvalue #13
                countvalue, binedge = np.histogram(distance[countsub  == 3], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 7] += countvalue #14
                countvalue, binedge = np.histogram(distance[(countsum  == 5) & (countsub == 1)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 8] += countvalue #23
                countvalue, binedge = np.histogram(distance[(countsum  == 6) & (countsub == 2)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 9] += countvalue #24
                countvalue, binedge = np.histogram(distance[countsum  == 7], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:,10] += countvalue #34

        binleft  = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults[:, 0] = grresults[:, 0] * 2 / self.nsnapshots / self.nparticle / (nideal * self.rhototal)
        grresults[:, 1] = grresults[:, 1] * 2 / self.nsnapshots / self.typenumber[0] / (nideal * self.rhotype[0])
        grresults[:, 2] = grresults[:, 2] * 2 / self.nsnapshots / self.typenumber[1] / (nideal * self.rhotype[1])
        grresults[:, 3] = grresults[:, 3] * 2 / self.nsnapshots / self.typenumber[2] / (nideal * self.rhotype[2])
        grresults[:, 4] = grresults[:, 4] * 2 / self.nsnapshots / self.typenumber[3] / (nideal * self.rhotype[3])

        grresults[:, 5] = grresults[:, 5] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[0] / self.typenumber[1] / 2.0
        grresults[:, 6] = grresults[:, 6] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[0] / self.typenumber[2] / 2.0
        grresults[:, 7] = grresults[:, 7] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[0] / self.typenumber[3] / 2.0
        grresults[:, 8] = grresults[:, 8] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[1] / self.typenumber[2] / 2.0
        grresults[:, 9] = grresults[:, 9] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[1] / self.typenumber[3] / 2.0
        grresults[:,10] = grresults[:,10] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typenumber[2] / self.typenumber[3] / 2.0

        binright = binright - 0.5 * rdelta #middle of each bin
        results  = np.column_stack((binright, grresults))
        names = 'r  g(r)  g11(r)  g22(r)  g33(r)  g44(r)  g12(r)  g13(r)  g14(r)  g23(r)  g24(r)  g34(r)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header=names, comments='')

        logger.info('Finish Calculating PCF of a Quarternary System')

    def quinary(self, ppp: list, rdelta: float=0.01, outputfile :str=None) -> None:
        """
        Calculating PCF for quinary system
        
        Inputs:
            1. ppp (list): the periodic boundary conditions,
                           setting 1 for yes and 0 for no, default [1,1,1],
                           that is, PBC is applied in all three dimensions for 3D box.
                           set [1, 1] for two-dimensional systems
            2. rdelta (float): bin size calculating g(r), the default value is 0.01
            3. outputfile (str): the file name to save the calculated g(r)

        Return:
            None [output calculated g(r) to a document]
        """
        logger.info('Start Calculating PCF of a Quinary System')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        maxbin = int(self.boxlength.min() / 2.0 / rdelta)
        grresults   = np.zeros((maxbin, 16))
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle - 1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:], np.zeros_like(snapshot.particle_type[i+1:]) + snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))

                countvalue, binedge = np.histogram(distance, bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 0] += countvalue

                countsum   = TIJ.sum(axis = 1)
                countsub   = np.abs(TIJ[:, 0] - TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum == 2], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 1] += countvalue #11
                countvalue, binedge = np.histogram(distance[(countsum  == 4) & (countsub == 0)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 2] += countvalue #22
                countvalue, binedge = np.histogram(distance[(countsum  == 6) & (countsub == 0)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 3] += countvalue #33
                countvalue, binedge = np.histogram(distance[(countsum  == 8) & (countsub == 0)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 4] += countvalue #44
                countvalue, binedge = np.histogram(distance[countsum  == 10], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 5] += countvalue #55
                countvalue, binedge = np.histogram(distance[countsum  == 3], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 6] += countvalue #12
                countvalue, binedge = np.histogram(distance[(countsum  == 4) & (countsub == 2)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 7] += countvalue #13
                countvalue, binedge = np.histogram(distance[(countsum  == 5) & (countsub == 3)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 8] += countvalue #14
                countvalue, binedge = np.histogram(distance[(countsum  == 6) & (countsub == 4)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:, 9] += countvalue #15
                countvalue, binedge = np.histogram(distance[(countsum  == 5) & (countsub == 1)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:,10] += countvalue #23
                countvalue, binedge = np.histogram(distance[(countsum  == 6) & (countsub == 2)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:,11] += countvalue #24
                countvalue, binedge = np.histogram(distance[(countsum  == 7) & (countsub == 3)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:,12] += countvalue #25
                countvalue, binedge = np.histogram(distance[(countsum  == 7) & (countsub == 1)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:,13] += countvalue #34
                countvalue, binedge = np.histogram(distance[(countsum  == 8) & (countsub == 2)], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:,14] += countvalue #35
                countvalue, binedge = np.histogram(distance[countsum  == 9], bins = maxbin, range = (0, maxbin * rdelta))
                grresults[:,15] += countvalue #45

        binleft = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults[:, 0]  = grresults[:, 0] * 2 / self.nsnapshots/ self.nparticle / (nideal * self.rhototal)
        grresults[:, 1]  = grresults[:, 1] * 2 / self.nsnapshots/ self.typenumber[0] / (nideal * self.rhotype[0])
        grresults[:, 2]  = grresults[:, 2] * 2 / self.nsnapshots/ self.typenumber[1] / (nideal * self.rhotype[1])
        grresults[:, 3]  = grresults[:, 3] * 2 / self.nsnapshots/ self.typenumber[2] / (nideal * self.rhotype[2])
        grresults[:, 4]  = grresults[:, 4] * 2 / self.nsnapshots/ self.typenumber[3] / (nideal * self.rhotype[3])
        grresults[:, 5]  = grresults[:, 5] * 2 / self.nsnapshots/ self.typenumber[4] / (nideal * self.rhotype[4])

        grresults[:, 6]  = grresults[:, 6] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[0] / self.typenumber[1] /2.0
        grresults[:, 7]  = grresults[:, 7] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[0] / self.typenumber[2] /2.0
        grresults[:, 8]  = grresults[:, 8] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[0] / self.typenumber[3] /2.0
        grresults[:, 9]  = grresults[:, 9] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[0] / self.typenumber[4] /2.0
        grresults[:,10]  = grresults[:,10] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[1] / self.typenumber[2] /2.0
        grresults[:,11]  = grresults[:,11] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[1] / self.typenumber[3] /2.0
        grresults[:,12]  = grresults[:,12] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[1] / self.typenumber[4] /2.0
        grresults[:,13]  = grresults[:,13] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[2] / self.typenumber[3] /2.0
        grresults[:,14]  = grresults[:,14] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[2] / self.typenumber[4] /2.0
        grresults[:,15]  = grresults[:,15] * 2 / self.nsnapshots/ nideal * self.boxvolume /self.typenumber[3] / self.typenumber[4] /2.0

        binright = binright - 0.5 * rdelta #middle of each bin
        results  = np.column_stack((binright, grresults))
        names    = 'r  g(r)  g11(r)  g22(r)  g33(r)  g44(r)  g55(r)  g12(r)  g13(r)  g14(r)  g15(r)  g23(r)  g24(r)  g25(r)  g34(r)  g35(r)  g45(r)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        
        logger.info('Finish Calculating PCF of a Quinary System')

    def senary(self, ppp: list, rdelta: float=0.01, outputfile :str=None) -> None:
        """
        Calculating PCF for senary system
        
        Inputs:
            1. ppp (list): the periodic boundary conditions,
                           setting 1 for yes and 0 for no, default [1,1,1],
                           that is, PBC is applied in all three dimensions for 3D box.
                           set [1, 1] for two-dimensional systems
            2. rdelta (float): bin size calculating g(r), the default value is 0.01
            3. outputfile (str): the file name to save the calculated g(r)

        Return:
            None [output calculated g(r) to a document]
        """
        logger.info('Start Calculating PCF of a Senary System')
        logger.info('Only calculate the overall g(r) at this stage')
        logger.info(f'Particle Type: {self.type}')
        logger.info(f'Particle typenumber: {self.typenumber}')

        maxbin = int(self.boxlength.min() / 2.0 / rdelta)
        grresults = np.zeros(maxbin)
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))
                countvalue, binedge = np.histogram(distance, bins=maxbin, range=(0, maxbin * rdelta))
                grresults += countvalue
        binleft = binedge[:-1]
        binright = binedge[1:]
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults = grresults * 2 / self.nparticle / self.nsnapshots / (nideal * self.rhototal)
        
        # middle of each bin
        binright = binright - 0.5 * rdelta 
        results  = np.column_stack((binright, grresults))
        names = 'r  g(r)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        
        logger.info('Finish Calculating PCF of a Senary System')
        logger.info('Only the overall g(r) is calculated')
