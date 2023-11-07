# coding = utf-8

"""see documentation @ ../docs/static.md"""

from typing import Optional, Callable
import numpy as np
import pandas as pd
from reader.reader_utils import SingleSnapshot, Snapshots
from utils.pbc import remove_pbc
from utils.Nideal import Nidealfac
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

def selection_gr(
    snapshot: SingleSnapshot,
    selection: np.ndarray,
    ppp: list=[1,1,1],
    rdelta: float=0.01
    ) -> pd.DataFrame:
    """
    Calculate the pair correlation function of a single configuration for selected particles,
    it is also useful to calculate the Fourier-Transform of a physical quantity

    Input:
        1. snapshot (reader.reader_utils.SingleSnapshot): single snapshot object of input trajectory
        2. selection (np.ndarray): particle-level condition for g(r)
        3. ppp (list): the periodic boundary conditions,
                       setting 1 for yes and 0 for no, default [1,1,1],
                       set [1, 1] for two-dimensional systems
        4. rdelta (float): bin size calculating g(r), the default value is 0.01

    Return:
        calculated conditional g(r) (pd.DataFrame)
    """
    Natom = snapshot.nparticle
    positions = snapshot.positions
    ndim = positions.shape[1]
    boxlength = snapshot.boxlength

    logger.info(f"Calculating conditional g(r) for {Natom}-atom system")

    maxbin = int(boxlength.min() / 2.0 / rdelta)
    grresults = np.zeros(maxbin)

    if np.array(selection).dtype=="bool":
        selection = selection.astype(np.int32)
        Natom = selection.sum()

    for i in range(positions.shape[0]):
        RIJ = positions[i+1:] - positions[i]
        RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
        distance = np.linalg.norm(RIJ, axis = 1)
        SIJ = selection[i+1:] * selection[i]
        countvalue, binedge = np.histogram(distance, bins=maxbin, range=(0, maxbin*rdelta), weights=SIJ)
        grresults += countvalue
        
    binleft = binedge[:-1]
    binright = binedge[1:]
    Nideal = Nidealfac(ndim) * np.pi * (binright**ndim-binleft**ndim)
    rhototal = Natom / np.prod(boxlength)
    grresults = grresults * 2 / Natom / (Nideal * rhototal)

    binright = binright - 0.5 * rdelta
    results  = np.column_stack((binright, grresults))
    return results


class gr:
    """
    This module is used to calculate pair correlation functions g(r)
    covering unary to senary systems.
    """
    def __init__(
            self,
            snapshots: Snapshots,
            ppp: list=[1,1,1],
            rdelta: float=0.01,
            outputfile: str=None
            ) -> None:
        """
        Initializing g(r) class

        Inputs:
            1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory 
                         (returned by reader.dump_reader.DumpReader)
            2. ppp (list): the periodic boundary conditions,
                           setting 1 for yes and 0 for no, default [1,1,1],
                           set [1, 1] for two-dimensional systems
            3. rdelta (float): bin size calculating g(r), the default value is 0.01
            4. outputfile (str): the name of csv file to save the calculated g(r)

        Return:
            None
        """
        self.snapshots = snapshots
        self.ppp = ppp
        self.rdelta = rdelta
        self.outputfile = outputfile

        self.nsnapshots = self.snapshots.nsnapshots
        self.ndim = self.snapshots.snapshots[0].positions.shape[1]
        self.nparticle = snapshots.snapshots[0].nparticle
        assert len({snapshot.nparticle for snapshot in self.snapshots.snapshots}) == 1,\
            "Paticle Number Changes during simulation"
        assert len({tuple(snapshot.boxlength) for snapshot in self.snapshots.snapshots}) == 1,\
            "Simulation Box Length Changes during simulation"

        self.boxvolume = np.prod(self.snapshots.snapshots[0].boxlength)
        self.typenumber, self.typecount = np.unique(self.snapshots.snapshots[0].particle_type, return_counts=True)
        assert np.sum(self.typecount) == self.nparticle,\
            "Sum of Indivdual Types is Not the Total Amount"

        self.nidealfac = Nidealfac(self.ndim)
        self.rhototal = self.nparticle / self.boxvolume
        self.rhotype = self.typecount / self.boxvolume
        self.maxbin = int(self.snapshots.snapshots[0].boxlength.min()/2.0/self.rdelta)

    def getresults(self) -> Optional[Callable]:
        """
        Calculating g(r) for system with different particle type numbers

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
        if len(self.typenumber) > 6:
            logger.info(f"This is a {len(self.typenumber)} system, only overall S(q) calculated")
            return self.unary()

    def unary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for unary system

        Return:
            calculated g(r) (pd.DataFrame)
        """
        logger.info('Start Calculating g(r) of a Unary System')

        grresults = np.zeros(self.maxbin)
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0,self.maxbin*self.rdelta))
                grresults += countvalue
        binleft = binedge[:-1]
        binright = binedge[1:]
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults = grresults * 2 / self.nparticle / self.nsnapshots / (nideal*self.rhototal)

        binright = binright - 0.5 * self.rdelta
        names = 'r  g(r)'
        results = pd.DataFrame(np.column_stack((binright, grresults)), columns=names.split())
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating PCF of a Unary System')
        return results

    def binary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for binary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start Calculating g(r) of a Binary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')

        grresults = np.zeros((self.maxbin, 4))
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:],
                            np.zeros_like(snapshot.particle_type[i+1:]) + snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 0] += countvalue

                countsum = TIJ.sum(axis=1)
                countvalue, binedge = np.histogram(distance[countsum==2],
                                                   bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 1] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==3],
                                                   bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 2] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==4],
                                                   bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 3] += countvalue

        binleft = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults[:, 0] = grresults[:, 0] * 2 / self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults[:, 1] = grresults[:, 1] * 2 / self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults[:, 2] = grresults[:, 2] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults[:, 3] = grresults[:, 3] * 2 / self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])

        binright = binright - 0.5 * self.rdelta
        names = 'r  g(r)  g11(r)  g12(r)  g22(r)'
        results = pd.DataFrame(np.column_stack((binright, grresults)), columns=names.split())
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating g(r) of a Binary System')
        return results


    def ternary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for ternary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start Calculating g(r) of a Ternary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')
        
        grresults = np.zeros((self.maxbin, 7))
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:], np.zeros_like(snapshot.particle_type[i+1:])+snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 0] += countvalue

                countsum = TIJ.sum(axis=1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum== 2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 1] += countvalue # 11
                countvalue, binedge = np.histogram(distance[(countsum==4) & (countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 2] += countvalue # 22
                countvalue, binedge = np.histogram(distance[countsum==6], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 3] += countvalue # 33
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 4] += countvalue # 12
                countvalue, binedge = np.histogram(distance[(countsum==4) & (countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 5] += countvalue # 13
                countvalue, binedge = np.histogram(distance[countsum==5], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 6] += countvalue # 23

        binleft = binedge[:-1]    # real value of each bin edge, not index
        binright = binedge[1:]    # len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
        grresults[:, 0] = grresults[:, 0] * 2 / self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults[:, 1] = grresults[:, 1] * 2 / self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults[:, 2] = grresults[:, 2] * 2 / self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults[:, 3] = grresults[:, 3] * 2 / self.nsnapshots / self.typecount[2] / (nideal * self.rhotype[2])
        grresults[:, 4] = grresults[:, 4] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults[:, 5] = grresults[:, 5] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[2] / 2.0
        grresults[:, 6] = grresults[:, 6] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[2] / 2.0

        binright = binright - 0.5 * self.rdelta
        names = 'r  g(r)  g11(r)  g22(r)  g33(r)  g12(r)  g13(r)  g23(r)'
        results = pd.DataFrame(np.column_stack((binright, grresults)), columns=names.split())
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating g(r) of a Ternary System')
        return results

    def quarternary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for Quarternary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start Calculating g(r) of a quarternary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')

        grresults = np.zeros((self.maxbin, 11))
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:], np.zeros_like(snapshot.particle_type[i+1:])+snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 0] += countvalue

                countsum = TIJ.sum(axis = 1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 1] += countvalue #11
                countvalue, binedge = np.histogram(distance[(countsum== 4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 2] += countvalue #22
                countvalue, binedge = np.histogram(distance[(countsum== 6)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 3] += countvalue #33
                countvalue, binedge = np.histogram(distance[countsum== 8], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 4] += countvalue #44
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 5] += countvalue #12
                countvalue, binedge = np.histogram(distance[(countsum== 4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 6] += countvalue #13
                countvalue, binedge = np.histogram(distance[countsub==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 7] += countvalue #14
                countvalue, binedge = np.histogram(distance[(countsum==5)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 8] += countvalue #23
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 9] += countvalue #24
                countvalue, binedge = np.histogram(distance[countsum==7], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:,10] += countvalue #34

        binleft  = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults[:, 0] = grresults[:, 0] * 2 / self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults[:, 1] = grresults[:, 1] * 2 / self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults[:, 2] = grresults[:, 2] * 2 / self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults[:, 3] = grresults[:, 3] * 2 / self.nsnapshots / self.typecount[2] / (nideal * self.rhotype[2])
        grresults[:, 4] = grresults[:, 4] * 2 / self.nsnapshots / self.typecount[3] / (nideal * self.rhotype[3])

        grresults[:, 5] = grresults[:, 5] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults[:, 6] = grresults[:, 6] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[2] / 2.0
        grresults[:, 7] = grresults[:, 7] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[3] / 2.0
        grresults[:, 8] = grresults[:, 8] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[2] / 2.0
        grresults[:, 9] = grresults[:, 9] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[3] / 2.0
        grresults[:,10] = grresults[:,10] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[2] / self.typecount[3] / 2.0

        binright = binright - 0.5 * self.rdelta
        names = 'r  g(r)  g11(r)  g22(r)  g33(r)  g44(r)  g12(r)  g13(r)  g14(r)  g23(r)  g24(r)  g34(r)'
        results = pd.DataFrame(np.column_stack((binright, grresults)), columns=names.split())
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating g(r) of a Quarternary System')
        return results

    def quinary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for quinary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start Calculating g(r) of a Quinary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')

        grresults = np.zeros((self.maxbin, 16))
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle - 1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:], np.zeros_like(snapshot.particle_type[i+1:]) + snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 0] += countvalue

                countsum = TIJ.sum(axis = 1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 1] += countvalue #11
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 2] += countvalue #22
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 3] += countvalue #33
                countvalue, binedge = np.histogram(distance[(countsum==8)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 4] += countvalue #44
                countvalue, binedge = np.histogram(distance[countsum==10], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 5] += countvalue #55
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 6] += countvalue #12
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 7] += countvalue #13
                countvalue, binedge = np.histogram(distance[(countsum==5)&(countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 8] += countvalue #14
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==4)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:, 9] += countvalue #15
                countvalue, binedge = np.histogram(distance[(countsum== 5)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:,10] += countvalue #23
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:,11] += countvalue #24
                countvalue, binedge = np.histogram(distance[(countsum==7)&(countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:,12] += countvalue #25
                countvalue, binedge = np.histogram(distance[(countsum==7)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:,13] += countvalue #34
                countvalue, binedge = np.histogram(distance[(countsum==8)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:,14] += countvalue #35
                countvalue, binedge = np.histogram(distance[countsum==9], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults[:,15] += countvalue #45

        binleft = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
        grresults[:, 0] = grresults[:, 0] * 2 / self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults[:, 1] = grresults[:, 1] * 2 / self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults[:, 2] = grresults[:, 2] * 2 / self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults[:, 3] = grresults[:, 3] * 2 / self.nsnapshots / self.typecount[2] / (nideal * self.rhotype[2])
        grresults[:, 4] = grresults[:, 4] * 2 / self.nsnapshots / self.typecount[3] / (nideal * self.rhotype[3])
        grresults[:, 5] = grresults[:, 5] * 2 / self.nsnapshots / self.typecount[4] / (nideal * self.rhotype[4])

        grresults[:, 6] = grresults[:, 6] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults[:, 7] = grresults[:, 7] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[2] / 2.0
        grresults[:, 8] = grresults[:, 8] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[3] / 2.0
        grresults[:, 9] = grresults[:, 9] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[4] / 2.0
        grresults[:,10] = grresults[:,10] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[2] / 2.0
        grresults[:,11] = grresults[:,11] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[3] / 2.0
        grresults[:,12] = grresults[:,12] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[4] / 2.0
        grresults[:,13] = grresults[:,13] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[2] / self.typecount[3] / 2.0
        grresults[:,14] = grresults[:,14] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[2] / self.typecount[4] / 2.0
        grresults[:,15] = grresults[:,15] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[3] / self.typecount[4] / 2.0

        binright = binright - 0.5 * self.rdelta
        names = 'r  g(r)  g11(r)  g22(r)  g33(r)  g44(r)  g55(r)  g12(r)  g13(r)  g14(r)  g15(r)  g23(r)  g24(r)  g25(r)  g34(r)  g35(r)  g45(r)'
        results = pd.DataFrame(np.column_stack((binright, grresults)), columns=names.split())
        if self.outputfile:
            results.to_csv(self.outputfile, float_format="%.6f", index=False)
        
        logger.info('Finish Calculating g(r) of a Quinary System')
        return results
