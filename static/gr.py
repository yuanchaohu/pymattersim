# coding = utf-8

"""see documentation @ ../docs/static_properties.md"""

from typing import Optional, Callable
import numpy as np
import pandas as pd
from reader.reader_utils import SingleSnapshot, Snapshots
from utils.pbc import remove_pbc
from utils.funcs import nidealfac
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

def conditional_gr(
    snapshot: SingleSnapshot,
    condition: np.ndarray,
    ppp: list=[1,1,1],
    rdelta: float=0.01
    ) -> pd.DataFrame:
    """
    Calculate the pair correlation function of a single configuration for selected particles,
    it is also useful to calculate the Fourier-Transform of a physical quantity

    Input:
        1. snapshot (reader.reader_utils.SingleSnapshot): single snapshot object of input trajectory
        2. condition (np.ndarray): particle-level condition for g(r)
        3. ppp (list): the periodic boundary conditions,
                       setting 1 for yes and 0 for no, default [1,1,1],
                       set [1, 1] for two-dimensional systems
        4. rdelta (float): bin size calculating g(r), the default value is 0.01

    Return:
        calculated conditional g(r) (pd.DataFrame)
    """
    Natom = snapshot.nparticle
    ndim = snapshot.positions.shape[1]
    
    logger.info(f"Calculating conditional g(r) for {Natom}-atom system")
    maxbin = int(snapshot.boxlength.min() / 2.0 / rdelta)
    grresults = pd.DataFrame(0, index=range(maxbin), columns="r gr".split())

    if np.array(condition).dtype=="bool":
        condition = condition.astype(np.int32)
        Natom = condition.sum()

    for i in range(snapshot.nparticle):
        RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
        RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
        distance = np.linalg.norm(RIJ, axis=1)
        SIJ = condition[i+1:] * condition[i]
        countvalue, binedge = np.histogram(distance, bins=maxbin, range=(0, maxbin*rdelta), weights=SIJ)
        grresults["gr"] += countvalue
        
    binleft = binedge[:-1]
    binright = binedge[1:]
    Nideal = nidealfac(ndim) * np.pi * (binright**ndim-binleft**ndim)
    rhototal = Natom / np.prod(snapshot.boxlength)
    grresults["gr"] = grresults["gr"]*2 / Natom / (Nideal*rhototal)

    grresults["r"] = binright - 0.5*rdelta
    return grresults


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

        self.nidealfac = nidealfac(self.ndim)
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

        grresults = pd.DataFrame(0, index=range(self.maxbin), columns='r g(r)'.split())
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0,self.maxbin*self.rdelta))
                grresults["g(r)"] += countvalue
        binleft = binedge[:-1]
        binright = binedge[1:]
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults["g(r)"] = grresults["g(r)"] * 2 / self.nparticle / self.nsnapshots / (nideal*self.rhototal)

        grresults["r"] = binright - 0.5 * self.rdelta
        if self.outputfile:
            grresults.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating PCF of a Unary System')
        return grresults

    def binary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for binary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start Calculating g(r) of a Binary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')

        grresults = pd.DataFrame(0, index=range(self.maxbin), columns='r g(r) g11(r) g22(r) g12(r)'.split())
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:],
                            np.zeros_like(snapshot.particle_type[i+1:]) + snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g(r)"] += countvalue

                countsum = TIJ.sum(axis=1)
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g11(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==4], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g22(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g12(r)"] += countvalue

        binleft = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults["g(r)"] = grresults["g(r)"] * 2 / self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults["g11(r)"] = grresults["g11(r)"] * 2 / self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults["g22(r)"] = grresults["g22(r)"] * 2 / self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults["g12(r)"] = grresults["g12(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0

        grresults["r"] = binright - 0.5 * self.rdelta
        if self.outputfile:
            grresults.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating g(r) of a Binary System')
        return grresults

    def ternary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for ternary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start Calculating g(r) of a Ternary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')
        
        grresults = pd.DataFrame(0, index=range(self.maxbin), columns='r g(r) g11(r) g22(r) g33(r) g12(r) g13(r) g23(r)'.split())
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:], np.zeros_like(snapshot.particle_type[i+1:])+snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g(r)"] += countvalue

                countsum = TIJ.sum(axis=1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum== 2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g11(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==4) & (countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g22(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==6], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g33(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g12(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==4) & (countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g13(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==5], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g23(r)"] += countvalue

        binleft = binedge[:-1]    # real value of each bin edge, not index
        binright = binedge[1:]    # len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
        grresults["g(r)"] = grresults["g(r)"] * 2 / self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults["g11(r)"] = grresults["g11(r)"] * 2 / self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults["g22(r)"] = grresults["g22(r)"] * 2 / self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults["g33(r)"] = grresults["g33(r)"] * 2 / self.nsnapshots / self.typecount[2] / (nideal * self.rhotype[2])
        grresults["g12(r)"] = grresults["g12(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults["g13(r)"] = grresults["g13(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[2] / 2.0
        grresults["g23(r)"] = grresults["g23(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[2] / 2.0

        grresults["r"] = binright - 0.5 * self.rdelta
        if self.outputfile:
            grresults.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating g(r) of a Ternary System')
        return grresults

    def quarternary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for Quarternary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start Calculating g(r) of a quarternary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')

        grresults = pd.DataFrame(0, index=range(self.maxbin), columns='r g(r) g11(r) g22(r) g33(r) g44(r) g12(r) g13(r) g14(r) g23(r) g24(r) g34(r)'.split())
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:], np.zeros_like(snapshot.particle_type[i+1:])+snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g(r)"] += countvalue

                countsum = TIJ.sum(axis = 1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g11(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum== 4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g22(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum== 6)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g33(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum== 8], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g44(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g12(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum== 4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g13(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsub==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g14(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==5)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g23(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g24(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==7], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g34(r)"] += countvalue

        binleft  = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults["g(r)"] = grresults["g(r)"] * 2 / self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults["g11(r)"] = grresults["g11(r)"] * 2 / self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults["g22(r)"] = grresults["g22(r)"] * 2 / self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults["g33(r)"] = grresults["g33(r)"] * 2 / self.nsnapshots / self.typecount[2] / (nideal * self.rhotype[2])
        grresults["g44(r)"] = grresults["g44(r)"] * 2 / self.nsnapshots / self.typecount[3] / (nideal * self.rhotype[3])

        grresults["g12(r)"] = grresults["g12(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults["g13(r)"] = grresults["g13(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[2] / 2.0
        grresults["g14(r)"] = grresults["g14(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[3] / 2.0
        grresults["g23(r)"] = grresults["g23(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[2] / 2.0
        grresults["g24(r)"] = grresults["g24(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[3] / 2.0
        grresults["g34(r)"] = grresults["g34(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[2] / self.typecount[3] / 2.0

        grresults["r"] = binright - 0.5 * self.rdelta
        if self.outputfile:
            grresults.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish Calculating g(r) of a Quarternary System')
        return grresults

    def quinary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for quinary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start Calculating g(r) of a Quinary System')
        logger.info(f'System Composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')

        grresults = pd.DataFrame(0, index=range(self.maxbin), columns='r g(r) g11(r) g22(r) g33(r) g44(r) g55(r) g12(r) g13(r) g14(r) \
                                                                       g15(r) g23(r) g24(r) g25(r) g34(r) g35(r) g45(r)'.split())
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle - 1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                TIJ = np.c_[snapshot.particle_type[i+1:], np.zeros_like(snapshot.particle_type[i+1:]) + snapshot.particle_type[i]]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g(r)"] += countvalue

                countsum = TIJ.sum(axis = 1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g11(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g22(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g33(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==8)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g44(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==10], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g55(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g12(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g13(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==5)&(countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g14(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==4)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g15(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum== 5)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g23(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g24(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==7)&(countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g25(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==7)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g34(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==8)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g35(r)"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==9], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["g45(r)"] += countvalue

        binleft = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
        grresults["g(r)"] = grresults["g(r)"] * 2 / self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults["g11(r)"] = grresults["g11(r)"] * 2 / self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults["g22(r)"] = grresults["g22(r)"] * 2 / self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults["g33(r)"] = grresults["g33(r)"] * 2 / self.nsnapshots / self.typecount[2] / (nideal * self.rhotype[2])
        grresults["g44(r)"] = grresults["g44(r)"] * 2 / self.nsnapshots / self.typecount[3] / (nideal * self.rhotype[3])
        grresults["g55(r)"] = grresults["g55(r)"] * 2 / self.nsnapshots / self.typecount[4] / (nideal * self.rhotype[4])

        grresults["g12(r)"] = grresults["g12(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults["g13(r)"] = grresults["g13(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[2] / 2.0
        grresults["g14(r)"] = grresults["g14(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[3] / 2.0
        grresults["g15(r)"] = grresults["g15(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[4] / 2.0
        grresults["g23(r)"] = grresults["g23(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[2] / 2.0
        grresults["g24(r)"] = grresults["g24(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[3] / 2.0
        grresults["g25(r)"] = grresults["g25(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[4] / 2.0
        grresults["g34(r)"] = grresults["g34(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[2] / self.typecount[3] / 2.0
        grresults["g35(r)"] = grresults["g35(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[2] / self.typecount[4] / 2.0
        grresults["g45(r)"] = grresults["g45(r)"] * 2 / self.nsnapshots / nideal * self.boxvolume / self.typecount[3] / self.typecount[4] / 2.0

        grresults["r"] = binright - 0.5 * self.rdelta
        if self.outputfile:
            grresults.to_csv(self.outputfile, float_format="%.6f", index=False)
        
        logger.info('Finish Calculating g(r) of a Quinary System')
        return grresults
