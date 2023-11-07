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
    Also useful to calculate the spatial correlation of a physical quantity "A"

    Input:
        1. snapshot (reader.reader_utils.SingleSnapshot): single snapshot object of input trajectory
        2. condition (np.ndarray): particle-level condition for g(r)
        3. ppp (list): the periodic boundary conditions,
                       setting 1 for yes and 0 for no, default [1,1,1],
                       set [1, 1] for two-dimensional systems
        4. rdelta (float): bin size calculating g(r), default 0.01

    Return:
        calculated conditional g(r) (pd.DataFrame)
        (original g(r) is calculated for reference or post-processing)
    """
    Natom = snapshot.nparticle
    ndim = snapshot.positions.shape[1]
    maxbin = int(snapshot.boxlength.min() / 2.0 / rdelta)
    grresults = pd.DataFrame(0, index=range(maxbin), columns="r gr gA".split())

    if np.array(condition).dtype=="bool":
        condition = condition.astype(np.int32)
        Natom = condition.sum()
        logger.info(f"Calculate g(r) for {Natom} selected atoms")
    else:
        logger.info("Calculate spatial correlation of physical quantity 'A'")

    for i in range(snapshot.nparticle-1):
        RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
        RIJ = remove_pbc(RIJ, snapshot.hmatrix, ppp)
        distance = np.linalg.norm(RIJ, axis=1)
        # original g(r)
        countvalue, binedge = np.histogram(distance, bins=maxbin, range=(0, maxbin*rdelta))
        grresults["gr"] += countvalue
        # conditional g(r)
        SIJ = condition[i+1:] * condition[i]
        countvalue, binedge = np.histogram(distance, bins=maxbin, range=(0, maxbin*rdelta), weights=SIJ)
        grresults["gA"] += countvalue
        
    binleft = binedge[:-1]
    binright = binedge[1:]
    Nideal = nidealfac(ndim) * np.pi * (binright**ndim-binleft**ndim)
    rhototal = Natom / np.prod(snapshot.boxlength)

    grresults["r"] = binright - 0.5*rdelta
    grresults["gr"] = grresults["gr"]*2 / snapshot.nparticle / (Nideal*rhototal)
    grresults["gA"] = grresults["gA"]*2 / Natom / (Nideal*rhototal)

    return grresults


class gr:
    """
    This module is used to calculate pair correlation functions g(r)
    covering unary to quinary systems.
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
        logger.info(f'System composition: {":".join([str(i) for i in np.round(self.typecount / self.nparticle, 3)])}')
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
            logger.info(f"This is a {len(self.typenumber)} system, only overall g(r) calculated")
            return self.unary()

    def unary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for unary system

        Return:
            calculated g(r) (pd.DataFrame)
        """
        logger.info('Start calculating g(r) of a unary system')

        grresults = pd.DataFrame(0, index=range(self.maxbin), columns='r gr'.split())
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0,self.maxbin*self.rdelta))
                grresults["gr"] += countvalue
        binleft = binedge[:-1]
        binright = binedge[1:]
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults["r"] = binright - 0.5 * self.rdelta
        grresults["gr"] = grresults["gr"]*2 / self.nparticle / self.nsnapshots / (nideal*self.rhototal)

        if self.outputfile:
            grresults.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish calculating g(r) of a unary system')
        return grresults

    def binary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for binary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start calculating g(r) of a binary system')

        grresults = pd.DataFrame(
            0, 
            index=range(self.maxbin), 
            columns='r gr gr11 gr22 gr12'.split()
        )
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                TIJ = np.c_[
                    snapshot.particle_type[i+1:],
                    np.zeros_like(snapshot.particle_type[i+1:])+snapshot.particle_type[i]
                ]

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr"] += countvalue

                countsum = TIJ.sum(axis=1)
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr11"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==4], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr22"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr12"] += countvalue

        binleft = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults["r"] = binright - 0.5 * self.rdelta
        grresults["gr"] = grresults["gr"]*2/ self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults["gr11"] = grresults["gr11"]*2/ self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults["gr22"] = grresults["gr22"]*2/ self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults["gr12"] = grresults["gr12"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0

        if self.outputfile:
            grresults.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish calculating g(r) of a binary system')
        return grresults

    def ternary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for ternary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start calculating g(r) of a ternary system')
        
        grresults = pd.DataFrame(
            0, 
            index=range(self.maxbin), 
            columns='r gr gr11 gr22 gr33 gr12 gr13 gr23'.split()
        )
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                TIJ = np.c_[
                    snapshot.particle_type[i+1:], 
                    np.zeros_like(snapshot.particle_type[i+1:])+snapshot.particle_type[i]
                ]

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr"] += countvalue

                countsum = TIJ.sum(axis=1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum== 2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr11"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr22"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==6], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr33"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr12"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr13"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==5], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr23"] += countvalue

        binleft = binedge[:-1]    # real value of each bin edge, not index
        binright = binedge[1:]    # len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
        grresults["r"] = binright - 0.5 * self.rdelta
        grresults["gr"] = grresults["gr"]*2/ self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults["gr11"] = grresults["gr11"]*2/ self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults["gr22"] = grresults["gr22"]*2/ self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults["gr33"] = grresults["gr33"]*2/ self.nsnapshots / self.typecount[2] / (nideal * self.rhotype[2])
        grresults["gr12"] = grresults["gr12"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults["gr13"] = grresults["gr13"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[2] / 2.0
        grresults["gr23"] = grresults["gr23"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[2] / 2.0

        if self.outputfile:
            grresults.to_csv(self.outputfile, float_format="%.6f", index=False)

        logger.info('Finish calculating g(r) of a ternary system')
        return grresults

    def quarternary(self) -> pd.DataFrame:
        """
        Calculating pair correlation function g(r) for Quarternary system

        Return:
            calculated g(r) (pd.DataFrame)
        """

        logger.info('Start calculating g(r) of a quarternary system')

        grresults = pd.DataFrame(
            0, 
            index=range(self.maxbin), 
            columns='r gr gr11 gr22 gr33 gr44 gr12 gr13 gr14 gr23 gr24 gr34'.split()
        )
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle-1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                TIJ = np.c_[
                    snapshot.particle_type[i+1:], 
                    np.zeros_like(snapshot.particle_type[i+1:])+snapshot.particle_type[i]
                ]

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr"] += countvalue

                countsum = TIJ.sum(axis = 1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr11"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum== 4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr22"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum== 6)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr33"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum== 8], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr44"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr12"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum== 4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr13"] += countvalue
                countvalue, binedge = np.histogram(distance[countsub==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr14"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==5)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr23"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr24"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==7], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr34"] += countvalue

        binleft  = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim - binleft**self.ndim)
        grresults["r"] = binright - 0.5 * self.rdelta
        grresults["gr"] = grresults["gr"]*2/ self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults["gr11"] = grresults["gr11"]*2/ self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults["gr22"] = grresults["gr22"]*2/ self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults["gr33"] = grresults["gr33"]*2/ self.nsnapshots / self.typecount[2] / (nideal * self.rhotype[2])
        grresults["gr44"] = grresults["gr44"]*2/ self.nsnapshots / self.typecount[3] / (nideal * self.rhotype[3])

        grresults["gr12"] = grresults["gr12"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults["gr13"] = grresults["gr13"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[2] / 2.0
        grresults["gr14"] = grresults["gr14"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[3] / 2.0
        grresults["gr23"] = grresults["gr23"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[2] / 2.0
        grresults["gr24"] = grresults["gr24"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[3] / 2.0
        grresults["gr34"] = grresults["gr34"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[2] / self.typecount[3] / 2.0

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

        logger.info('Start calculating g(r) of a quinary system')

        grresults = pd.DataFrame(
            0, 
            index=range(self.maxbin), 
            columns='r gr gr11 gr22 gr33 gr44 gr55 gr12 gr13 gr14 gr15 gr23 gr24 gr25 gr34 gr35 gr45'.split()
        )
        for snapshot in self.snapshots.snapshots:
            for i in range(self.nparticle - 1):
                RIJ = snapshot.positions[i+1:] - snapshot.positions[i]
                RIJ = remove_pbc(RIJ, snapshot.hmatrix, self.ppp)
                distance = np.linalg.norm(RIJ, axis=1)
                TIJ = np.c_[
                    snapshot.particle_type[i+1:],
                    np.zeros_like(snapshot.particle_type[i+1:])+snapshot.particle_type[i]
                ]

                countvalue, binedge = np.histogram(distance, bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr"] += countvalue

                countsum = TIJ.sum(axis = 1)
                countsub = np.abs(TIJ[:, 0]-TIJ[:, 1])
                countvalue, binedge = np.histogram(distance[countsum==2], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr11"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr22"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr33"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==8)&(countsub==0)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr44"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==10], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr55"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==3], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr12"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==4)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr13"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==5)&(countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr14"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==4)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr15"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum== 5)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr23"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==6)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr24"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==7)&(countsub==3)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr25"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==7)&(countsub==1)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr34"] += countvalue
                countvalue, binedge = np.histogram(distance[(countsum==8)&(countsub==2)], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr35"] += countvalue
                countvalue, binedge = np.histogram(distance[countsum==9], bins=self.maxbin, range=(0, self.maxbin*self.rdelta))
                grresults["gr45"] += countvalue

        binleft = binedge[:-1]   #real value of each bin edge, not index
        binright = binedge[1:]   #len(countvalue) = len(binedge) - 1
        nideal = self.nidealfac * np.pi * (binright**self.ndim-binleft**self.ndim)
        grresults["r"] = binright - 0.5 * self.rdelta
        grresults["gr"] = grresults["gr"]*2/ self.nsnapshots / self.nparticle / (nideal*self.rhototal)
        grresults["gr11"] = grresults["gr11"]*2/ self.nsnapshots / self.typecount[0] / (nideal * self.rhotype[0])
        grresults["gr22"] = grresults["gr22"]*2/ self.nsnapshots / self.typecount[1] / (nideal * self.rhotype[1])
        grresults["gr33"] = grresults["gr33"]*2/ self.nsnapshots / self.typecount[2] / (nideal * self.rhotype[2])
        grresults["gr44"] = grresults["gr44"]*2/ self.nsnapshots / self.typecount[3] / (nideal * self.rhotype[3])
        grresults["gr55"] = grresults["gr55"]*2/ self.nsnapshots / self.typecount[4] / (nideal * self.rhotype[4])

        grresults["gr12"] = grresults["gr12"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[1] / 2.0
        grresults["gr13"] = grresults["gr13"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[2] / 2.0
        grresults["gr14"] = grresults["gr14"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[3] / 2.0
        grresults["gr15"] = grresults["gr15"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[0] / self.typecount[4] / 2.0
        grresults["gr23"] = grresults["gr23"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[2] / 2.0
        grresults["gr24"] = grresults["gr24"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[3] / 2.0
        grresults["gr25"] = grresults["gr25"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[1] / self.typecount[4] / 2.0
        grresults["gr34"] = grresults["gr34"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[2] / self.typecount[3] / 2.0
        grresults["gr35"] = grresults["gr35"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[2] / self.typecount[4] / 2.0
        grresults["gr45"] = grresults["gr45"]*2/ self.nsnapshots / nideal * self.boxvolume / self.typecount[3] / self.typecount[4] / 2.0

        if self.outputfile:
            grresults.to_csv(self.outputfile, float_format="%.6f", index=False)
        
        logger.info('Finish calculating g(r) of a quinary system')
        return grresults
