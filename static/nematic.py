#coding = utf-8

"""see documentation @ ../docs/orderings.md"""

import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from neighbors.read_neighbors import read_neighbors
from utils.logging import get_logger_handle
from util_functions import Kronecker

logger = get_logger_handle(__name__)

def Qtensor(
        Snapshots: Snapshots,
        ndim: int=2,
        neighborfile: str=None,
        Nmax: int=30,
        eigvals: bool=False,
        outputfile: str="",
    ) -> np.ndarray:
    """
    Calculate the nematic order parameter of a model system

    Inputs:
        1. Snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
                     DumpFileType=LAMMPSVECTOR
        2. ndim (int): dimensionality of the input configurations
        3. neighborfile (str): file name of particle neighbors (see module neighbors)
        4. Nmax (int): maximum number for neighbors, default 30
        5. eigvals (bool): whether calculate eigenvalue of the Qtensor or not, default False
        6. outputfile (str): file name of the calculation output
    
    Return:
        Q-tensor in numpy ndarray format
    """
    logger.info(f"Calcualte nematic order for {Snapshots.nsnapshots} configurations")

    QIJ = []
    for snapshot in Snapshots.snapshots:
        medium = np.zeros((snapshot.nparticle, ndim, ndim))
        for i in range(snapshot.nparticle):
            mu = snapshot.positions[i]
            for x in range(ndim):
                for y in range(ndim):
                    medium[i, x, y] = (ndim*mu[x]*mu[y]-Kronecker(x,y))/2
        QIJ.append(medium)
    QIJ = np.array(QIJ)
    np.save(outputfile+".QIJ_raw.npy", QIJ)

    if eigvals:
        eigenvalues = np.zeros((QIJ.shape[0], QIJ.shape[1]))
    Qtrace = np.zeros((QIJ.shape[0], QIJ.shape[1]))
    if neighborfile:
        # coarse-graining over certain volume if neighbor list provided
        logger.info("Compute the coarse-grained nematic order parameter")
        QIJ0 = np.copy(QIJ)
        f = open(neighborfile, "-r", encoding="utf-8")
        for n in range(Snapshots.nsnapshots):
            cnlist = read_neighbors(f, Snapshots.snapshots[n].nparticle, Nmax=Nmax)
            for i in range(Snapshots.snapshots[n].nparticle):
                for j in range(cnlist[i, 0]):
                    QIJ[n, i] += QIJ0[n, cnlist[i, j+1]]
                QIJ[n, i] /= (1+cnlist[i,0])
                Qtrace[n, i] = np.trace(np.matmul(QIJ[n,i], QIJ[n,i]))
                if eigvals:
                    eigenvalues[n, i] = np.linalg.eig(QIJ[n,i])[0].max()*2.0
        del QIJ0
        f.close()
        np.save(outputfile+".QIJ_CG.npy", QIJ)
    else:
        # local QIJ
        logger.info("Compute the local/original nematic order parameter")
        for n in range(Snapshots.nsnapshots):
            for i in range(Snapshots.snapshots[n].nparticle):
                Qtrace[n, i] = np.trace(np.matmul(QIJ[n,i], QIJ[n,i]))
                if eigvals:
                    eigenvalues[n, i] = np.linalg.eig(QIJ[n,i])[0].max()*2.0

    Qtrace *= ndim / (ndim-1)
    Qtrace = np.sqrt(Qtrace)
    np.save(outputfile, Qtrace)
    if eigvals:
        np.save(outputfile+".eigval.npy", eigenvalues)
    logger.info("Calculate the Nematic order tensor and scalar parameters done")