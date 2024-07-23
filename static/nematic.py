#coding = utf-8

"""see documentation @ ../docs/orderings.md"""

import numpy as np
from reader.reader_utils import Snapshots
from utils.funcs import kronecker
from utils.coarse_graining import spatial_average
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

def Qtensor(
        Snapshots: Snapshots,
        ndim: int=2,
        neighborfile: str="",
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
        Q-tensor or eigenvalue scalar nematic order parameter in numpy ndarray format
        shape as [num_of_snapshots, num_of_particles]
    """
    logger.info(f"Calcualte nematic order for {Snapshots.nsnapshots} configurations")
    # TODO add a function for three dimensioanl systems
    assert ndim==2, "please set the correction dimensionality"

    QIJ = []
    for snapshot in Snapshots.snapshots:
        medium = np.zeros((snapshot.nparticle, ndim, ndim))
        for i in range(snapshot.nparticle):
            mu = snapshot.positions[i]
            for x in range(ndim):
                for y in range(ndim):
                    medium[i, x, y] = (ndim*mu[x]*mu[y]-kronecker(x,y))/2
        QIJ.append(medium)
    QIJ = np.array(QIJ)

    if neighborfile:
        # coarse-graining over certain volume if neighbor list provided
        #TODO @Yibang please test this function before test the current function
        QIJ = spatial_average(
            input_property=QIJ,
            neighborfile=neighborfile,
            Nmax=Nmax,
        )
        np.save(outputfile+".QIJ_cg.npy", QIJ)
    else:
        np.save(outputfile+".QIJ_raw.npy", QIJ)

    logger.info("Compute the scalar nematic order parameter")
    if eigvals:
        eigenvalues = np.zeros((QIJ.shape[0], QIJ.shape[1]))
        for n in range(Snapshots.nsnapshots):
            for i in range(Snapshots.snapshots[n].nparticle):
                eigenvalues[n, i] = np.linalg.eig(QIJ[n,i])[0].max()*2.0
        np.save(outputfile+".eigval.npy", eigenvalues)
        return eigenvalues
    else:
        Qtrace = np.zeros((QIJ.shape[0], QIJ.shape[1]))
        for n in range(Snapshots.nsnapshots):
            for i in range(Snapshots.snapshots[n].nparticle):
                Qtrace[n, i] = np.trace(np.matmul(QIJ[n,i], QIJ[n,i]))
        Qtrace *= ndim / (ndim-1)
        Qtrace = np.sqrt(Qtrace)
        np.save(outputfile+".Qtrace.npy", Qtrace)
        return Qtrace 
    logger.info("Calculate the Nematic order tensor and scalar parameters done")
