# coding = utf-8

"""see documentation @ ../docs/dynamics.md"""

import numpy as np
import pandas as pd
from reader.reader_utils import Snapshots
from utils.pbc import remove_pbc
from utils.funcs import alpha2factor
from utils.logging import get_logger_handle

logger = get_logger_handle(__name__)


class dynamics:
    """
    This module calculates particle-level dynamics.
    Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF.
    Overlap function Qt and its corresponding dynamic susceptibility QtX4.
    Mean-square displacements msd; non-Gaussion parameter alpha2
    four-point dynamic structure factor of fast and slow particles, respectively

    The module also computes corresponding particle type related dynamics by using the function partial()
    The module accounts for systems ranging from unary to senary

    Mean-squared displacements and non-Gaussian parameter should be calculated with the absolute coordinates
    e.g. via LAMMPS (xu, yu, zu)

    Two sets of configurations are suggested to use together, one with (x, y, z) and one with (xu, yu, zu);
    the former is used to calculate structure factor and the latter for dynamics
    """

    def __init__(
        self,
        xu_snapshots: Snapshots=None,
        x_snapshots: Snapshots=None,
        dt: float=0.002,
        ppp: np.ndarray=np.array([0,0,0])
    ) -> None:
        """
        Initializing dynamics class

        Inputs:
            1. xu_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [xu, yu, zu], true coordinates, default None
            2. x_snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                            with dump format [x, y, z], coordinates with periodic boundary conditions, default None
            3. dt (float): timestep used in user simulations, default 0.002
            4. ppp (np.ndarray): the periodic boundary conditions,
                                 setting 1 for yes and 0 for no, default np.array([0,0,0]),
                                 set np.array([0,0]) for two-dimensional systems
        
        Return:
            None
        """
        logger.info('True coordinates by xu are suggested for dynamics calculation')

        self.ndim = len(ppp)
        self.dt = dt

        if x_snapshots and xu_snapshots:
            logger.info('Use both x and xu coordinates to calculate dynamics')

            self.ppp = np.array([0,0,0])[:self.ndim]
            self.PBC = False
            self.hmatrix = None
            self.positions = [snapshot.positions for snapshot in xu_snapshots.snapshots] # calculate dynamics
            self.original = [snapshot.positions for snapshot in x_snapshots.snapshots]   # calculate S4
            if xu_snapshots.nsnapshots != x_snapshots.nsnapshots:
                raise ValueError('----incompatible x and xu format coordinates----')
            self.snapshots = x_snapshots
            self.xu_snapshots = xu_snapshots
        elif xu_snapshots and not x_snapshots:
            logger.info('Use xu coordinates to calculate dynamics')
            self.ppp = np.array([0,0,0])[:self.ndim]
            self.PBC = False
            self.hmatrix = None
            self.positions = [snapshot.positions for snapshot in xu_snapshots.snapshots]
            self.original = self.positions.copy()
            self.snapshots = xu_snapshots
        elif x_snapshots and not xu_snapshots:
            logger.info('Use x coordinates to calculate dynamics')
            self.ppp = ppp[:self.ndim]
            self.PBC = True
            self.hmatrix = [snapshot.hmatrix for snapshot in x_snapshots.snapshots]
            self.positions = [snapshot.positions for snapshot in x_snapshots.snapshots]
            self.original = self.positions.copy()
            self.snapshots = x_snapshots

        self.timestep = set(np.diff([snapshot.timestep for snapshot in self.snapshots.snapshots])).pop()
        self.nparticle = set([snapshot.nparticle for snapshot in self.snapshots.snapshots]).pop()
        self.particle_type = self.snapshots.snapshots[0].particle_type
        self.nsnapshots = self.snapshots.nsnapshots
        self.boxlength = self.snapshots.snapshots[0].boxlength
        self.typenumber, self.typecount = np.unique(self.particle_type, return_counts=True)
        
    def total(
        self,
        qmax: float,
        a: float=1.0,
        outputfile: str=None
    ) -> np.ndarray:
        """
        Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2
        
        Inputs:
            1. qmax (float): wavenumber corresponding to the first peak of structure factor
            2. a (float): cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            3. outputfile (str): file name to save the calculated dynamic results
        
        Return:
            Calculated dynamics results in np.ndarray
        """
        logger.info(f'Start calculating dynamics in {self.ndim} dimensionality system')

        results = np.zeros(((self.nsnapshots-1), 7))
        names = 't  ISF  ISFX4  Qt  QtX4  msd  alpha2'
        
        cal_isf = pd.DataFrame(np.zeros((self.nsnapshots-1))[np.newaxis, :])
        cal_Qt = pd.DataFrame(np.zeros((self.nsnapshots-1))[np.newaxis, :])
        cal_msd = pd.DataFrame(np.zeros((self.nsnapshots-1))[np.newaxis, :])
        cal_alp = pd.DataFrame(np.zeros((self.nsnapshots-1))[np.newaxis, :])
        deltat = np.zeros(((self.nsnapshots-1), 2), dtype=np.int32) #deltat, deltatcounts
        for n in range(self.nsnapshots-1):
            RII = self.positions[n+1:] - self.positions[n]
            if self.PBC:
                for ii in range(len(RII)):
                    RII[ii] = remove_pbc(RII[ii], self.hmatrix[n], self.ppp)

            RII_isf = (np.cos(RII*qmax).mean(axis=2)).sum(axis=1) #index is timeinterval -1
            cal_isf = pd.concat([cal_isf, pd.DataFrame(RII_isf[np.newaxis, :])])
            distance = np.square(RII).sum(axis=2)
            RII_Qt = (distance<=a**2).sum(axis=1)
            cal_Qt = pd.concat([cal_Qt, pd.DataFrame(RII_Qt[np.newaxis, :])])
            cal_msd = pd.concat([cal_msd, pd.DataFrame(distance.sum(axis = 1)[np.newaxis, :])])
            distance2 = np.square(distance).sum(axis=1)
            cal_alp = pd.concat([cal_alp, pd.DataFrame(distance2[np.newaxis, :])])
       
        cal_isf = cal_isf.iloc[1:]
        cal_Qt = cal_Qt.iloc[1:]
        cal_msd = cal_msd.iloc[1:]
        cal_alp = cal_alp.iloc[1:]
        deltat[:, 0] = np.array(cal_isf.columns) + 1 # Timeinterval
        deltat[:, 1] = np.array(cal_isf.count())     # Timeinterval frequency

        results[:, 0] = deltat[:, 0] * self.timestep * self.dt 
        results[:, 1] = cal_isf.mean() / self.nparticle
        results[:, 2] = ((cal_isf**2).mean() - (cal_isf.mean())**2) / self.nparticle
        results[:, 3] = cal_Qt.mean() / self.nparticle
        results[:, 4] = ((cal_Qt**2).mean() - (cal_Qt.mean())**2) / self.nparticle
        results[:, 5] = cal_msd.mean() / self.nparticle
        results[:, 6] = cal_alp.mean() / self.nparticle
        results[:, 6] = alpha2factor(self.ndim) * results[:, 6] / np.square(results[:, 5]) - 1.0

        if outputfile:
            np.savetxt(outputfile, results, fmt='%.8f', header = names, comments = '')
        logger.info(f'Finish calculating dynamics in {self.ndim} dimensionality system')
        return results

    def partial(self, qmax=[], a=[1.0, 1.0], outputfile=''):
        """ Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF
            Overlap function Qt and its corresponding dynamic susceptibility QtX4
            Mean-square displacements msd; non-Gaussion parameter alpha2
        
            qmax is the wavenumber corresponding to the first peak of structure factor
            qmax accounts for six components so it is a list, input as a list for each particle type
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>), input as a list for each type
        """
        print ('----Compute Partial Dynamics----')

        partialresults = [] #a list containing results of all particle types           
        for i in self.Type:  #loop over different particle types
            #TYPESET = np.where(np.array(self.ParticleType) == i, 1, 0).astype(np.int)
            TYPESET = [j == i for j in self.ParticleType]

            results = np.zeros(((self.SnapshotNumber - 1), 7))
            names  = 't  ISF  ISFX4  Qt  QtX4  msd  alpha2'
            
            cal_isf  = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
            cal_Qt   = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
            cal_msd  = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
            cal_alp  = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
            deltat   = np.zeros(((self.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
            for n in range(self.SnapshotNumber - 1):  #loop over time intervals
                #RII    = self.Positions[n + 1:] - self.Positions[n]
                RII  = [ii[TYPESET[n]] - self.Positions[n][TYPESET[n]] for ii in self.Positions[n+1:]]
                if self.PBC:
                    hmatrixinv = np.linalg.inv(self.hmatrix[n])
                    for ii in range(len(RII)):
                        matrixij = np.dot(RII[ii], hmatrixinv)
                        RII[ii]  = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC

                RII       = np.array(RII)
                RII_isf   = ((np.cos(RII * qmax[i - 1]).mean(axis = 2))).sum(axis = 1) #index is timeinterval -1
                cal_isf   = pd.concat([cal_isf, pd.DataFrame(RII_isf[np.newaxis, :])])
                distance  = np.square(RII).sum(axis = 2)
                RII_Qt    = (distance <= a**2).sum(axis = 1)
                cal_Qt    = pd.concat([cal_Qt, pd.DataFrame(RII_Qt[np.newaxis, :])])
                cal_msd   = pd.concat([cal_msd, pd.DataFrame(distance.sum(axis = 1)[np.newaxis, :])])
                distance2 = (np.square(distance)).sum(axis = 1)
                cal_alp   = pd.concat([cal_alp, pd.DataFrame(distance2[np.newaxis, :])])
        
            cal_isf      = cal_isf.iloc[1:]
            cal_Qt       = cal_Qt.iloc[1:]
            cal_msd      = cal_msd.iloc[1:]
            cal_alp      = cal_alp.iloc[1:]
            deltat[:, 0] = np.array(cal_isf.columns) + 1 #Timeinterval
            deltat[:, 1] = np.array(cal_isf.count())     #Timeinterval frequency

            results[:, 0] = deltat[:, 0] * self.TimeStep * self.dt 
            results[:, 1] = cal_isf.mean() / self.TypeNumber[i - 1]
            results[:, 2] = ((cal_isf**2).mean() - (cal_isf.mean())**2) / self.TypeNumber[i - 1]
            results[:, 3] = cal_Qt.mean() / self.TypeNumber[i - 1]
            results[:, 4] = ((cal_Qt**2).mean() - (cal_Qt.mean())**2) / self.TypeNumber[i - 1]
            results[:, 5] = cal_msd.mean() / self.TypeNumber[i - 1]
            results[:, 6] = cal_alp.mean() / self.TypeNumber[i - 1]
            results[:, 6] = alpha2factor(self.ndim) * results[:, 6] / np.square(results[:, 5]) - 1.0

            if outputfile:
                np.savetxt('Type' + str(i) + '.' + outputfile, results, fmt='%.8f', header = names, comments = '')
            
            partialresults.append(results)

        print ('----Compute Partial Dynamics Over----')
        return partialresults, names

    def slowS4(self, X4time, a=1.0, qrange=10, onlypositive=False, outputfile=''):
        """ Compute four-point dynamic structure factor of slow atoms at peak timescale of dynamic susceptibility

            Based on overlap function Qt and its corresponding dynamic susceptibility QtX4     
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            X4time is the peaktime scale of X4
            Dynamics should be calculated before computing S4
            xu coordinates should be used to identify slow particles
            x  cooridnates should be used to calcualte FFT
        """
        print ('----Compute dynamic S4(q) of slow particles----')

        X4time = int(X4time / self.dt / self.TimeStep)
        
        twopidl = 2 * pi / self.Boxlength
        Numofq = int(qrange*2.0 / twopidl.min())
        qvector = choosewavevector(self.ndim, Numofq, onlypositive)
        qvector = qvector.astype(np.float64) * twopidl[np.newaxis, :] #considering non-cubic box
        qvalues = np.linalg.norm(qvector, axis=1)
        
        sqresults = np.zeros((qvector.shape[0], 2))
        sqresults[:, 0] = qvalues
        for n in range(self.SnapshotNumber - X4time):
            RII = self.Positions[n + X4time] - self.Positions[n]
            if self.PBC:
                #hmatrixinv = np.linalg.inv(self.hmatrix[n])
                #matrixij   = np.dot(RII, hmatrixinv)
                #RII        = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC
                RII = remove_PBC(RII, self.hmatrix[n], self.ppp)

            RII = np.linalg.norm(RII, axis = 1)
            RII = np.where(RII <= a, 1, 0)
            
            sqtotal = np.zeros_like(sqresults)
            for i in range(self.ParticleNumber):
                if RII[i]:
                    thetas = (self.original[n][i][np.newaxis, :] * qvector).sum(axis=1)
                    sqtotal[:, 0] += np.sin(thetas)
                    sqtotal[:, 1] += np.cos(thetas)
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis=1) / self.ParticleNumber
        sqresults[:, 1] /= (self.SnapshotNumber - X4time)
        
        sqresults = pd.DataFrame(sqresults).round(8)
        results = sqresults.groupby(sqresults[0]).mean().reset_index().values     
        names = 'q  S4'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.8f', header = names, comments = '')
        print ('----Compute S4(q) of slow particles over----')
        return results, names

    def fastS4(self, a=1.0, X4timeset=0, qrange=10, onlypositive=False, outputfile=''):
        """ Compute four-point dynamic structure factor for fast particles at peak timescale of dynamic susceptibility

            Based on overlap function Qt and its corresponding dynamic susceptibility QtX4     
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            X4timeset is the peaktime scale of X4, if 0 will use the calculated one
            Dynamics should be calculated before computing S4
            xu coordinates should be used to identify fast atoms by calculating Qt and X4
            x  coordinates should be used to calculate FFT
        """
        print ('----Compute dynamic S4(q) of fast particles----')

        #-----------calculte overall dynamics first----------------
        results = np.zeros(((self.SnapshotNumber - 1), 3))
        names  = 't  Qt  QtX4'
        
        cal_Qt   = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        deltat   = np.zeros(((self.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
        for n in range(self.SnapshotNumber - 1):  #time interval
            RII = self.Positions[n + 1:] - self.Positions[n]
            if self.PBC:
                hmatrixinv = np.linalg.inv(self.hmatrix[n])
                for ii in range(len(RII)):
                    matrixij = np.dot(RII[ii], hmatrixinv)
                    RII[ii]  = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC

            distance  = np.square(RII).sum(axis = 2)
            RII_Qt    = (distance >= a**2).sum(axis = 1)
            cal_Qt    = pd.concat([cal_Qt, pd.DataFrame(RII_Qt[np.newaxis, :])])
        
        cal_Qt       = cal_Qt.iloc[1:]
        deltat[:, 0] = np.array(cal_Qt.columns) + 1 #Timeinterval
        deltat[:, 1] = np.array(cal_Qt.count())     #Timeinterval frequency

        results[:, 0] = deltat[:, 0] * self.TimeStep * self.dt 
        results[:, 1] = cal_Qt.mean() / self.ParticleNumber
        results[:, 2] = ((cal_Qt**2).mean() - (cal_Qt.mean())**2) / self.ParticleNumber
        if outputfile:
            np.savetxt('Dynamics.' + outputfile, results, fmt='%.8f', header = names, comments = '')

        #-----------calculte S4(q) of fast particles----------------
        twopidl = 2 * pi / self.Boxlength
        Numofq = int(qrange*2.0 / twopidl.min())
        qvector = choosewavevector(self.ndim, Numofq, onlypositive)
        qvector = qvector.astype(np.float64) * twopidl[np.newaxis, :]
        qvalues = np.linalg.norm(qvector, axis=1)

        sqresults = np.zeros((qvector.shape[0], 2))
        sqresults[:, 0] = qvalues

        if X4timeset:
            X4time = int(X4timeset / self.dt / self.TimeStep)
        else:
            X4time = deltat[results[:, 2].argmax(), 0] 

        for n in range(self.SnapshotNumber - X4time):
            RII = self.Positions[n + X4time] - self.Positions[n]
            if self.PBC:
                #hmatrixinv = np.linalg.inv(self.hmatrix[n])
                #matrixij   = np.dot(RII, hmatrixinv)
                #RII        = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC
                RII = remove_PBC(RII, self.hmatrix[n], self.ppp)

            RII = np.linalg.norm(RII, axis = 1)
            RII = np.where(RII >= a, 1, 0)
            
            sqtotal = np.zeros_like(sqresults)
            for i in range(self.ParticleNumber):
                if RII[i]:
                    thetas = (self.original[n][i][np.newaxis, :] * qvector).sum(axis=1)
                    sqtotal[:, 0] += np.sin(thetas)
                    sqtotal[:, 1] += np.cos(thetas)
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis=1) / self.ParticleNumber
        sqresults[:, 1] /= (self.SnapshotNumber - X4time)
        
        sqresults = pd.DataFrame(sqresults).round(8)
        results = sqresults.groupby(sqresults[0]).mean().reset_index().values      
        names = 'q  S4'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.8f', header = names, comments = '')
        print ('----Compute S4(q) of fast particles over----')
        return results, names
