# coding = utf-8

"""This module provide helper functions to read gsd files"""

import os
from typing import Any

import numpy as np

from ..reader.reader_utils import SingleSnapshot, Snapshots
from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)


def read_gsd_wrapper(file_name: str, ndim: int) -> Snapshots:
    """
    Wrapper function around read gsd file

    Inputs:
        1. file_name (str): file name of gsd snapshots

        2. ndim (int): dimensionality

    Return:
        list of single snapshot information
    """
    logger.info("---------Start reading GSD file reading -----------")
    try:
        import gsd, gsd.hoomd
    except BaseException:
        raise ImportError("***Please install the GSD library***")

    f = gsd.hoomd.open(file_name, mode="r")
    snapshots = read_gsd(f, ndim)
    logger.info("---------GSD file reading completed -----------")
    return snapshots


def read_gsd_dcd_wrapper(file_name: str, ndim: int) -> Snapshots:
    """
    Wrapper function around read gsd and dcd file

    Inputs:
        1. file_name (str): file name of gsd snapshots

        2. ndim (int): dimensionality

    Return:
        list of single snapshot information
    """
    logger.info("---------Start reading GSD & DCD file -----------")
    gsd_filename = file_name
    gsd_filepath = os.path.dirname(gsd_filename)
    dcd_filename = gsd_filepath + "/" + os.path.basename(gsd_filename)[:-3] + "dcd"

    try:
        import gsd, gsd.hoomd
    except BaseException:
        raise ImportError("***Please install the GSD library***")

    try:
        from mdtraj.formats import DCDTrajectoryFile
    except BaseException:
        raise ImportError("***Please install the mdtraj library***")

    f_gsd = gsd.hoomd.open(gsd_filename, mode="r")
    f_dcd = DCDTrajectoryFile(dcd_filename, "r")
    snapshots = read_gsd_dcd(f_gsd, f_dcd, ndim)
    f_dcd.close()
    logger.info("---------GSD & DCD file reading completed-----------")
    return snapshots


def read_gsd(f: Any, ndim: int) -> Snapshots:
    """
    Read gsd file from HOOMD-blue
    gsd file provides all the configuration information
    ref: https://gsd.readthedocs.io/en/stable/hoomd-examples.html

    Inputs:
        1. f: open file type by python

        2. ndim (int): dimensionality

    Return:
        list of single snapshot information
    """
    if f[0].configuration.dimensions != ndim:
        logger.error("---*Warning*: Wrong dimension information given---")
        return None

    snapshots = []
    for onesnapshot in f:
        # ------------configuration information---------------
        boxlength = onesnapshot.configuration.box[:ndim]
        hmatrix = np.diag(boxlength)
        # ------------particles information-------------------
        positions = onesnapshot.particles.position[:, :ndim]
        boxbounds = np.column_stack((positions.min(axis=0), positions.max(axis=0)))

        snapshot = SingleSnapshot(
            timestep=onesnapshot.configuration.step,
            nparticle=onesnapshot.particles.N,
            particle_type=onesnapshot.particles.typeid + 1,
            positions=positions,
            boxlength=boxlength,
            boxbounds=boxbounds,
            realbounds=None,
            hmatrix=hmatrix,
        )
        snapshots.append(snapshot)
    return Snapshots(nsnapshots=len(f), snapshots=snapshots)


def read_gsd_dcd(f_gsd: Any, f_dcd: Any, ndim: int) -> Snapshots:
    """
    Read gsd and dcd file from HOOMD-blue
        1. gsd file provides all the configuration information except positions
            with periodic boundary conditions;
        2. dcd file provides the unwrap positions without periodic boundary conditions.
    gsd is to get static information about the trajectory
    dcd is to get the absolute displacement to calculate dynamics
    ref: https://gsd.readthedocs.io/en/stable/hoomd-examples.html
    ref: http://mdtraj.org/1.6.2/api/generated/mdtraj.formats.DCDTrajectoryFile.html

    Inputs:
        1. f_gsd (str): file name of input gsd file

        2. f_dcd (str): file name of input dcd file

        3. ndim (int): dimensionality

    Return:
        list of single snapshot information
    """

    if f_gsd[0].configuration.dimensions != ndim:
        logger.info("---*Warning*: Wrong dimension information given---")
        return None

    # -----------------read gsd file-------------------------
    snapshots = []
    for onesnapshot in f_gsd:
        # ------------configuration information---------------
        boxlength = onesnapshot.configuration.box[:ndim]
        hmatrix = np.diag(boxlength)
        # ------------particles information-------------------
        positions = onesnapshot.particles.position[:, :ndim]
        boxbounds = np.column_stack((positions.min(axis=0), positions.max(axis=0)))
        snapshot = SingleSnapshot(
            timestep=onesnapshot.configuration.step,
            nparticle=onesnapshot.particles.N,
            particle_type=onesnapshot.particles.typeid + 1,
            positions=None,
            boxlength=boxlength,
            boxbounds=boxbounds,
            realbounds=None,
            hmatrix=hmatrix,
        )
        snapshots.append(snapshot)

    # -----------------read dcd file-------------------------
    positions = f_dcd.read()[0]
    # ----------------gsd and dcd should be consistent--------
    if len(snapshots) != positions.shape[0]:
        logger.error("---*Warning*: Inconsistent configuration in gsd and dcd files---")
        return None

    if snapshots[0].nparticle != positions[0].shape[0]:
        logger.error("---*Warning*: Inconsistent particle number in gsd and dcd files---")
        return None

    for i in range(positions.shape[0]):
        snapshots[i].positions = positions[i][:, :ndim]

    return Snapshots(nsnapshots=len(f_gsd), snapshots=snapshots)
