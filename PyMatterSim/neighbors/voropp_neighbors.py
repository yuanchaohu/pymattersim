# coding = utf-8

"""see documentation @ ../../docs/neighbors.md"""

import os
import re
import subprocess

import numpy as np
import pandas as pd

from ..reader.reader_utils import Snapshots
from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
# pylint: disable=consider-using-with
# pylint: disable=too-many-statements
# pylint: disable=invalid-name


def get_input(snapshots: Snapshots, radii: dict = {1: 0.5, 2: 0.5}) -> tuple[list, list]:
    """
    Design input file for Voro++ by considering particle radii

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                      (returned by reader.dump_reader.DumpReader)
        2. radii (dict): radii of particles, must be a dict like {1 : 1.28, 2 : 1.60}
                         if you do not want to consider radii, set the radii the same
    Return:
        1. position (list): input file for Voro++ with the format:
                            particle_ID x_coordinate y_coordinate z_coordinate radius
        2. bounds (list): box bounds for snapshots
    """
    position = []
    for snapshot in snapshots.snapshots:
        particle_radii = np.array(pd.Series(snapshot.particle_type).map(radii))
        position_radii = np.column_stack((snapshot.positions, particle_radii))
        voroinput = np.column_stack((np.arange(snapshot.nparticle) + 1, position_radii))
        position.append(voroinput)
    bounds = [snapshot.boxbounds for snapshot in snapshots.snapshots]

    return position, bounds


def cal_voro(
    snapshots: Snapshots,
    ppp: str = "-p",
    radii: dict = {1: 0.5, 2: 0.5},
    outputfile: str = None,
) -> None:
    """
    Radical Voronoi Tessellation using voro++ for periodic boundary conditions

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. ppp (str): Make the container periodic in all three directions, default ppp='-p'
        3. radii (dict): radii of particles, must be a dict like {1 : 1.28, 2 : 1.60}
                         if you do not want to consider radii, set the radii the same
        4. outfile (str): filename of output, including neighborlist, facearealist,
                          voronoi index, overall (facearea and volume for centered particle)
    Return:
        None [saved to file]
        for neighbor list, file with name outputfile+'.neighbor.dat',
        for facearea list, file with name outputfile+'.facearea.dat',
        for voronoi index, file with name outputfile+'.voroindex.dat',
        for overall, file with name outputfile+'.overall.dat'
    """
    logger.info("Start calculating particle neighbors by voro++ with PBC")
    fneighbor = open(outputfile + ".neighbor.dat", "w", encoding="utf-8")
    ffacearea = open(outputfile + ".facearea.dat", "w", encoding="utf-8")
    findex = open(outputfile + ".voroindex.dat", "w", encoding="utf-8")
    findex.write("id   voro_index   0_to_7_faces\n")
    foverall = open(outputfile + ".overall.dat", "w", encoding="utf-8")
    foverall.write("id   cn   volume   facearea\n")

    position, bounds = get_input(snapshots, radii)
    ndim = snapshots.snapshots[0].positions.shape[1]
    for n in range(snapshots.nsnapshots):
        fileformat = "%d " + "%.6f " * ndim + "%.6f"
        np.savetxt("dumpused", position[n], fmt=fileformat)

        boxbounds = bounds[n].ravel()
        cmdline = "voro++ " + ppp + ' -r -c "%i %s %v %F @%i %A @%i %s %n @%i %s %f" ' + ("%f %f " * ndim % tuple(boxbounds)) + "dumpused"
        if n == 0:
            logger.info(f"Start calculating Voronoi for PBC by voro++, command: {cmdline}")
        try:
            subprocess.run(cmdline, shell=True, check=False)
        except BaseException:
            raise ImportError("***Please install VORO++***")

        fneighbor.write("id   cn   neighborlist\n")
        ffacearea.write("id   cn   facearealist\n")
        f = open("dumpused.vol", "r", encoding="utf-8")
        for _ in range(len(position[n][:, 0])):
            item = f.readline().split("@")
            foverall.write(item[0] + "\n")
            findex.write(item[1] + "\n")
            fneighbor.write(item[2] + "\n")
            ffacearea.write(item[3])
        f.close()

    os.remove("dumpused")  # delete temporary files
    os.remove("dumpused.vol")
    fneighbor.close()
    ffacearea.close()
    foverall.close()
    findex.close()
    logger.info("Finish calculating Voronoi for PBC by voro++")


def voronowalls(
    snapshots: Snapshots,
    ppp: str,
    radii: dict = {1: 0.5, 2: 0.5},
    outputfile: str = None,
) -> None:
    """
    Radical Voronoi Tessellation using voro++ for non-periodic boundary conditions
    Output Results by Removing Artifacial Walls

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. ppp (str): Make the container periodic in a desired direction
                      '-px', '-py', and '-pz' for x, y, and z directions, respectively
        3. radii (dict): radii of particles, must be a dict like {1 : 1.28, 2 : 1.60}
                         if you do not want to consider radii, set the radii the same
        4. outfile (str): filename of output, including neighborlist, facearealist,
                          voronoi index, overall (facearea and volume for centered particle)
    Return:
        None [saved to file]
        for neighbor list, file with name outputfile+'.neighbor.dat',
        for facearea list, file with name outputfile+'.facearea.dat',
        for voronoi index, file with name outputfile+'.voroindex.dat',
        for overall, file with name outputfile+'.overall.dat'
    """
    logger.info("Start calculating particle neighbors by voro++ without PBC")
    fneighbor = open(outputfile + ".neighbor.dat", "w", encoding="utf-8")
    ffacearea = open(outputfile + ".facearea.dat", "w", encoding="utf-8")
    findex = open(outputfile + ".voroindex.dat", "w", encoding="utf-8")
    findex.write("id   voro_index   0_to_7_faces\n")
    foverall = open(outputfile + ".overall.dat", "w", encoding="utf-8")
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    foverall.write("id   cn   volume   facearea\n")

    position, bounds = get_input(snapshots, radii)
    ndim = snapshots.snapshots[0].positions.shape[1]
    for n in range(snapshots.nsnapshots):
        fileformat = "%d " + "%.6f " * ndim + "%.6f"
        np.savetxt("dumpused", position[n], fmt=fileformat)

        Boxbounds = bounds[n].ravel()
        cmdline = "voro++ " + ppp + ' -r -c "%i %s %v %F @%i %A @%i %s %n @%i %s %f" ' + ("%f %f " * ndim % tuple(Boxbounds)) + "dumpused"
        if n == 0:
            logger.info(f"Start calculating Voronoi for non-PBC by voro++, command: {cmdline}")
        subprocess.run(cmdline, shell=True, check=False)

        fneighbor.write("id   cn   neighborlist\n")
        ffacearea.write("id   cn   facearealist\n")
        f = open("dumpused.vol", "r", encoding="utf-8")
        for _ in range(len(position[n][:, 0])):
            item = f.readline().split("@")

            medium = [int(j) for j in item[2].split()]
            mneighbor = np.array(medium, dtype=np.int32)
            neighbor = mneighbor[mneighbor > 0]
            neighbor[1] = len(neighbor[2:])

            medium = [float(j) for j in item[3].split()]
            facearea = np.array(medium)
            facearea = facearea[mneighbor > 0]
            facearea[1] = neighbor[1]

            medium = [float(j) for j in item[0].split()]
            overall = np.array(medium)
            overall[1] = neighbor[1]
            overall[3] = facearea[2:].sum()

            # -----write Overall results-----
            np.savetxt("temp", overall[np.newaxis, :], fmt="%d %d %.6f %.6f")
            with open("temp", "r", encoding="utf-8") as temp:
                foverall.write(temp.read())
            # -----write voronoi index-------
            findex.write(item[1] + "\n")
            # -----write facearea list-------
            np.savetxt("temp", facearea[np.newaxis, :], fmt="%d " * 2 + "%.6f " * neighbor[1])
            with open("temp", "r", encoding="utf-8") as temp:
                ffacearea.write(temp.read())
            # -----write neighbor list-------
            fneighbor.write(re.sub(r"[\[\]]", " ", np.array2string(neighbor) + "\n"))
        f.close()

    os.remove("dumpused")  # delete temporary files
    os.remove("dumpused.vol")
    os.remove("temp")
    fneighbor.close()
    ffacearea.close()
    foverall.close()
    findex.close()
    logger.info("Finish calculating Voronoi for non-PBC by voro++")


def indicehis(inputfile: str, outputfile: str = None) -> None:
    """
    Statistics the frequency of voronoi index from the output of voronoi analysis
    Only the top 50 voronoi index will be output along with their fractions

    Inputs:
        inputfile (str): the filename of saved Voronoi index
    Return:
        None [saved to file]
        Frequency of Voronoi Indices
    """

    logger.info("Getting the statistics of Voronoi clusters")
    with open(inputfile, "r", encoding="utf-8") as f:
        totaldata = len(f.readlines()) - 1

    f = open(inputfile, "r", encoding="utf-8")
    f.readline()
    # totaldata = SnapshotNumber * ParticleNumber
    medium = np.zeros((totaldata, 15), dtype=np.int32)
    for n in range(totaldata):
        item = f.readline().split()[1:]
        medium[n, : len(item)] = item  # [int(j) for j in item]

    medium = medium[:, 3:7]  # <n3 n4 n5 n6>
    indices, counts = np.unique(medium, return_counts=True, axis=0)
    sort_indices = np.argsort(counts)[::-1]
    freq_indices = counts / totaldata  # / ParticleNumber / SnapshotNumber
    results = np.column_stack((indices[sort_indices], freq_indices[sort_indices]))
    fformat = "%d " * medium.shape[1] + "%.6f "
    names = "Voronoi Indices,  Frequency"
    if outputfile:
        np.savetxt(outputfile, results, fmt=fformat, header=names, comments="")
    f.close()

    logger.info("Now you have the fractions of Voronoi clusters")
