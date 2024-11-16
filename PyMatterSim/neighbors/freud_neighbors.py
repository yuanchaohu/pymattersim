# coding = utf-8

"""see documentation @ ../../docs/neighbors.md"""

import freud
import numpy as np
import numpy.typing as npt

from ..reader.reader_utils import Snapshots
from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

# pylint: disable=dangerous-default-value
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


def convert_configuration(snapshots: Snapshots):
    """
    change particle coordinates to [-L/2, L/2] and create instances for freud

    Inputs:
        snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                  (returned by reader.dump_reader.DumpReader)
    Return:
        list_box (list): the list of box information for freud analysis
        list_points (list): the list of particle positions for freud analysis
    """
    list_box = []
    list_points = []
    for snapshot in snapshots.snapshots:
        if snapshot.boxbounds.sum() != 0:
            shiftfactor = snapshot.boxbounds[:, 0] + snapshot.boxlength / 2
            points = snapshot.positions - shiftfactor[np.newaxis, :]
        else:
            points = snapshot.positions

        # pad 0 for z coordinates
        if snapshot.positions.shape[1] == 2:
            points = np.hstack((points, np.zeros((snapshot.nparticle, 1))))

        list_points.append(points)
        list_box.append(freud.box.Box.from_box(snapshot.boxlength))

    return list_box, list_points


def cal_neighbors(snapshots: Snapshots, outputfile: str = None) -> None:
    """
    calculate the particle neighbors and bond properties from freud

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader)
        2. outputfile: filename of neighbor list and bond info,
                       such as edge length (2D) or facearea(3D)

    Returns: None [saved to file]
             for neighbor list, file with name outputfile+'.neighbor.dat',
             for edgelength list (2D box), file with name outputfile+'.edgelength.dat'
             for facearea list (3D box), file with name outputfile+'.facearea.dat'
    """
    logger.info("Start calculating neighbors by freud")

    list_box, list_points = convert_configuration(snapshots)

    foverall = open(outputfile + ".overall.dat", "w", encoding="utf-8")
    foverall.write("id cn area_or_volume\n")
    fneighbors = open(outputfile + ".neighbor.dat", "w", encoding="utf-8")

    ndim = snapshots.snapshots[0].positions.shape[1]
    if ndim == 2:
        fbondinfos = open(outputfile + ".edgelength.dat", "w", encoding="utf-8")
    else:
        fbondinfos = open(outputfile + ".facearea.dat", "w", encoding="utf-8")

    for n in range(snapshots.nsnapshots):
        # write header for each configuration
        fneighbors.write("id   cn   neighborlist\n")
        if ndim == 2:
            fbondinfos.write("id   cn   edgelengthlist\n")
        else:
            fbondinfos.write("id   cn   facearealist\n")

        # calculation
        box, points = list_box[n], list_points[n]
        voro = freud.locality.Voronoi()
        voro.compute((box, points))
        # change to particle ID
        nlist = np.array(voro.nlist) + 1
        weights = voro.nlist.weights
        volumes = voro.volumes

        # output
        unique, counts = np.unique(nlist[:, 0], return_counts=True)
        nn = 0
        for i in range(unique.shape[0]):
            atomid = unique[i]
            if (atomid != nlist[nn, 0]) or (i + 1 != atomid):
                raise ValueError("neighbor list not sorted")
            i_cn = counts[i]
            fneighbors.write("%d %d " % (atomid, i_cn))
            fbondinfos.write("%d %d " % (atomid, i_cn))
            foverall.write("%d %d %.6f\n" % (atomid, i_cn, volumes[i]))
            for _ in range(i_cn):
                fneighbors.write("%d " % nlist[nn, 1])
                fbondinfos.write("%.6f " % weights[nn])
                nn += 1
            fneighbors.write("\n")
            fbondinfos.write("\n")
    fneighbors.close()
    fbondinfos.close()
    foverall.close()

    logger.info("Finish calculating neighbors by freud")


# TODO @Yibang please benchmark with
# https://github.com/yuanchaohu/MyCodes/blob/master/CalfromFreud.py#L122


def VolumeMatrix(
    snapshots: Snapshots,
    ndim: int = 2,
    nconfig: int = 0,
    deltar: float = 0.01,
    transform_matrix: bool = True,
    outputfile: str = "",
) -> npt.NDArray:
    """
    Calculate the Voronoi volume matrix for real-space vector decomposition

    Inputs:
        1. snapshots (reader.reader_utils.Snapshots): snapshot object of input trajectory
                     (returned by reader.dump_reader.DumpReader),
                     ONLY consider the first snapshot
        2. ndim (int): dimensionality of the input trajectory
        3. nconfig (int): calcualte for the specified configuration by its index nconfig
        4. deltar (float): minimal movement distance to perturb volume change
        5. transform_matrix (bool): whether to transform the volume matrix internally
        6. outputfile (str): output file name of the calculated volume matrix,
                                no output written if None

    Return:
        local volume matrix or transformed matrix in numpy ndarray
    """
    list_box, list_points = convert_configuration(snapshots)
    logger.info(f"Calculate the Voronoi volume matrix for configuration No.{nconfig}")
    box = list_box[nconfig]
    points = list_points[nconfig]
    num_particles = points.shape[nconfig]
    matrixA = np.zeros((num_particles, num_particles * ndim))

    # original voronoi volume
    voro = freud.locality.Voronoi()
    original = voro.compute((box, points)).volumes

    # perturbations by small displacement
    atomids = np.arange(num_particles, dtype=np.int32)
    for i in range(num_particles):
        condition = atomids != i
        for j in range(ndim):
            # move positive deltar
            points[i, j] += deltar
            voro = freud.locality.Voronoi()
            V1 = voro.compute((box, points)).volumes

            # move negative deltar
            points[i, j] -= 2 * deltar
            voro = freud.locality.Voronoi()
            V2 = voro.compute((box, points)).volumes

            # move back to the original position
            points[i, j] += deltar

            medium = (V1 - V2) / 2 / deltar
            matrixA[condition, ndim * i + j] = medium[condition]

    # for pair i-i
    for i in range(num_particles):
        medium = matrixA[i].reshape(num_particles, ndim)
        matrixA[i, ndim * i : ndim * i + ndim] = -medium.sum(axis=0)

    # local volume matrix
    matrixA /= original[:, np.newaxis]

    if transform_matrix:
        # matrix manipulation
        medium = np.matmul(matrixA, matrixA.T)
        medium = np.linalg.inv(medium)
        medium = np.matmul(matrixA.T, medium)
        matrixA_transformation = np.matmul(medium, matrixA)
        if outputfile:
            np.save(outputfile, matrixA_transformation)
        return matrixA_transformation
    if outputfile:
        np.save(matrixA, outputfile)
    return matrixA
