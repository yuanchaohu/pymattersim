# coding = utf-8

import numpy as np
import freud

from reader.reader_utils import Snapshots
from utils.logging_utils import get_logger_handle

logger = get_logger_handle(__name__)


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


def cal_neighbors(snapshots: Snapshots, outputfile: str='') -> None:
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

    foverall = open(outputfile+'.overall.dat', 'w', encoding="utf-8")
    foverall.write('id cn area_or_volume\n')
    fneighbors = open(outputfile+'.neighbor.dat', 'w', encoding="utf-8")

    ndim = snapshots.snapshots[0].positions.shape[1]
    if ndim == 2:
        fbondinfos = open(outputfile+'.edgelength.dat', 'w', encoding="utf-8")
    else:
        fbondinfos = open(outputfile+'.facearea.dat', 'w', encoding="utf-8")

    for n in range(snapshots.nsnapshots):
        # write header for each configuration
        fneighbors.write('id   cn   neighborlist\n')
        if ndim == 2:
            fbondinfos.write('id   cn   edgelengthlist\n')
        else:
            fbondinfos.write('id   cn   facearealist\n')

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
                raise ValueError('neighbor list not sorted')
            i_cn = counts[i]
            fneighbors.write('%d %d ' % (atomid, i_cn))
            fbondinfos.write('%d %d ' % (atomid, i_cn))
            foverall.write('%d %d %.6f\n' % (atomid, i_cn, volumes[i]))
            for j in range(i_cn):
                fneighbors.write('%d ' %nlist[nn, 1])
                fbondinfos.write('%.6f '% weights[nn])
                nn += 1
            fneighbors.write('\n')
            fbondinfos.write('\n')
    fneighbors.close()
    fbondinfos.close()
    foverall.close()

    logger.info("Finish calculating neighbors by freud")
