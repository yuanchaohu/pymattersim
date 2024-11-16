
"""see documentation @ ../../docs/orderings.md"""

from typing import Any, List

import numpy as np
import numpy.typing as npt

from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)


def gyration_tensor(pos_group: npt.NDArray) -> List[Any]:
    """calculate gyration tensor of a cluster of atoms

    This module calculates gyration tensor which is a tensor that describes
    the second moments of posiiton of a collection of particles

    gyration tensor is a symmetric matrix of shape (ndim, ndim)
    ref: https://en.wikipedia.org/wiki/Gyration_tensor

    a group of atoms should be first defined
    groupofatoms are the original coordinates of the selected group
    of a single configuration,

    the atom coordinates of the cluster should be removed from PBC
    which can be realized by ovito 'cluster analysis' method
    by choosing 'unwrap particle coordinates'

    Input:
        1. pos_group (npt.NDArray): unwrapped particle positions of a
            group of atoms, shape as [num_of_particles, dimensionality]

    Return:
        shape factors in a list, such as
        3D: radius_of_gyration, asphericity, acylindricity, shape_anisotropy, fractal_dimension
        2D: radius_of_gyration, acylindricity, fractal_dimension
    """

    (num_particles, ndim) = pos_group.shape
    logger.info(f"Calculating gyration tensor of {num_particles} atoms in {ndim} - dimension")

    # shift the original coordinates to be centered at (0,0,0)
    center_of_mass = pos_group.mean(axis=0)[np.newaxis, :]
    pos_group -= center_of_mass

    if ndim == 3:
        # 0, 1, 2 = x, y, z
        combinations = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    elif ndim == 2:
        combinations = [(0, 0), (0, 1), (1, 1)]
    else:
        raise ValueError("Wrong input dimensionality")

    # calculate the tensor
    results = np.zeros((ndim, ndim))
    for (m, n) in combinations:
        Smn = 0
        for i in range(num_particles):
            Smn += pos_group[i, m] * pos_group[i, n]
        results[m, n] = Smn / num_particles
        results[n, m] = Smn / num_particles

    # calculate shape descriptors
    principal_component = np.sort(np.linalg.eig(results)[0])

    radius_of_gyration = np.sqrt(principal_component.sum())
    acylindricity = principal_component[1] - principal_component[0]
    fractal_dimension = np.log10(num_particles) / np.log10(radius_of_gyration)

    if ndim == 3:
        asphericity = 1.5 * \
            principal_component[2] - 0.5 * principal_component.sum()
        shape_anisotropy = (asphericity ** 2 + 0.75 *
                            acylindricity ** 2) / radius_of_gyration**4
        logger.info(
            f"""
            Calculated shape factors are:
                radius_of_gyration: {radius_of_gyration},
                asphericity: {asphericity},
                acylindricity: {acylindricity},
                shape_anisotropy: {shape_anisotropy},
                fractal_dimension: {fractal_dimension}
            """
        )
        return [
            radius_of_gyration,
            asphericity,
            acylindricity,
            shape_anisotropy,
            fractal_dimension]
    else:
        logger.info(
            f"""
        Calculated shape factors are:
            radius_of_gyration: {radius_of_gyration},
            acylindricity: {acylindricity},
            fractal_dimension: {fractal_dimension}
        """
        )
        return [radius_of_gyration, acylindricity, fractal_dimension]
