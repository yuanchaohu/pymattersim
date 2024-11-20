# coding = utf-8

"""see documentation @ ../../docs/utils.md"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)

# pylint: disable=invalid-name
# pylint: disable=too-many-locals


def Filon_COS(C: npt.NDArray, t: npt.NDArray, a: float = 0, outputfile: str = "") -> pd.DataFrame:
    """
    This module calculates the Fourier transformation of an autocorrelation function
    by Filon's integration method

    Inputs:
    1. C (npt.NDArray): the auto-correlation function
    2. t (npt.NDArray): the time corresponding to C
    3. a (float): the frequency interval, default 0
    4. outputfile (str): filename to save the calculated results

    Return:
        FFT results (pd.DataFrame)
    """
    logger.info("Calculate the Fast Fourier Transformation using Filon COS method")

    if len(C) % 2 == 0:
        logger.info("Warning: number of input data is not odd")
        C = C[:-1]
        t = t[:-1]

    if a == 0:  # a is not specified
        a = 2 * np.pi / t[-1]

    Nmax = len(C)
    dt = round(t[1] - t[0], 3)
    if dt != round(t[-1] - t[-2], 3):
        raise ValueError("time is not evenly distributed")

    results = pd.DataFrame(0, index=range(Nmax), columns="omega FFT".split()).astype("float64")
    for n in range(Nmax):
        omega = n * a
        results.iloc[n, 0] = omega

        # calculate the filon parameters
        theta = omega * dt
        theta2 = theta * theta
        theta3 = theta * theta2
        if theta == 0:
            alpha = 0.0
            beta = 2.0 / 3.0
            gamma = 4.0 / 3.0
        else:
            alpha = 1.0 / theta + np.sin(2 * theta) / 2.0 / theta2 - 2.0 * np.sin(theta) * np.sin(theta) / theta3
            beta = 2.0 * ((1 + np.cos(theta) * np.cos(theta)) / theta2 - np.sin(2 * theta) / theta3)
            gamma = 4.0 * (np.sin(theta) / theta3 - np.cos(theta) / theta2)

        C_even = 0
        for i in range(0, Nmax, 2):
            C_even += C[i] * np.cos(omega * i * dt)
        C_even -= 0.5 * (C[-1] * np.cos(omega * t[-1]) + C[0] * np.cos(omega * t[0]))

        C_odd = 0
        for i in range(1, Nmax - 1, 2):
            C_odd += C[i] * np.cos(omega * i * dt)

        results.iloc[n, 1] = 2.0 * dt * (alpha * (C[-1] * np.sin(omega * t[-1]) - C[0] * np.sin(omega * t[0])) + beta * C_even + gamma * C_odd)

    results["FFT"] /= np.pi
    if outputfile:
        results.to_csv(outputfile, float_format="%.6f", index=False)
    return results
