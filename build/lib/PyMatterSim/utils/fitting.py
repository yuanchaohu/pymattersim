"""see documentation @ ../../docs/utils.md"""

from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from ..utils.logging import get_logger_handle

logger = get_logger_handle(__name__)
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=line-too-long
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace


def fits(
    fit_func: Any,
    xdata: npt.NDArray,
    ydata: npt.NDArray,
    rangea: float = 0,
    rangeb: float = 0,
    p0: List[float] = [],
    bounds: Tuple[List[float]] = (),
    max_iterations: int = 5000000,
    style="linear",
) -> List[Any]:
    """
    This function is used to fit existing data in numpy array
    A fitting function and (X, Y) should be provided

    Inputs:
        1. fit_func: fitting function
        2. xdata (npt.NDArray): x-variable
        3. ydata (npt.NDArray): y-variable
        4. rangea (float): lower x-limit of fitting result, default 0 (False)
        5. rangeb (float): upper x-limit of fitting result, default 0 (False)
        6. p0 (list of float): initial values of the fit_func arguments
        7. bounds (tuple of list): lower and upper bounds of the fit_func arguments
                                   each in a list
        8. max_iterations (int): maximum iteration numbers for curve fit, default 5000000
        9. style (str): scale of returned fitting results of x,
                        select from 'linear'(default) or 'log'

    Return:
        list of fitting results, such as
        popt, perr, xfit, yfit
        popt: list of fitting parameter values
        perr: list of error bar of popt
        xfit: fitting result of x
        yfit: fitting result of y
    """
    if len(p0) >= 1 and len(bounds) >= 1:
        popt, pcov = curve_fit(fit_func, xdata, ydata, maxfev=max_iterations, p0=p0, bounds=bounds)
    elif len(p0) >= 1 and len(bounds) == 0:
        popt, pcov = curve_fit(fit_func, xdata, ydata, maxfev=max_iterations, p0=p0)
    elif len(p0) == 0 and len(bounds) >= 1:
        popt, pcov = curve_fit(fit_func, xdata, ydata, maxfev=max_iterations, bounds=bounds)
    else:
        popt, pcov = curve_fit(fit_func, xdata, ydata, maxfev=5000000)

    perr = np.sqrt(np.diag(pcov))
    residuals = ydata - fit_func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.square(ydata - ydata.mean()).sum()
    R2 = 1 - (ss_res / ss_tot)
    logger.info("fitting R^2 = %.6f" % R2)
    logger.info("fitting parameters values: " + " ".join(map("{:.6f}".format, popt)))
    logger.info("fitting parameters errors: " + " ".join(map("{:.6f}".format, perr)))

    if rangeb == 0:
        if style == "log":
            xfit = np.geomspace(xdata.min(), xdata.max(), 10000)
        else:
            xfit = np.linspace(xdata.min(), xdata.max(), 10000)
    else:
        if style == "log":
            xfit = np.geomspace(rangea, rangeb, 10000)
        else:
            xfit = np.linspace(rangea, rangeb, 10000)

    yfit = fit_func(xfit, *popt)
    return [popt, perr, xfit, yfit]
