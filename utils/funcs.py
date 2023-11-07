# coding = utf-8

"""small mathmatical functions for feasible computation"""

def nidealfac(ndim: int=3) -> float:
    """
    Choose factor of Nideal in g(r) calculation
    
    Inputs:
        ndim (int): system dimensionality, default 3

    Return:
        (float): Nideal
    """
    if ndim == 3:
        return 4.0 / 3
    elif ndim == 2:
        return 1.0 